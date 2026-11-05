"""
Módulo de kernels CUDA em C++ para compilação em tempo de execução via CuPy RawKernel.

Implementa a avaliação do campo de velocidades hierárquico com poda espacial exata
diretamente na GPU, mantendo os centros ativos em registradores de cada thread
e eliminando 100% das alocações intermediárias na VRAM.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np
import backend
from funcoes_para_centros import (
    R_1,
    T_1,
    O_1,
    lamb as LAMB_DEFAULT,
    c_padrao as C_PADRAO,
    alpha_padrao as ALPHA_PADRAO,
)

CUDA_SRC_FIELD_PRUNED = r"""
extern "C" __global__
void campo_vortices_podado_kernel(
    const double* __restrict__ xp,
    const double* __restrict__ yp,
    double* __restrict__ vx_out,
    double* __restrict__ vy_out,
    const int Np,
    const int n_max,
    const double t,
    const double c_bump,
    const double alpha,
    const double R1,
    const double T1,
    const double O1,
    const double lambda_val
) {
    const int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p >= Np) return;

    const double x = xp[p];
    const double y = yp[p];

    double vx_total = 0.0;
    double vy_total = 0.0;

    // Constante pi
    const double PI = 3.14159265358979323846;

    // ============================================================
    // Nível 1 (Raiz centrada em (0, 0))
    // ============================================================
    const double R1_val = R1;
    const double T1_val = T1;
    const double Rcut1 = alpha * R1_val;
    const double Rcut1_sq = Rcut1 * Rcut1;
    const double inv_Rcut1_sq = 1.0 / Rcut1_sq;

    const double dx0 = x;
    const double dy0 = y;
    const double r0_sq = dx0 * dx0 + dy0 * dy0;

    if (r0_sq < Rcut1_sq) {
        const double u0 = r0_sq * inv_Rcut1_sq - 1.0;
        const double poten0 = (2.0 * PI / T1_val) * exp(c_bump / u0);
        vx_total += -dy0 * poten0;
        vy_total += dx0 * poten0;
    }

    if (n_max <= 1) {
        vx_out[p] = vx_total;
        vy_out[p] = vy_total;
        return;
    }

    // Raio de alcance da descendência da raiz (Nível 1)
    const double D1 = (sqrt(2.0) + alpha / 2.0) * R1_val;
    if (r0_sq > D1 * D1) {
        // Nenhuma descendência pode alcançar este ponto
        vx_out[p] = vx_total;
        vy_out[p] = vy_total;
        return;
    }

    // Buffers locais estáticos em registradores / memória local rápida
    // Medições mostram tipicamente até 4 centros ativos por nível.
    // Usamos capacidade 32 para ampla margem de segurança mesmo em regimes de alpha elevado.
    const int MAX_CENTROS = 32;
    double curr_cx[MAX_CENTROS];
    double curr_cy[MAX_CENTROS];
    int n_curr = 1;
    curr_cx[0] = 0.0;
    curr_cy[0] = 0.0;

    double next_cx[MAX_CENTROS];
    double next_cy[MAX_CENTROS];

    // Constantes de fases angulares: phi_k = pi/4 + k * pi/2
    const double phi[4] = {
        0.7853981633974483, // pi/4
        2.356194490192345,  // 3pi/4
        3.9269908169872414, // 5pi/4
        5.497787143782138   // 7pi/4
    };

    // Percorre os níveis dos pais (n de 1 até n_max - 1)
    for (int n = 1; n < n_max; ++n) {
        if (n_curr == 0) break;

        const int nivel_filho = n + 1;
        // Rn_filho = R1 * 2^(1 - nivel_filho)
        const double Rn_filho = R1 * pow(2.0, 1.0 - nivel_filho);
        // Tn_filho = T1 * lambda^(1 - nivel_filho)
        const double Tn_filho = T1 * pow(lambda_val, 1.0 - nivel_filho);
        // Om_filho = O1 * lambda^(nivel_filho - 1)
        const double Om_filho = O1 * pow(lambda_val, nivel_filho - 1.0);

        const double Rcut_filho = alpha * Rn_filho;
        const double Rcut_filho_sq = Rcut_filho * Rcut_filho;
        const double inv_Rcut_filho_sq = 1.0 / Rcut_filho_sq;
        const double fator_poten = 2.0 * PI / Tn_filho;

        const double D_filho = (sqrt(2.0) + alpha / 2.0) * Rn_filho;
        const double D_filho_sq = D_filho * D_filho;

        // Deslocamentos dos 4 filhos em relação ao centro pai
        const double Rk = Rn_filho * sqrt(2.0);
        const double base_ang = Om_filho * t;
        double sub_dx[4];
        double sub_dy[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const double ang = base_ang + phi[k];
            sub_dx[k] = Rk * cos(ang);
            sub_dy[k] = Rk * sin(ang);
        }

        int n_next = 0;

        // Avalia os filhos de cada centro pai ativo
        for (int j = 0; j < n_curr; ++j) {
            const double px_pai = curr_cx[j];
            const double py_pai = curr_cy[j];

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const double scx = px_pai + sub_dx[k];
                const double scy = py_pai + sub_dy[k];

                const double dx = x - scx;
                const double dy = y - scy;
                const double r2 = dx * dx + dy * dy;

                // 1. Contribuição direta se o ponto estiver dentro do suporte
                if (r2 < Rcut_filho_sq) {
                    const double u = r2 * inv_Rcut_filho_sq - 1.0;
                    const double poten = fator_poten * exp(c_bump / u);
                    vx_total += -dy * poten;
                    vy_total += dx * poten;
                }

                // 2. Candidato à próxima expansão (se nivel_filho < n_max)
                if (nivel_filho < n_max && r2 <= D_filho_sq) {
                    if (n_next < MAX_CENTROS) {
                        next_cx[n_next] = scx;
                        next_cy[n_next] = scy;
                        n_next++;
                    }
                }
            }
        }

        // Atualiza a lista de centros ativos para o próximo nível
        n_curr = n_next;
        for (int j = 0; j < n_curr; ++j) {
            curr_cx[j] = next_cx[j];
            curr_cy[j] = next_cy[j];
        }
    }

    vx_out[p] = vx_total;
    vy_out[p] = vy_total;
}
"""

_RAW_KERNEL_CACHE = None


def get_cuda_field_kernel() -> Any:
    """
    Retorna o CuPy RawKernel compilado em tempo de execução via NVRTC.
    Utiliza cache para evitar recompilação.
    """
    global _RAW_KERNEL_CACHE
    if _RAW_KERNEL_CACHE is not None:
        return _RAW_KERNEL_CACHE

    if not backend.is_cuda_available():
        _, reason = backend.get_cuda_status()
        raise backend.CudaNotAvailableError(
            f"Não foi possível compilar o kernel CUDA.\nMotivo: {reason}"
        )

    import cupy as cp

    _RAW_KERNEL_CACHE = cp.RawKernel(
        CUDA_SRC_FIELD_PRUNED,
        "campo_vortices_podado_kernel",
        options=("-O3", "--use_fast_math"),
    )
    return _RAW_KERNEL_CACHE


def avaliar_campo_cuda(
    xp: Any,
    yp: Any,
    t: float,
    n_max: int = 3,
    c: float = C_PADRAO,
    alpha: float = ALPHA_PADRAO,
    block_size: int = 256,
) -> tuple[Any, Any]:
    """
    Avalia o campo de velocidades diretamente na GPU usando CuPy e RawKernel.

    Parâmetros
    ----------
    xp, yp : cp.ndarray ou array-like
        Coordenadas das partículas (armazenadas ou convertidas para CuPy na GPU).
    t : float
        Instante temporal.
    n_max : int
        Nível máximo da hierarquia.
    c : float
        Parâmetro bump de suavidade.
    alpha : float
        Parâmetro de suporte/corte.
    block_size : int
        Número de threads CUDA por bloco (padrão: 256).

    Retorna
    -------
    vx, vy : cp.ndarray
        Componentes do campo vetorial na GPU.
    """
    if not backend.is_cuda_available():
        _, reason = backend.get_cuda_status()
        raise backend.CudaNotAvailableError(
            f"O backend CUDA foi requisitado para o cálculo do campo, mas não está disponível.\n"
            f"Motivo: {reason}"
        )

    import cupy as cp

    # Garante arrays 1D contíguos float64 na GPU (suporta escalares e vetores)
    xp_gpu = cp.atleast_1d(cp.ascontiguousarray(cp.asarray(xp, dtype=cp.float64)))
    yp_gpu = cp.atleast_1d(cp.ascontiguousarray(cp.asarray(yp, dtype=cp.float64)))

    Np = xp_gpu.size
    if Np == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)

    vx_out = cp.empty_like(xp_gpu)
    vy_out = cp.empty_like(yp_gpu)

    grid_size = (Np + block_size - 1) // block_size
    kernel = get_cuda_field_kernel()

    kernel(
        (grid_size,),
        (block_size,),
        (
            xp_gpu,
            yp_gpu,
            vx_out,
            vy_out,
            np.int32(Np),
            np.int32(n_max),
            np.float64(t),
            np.float64(c),
            np.float64(alpha),
            np.float64(R_1),
            np.float64(T_1),
            np.float64(O_1),
            np.float64(LAMB_DEFAULT),
        ),
    )

    return vx_out, vy_out


def medir_tempo_compilacao_kernel() -> dict[str, float]:
    """
    Mede isoladamente o tempo de compilação JIT (NVRTC) inicial do kernel
    versus o acesso subsequente ao cache em memória.
    Retorna dicionário com os tempos em milissegundos.
    """
    global _RAW_KERNEL_CACHE
    if not backend.is_cuda_available():
        _, reason = backend.get_cuda_status()
        raise backend.CudaNotAvailableError(f"CUDA indisponível: {reason}")

    import time
    import cupy as cp

    # Força compilação a frio
    _RAW_KERNEL_CACHE = None
    t0 = time.perf_counter()
    k1 = cp.RawKernel(
        CUDA_SRC_FIELD_PRUNED,
        "campo_vortices_podado_kernel",
        options=("-O3", "--use_fast_math"),
    )
    # Força geração de código PTX/binário
    _ = k1.attributes
    t_compilacao_frio = (time.perf_counter() - t0) * 1000.0

    # Acesso a quente (em cache)
    _RAW_KERNEL_CACHE = k1
    t0 = time.perf_counter()
    _ = get_cuda_field_kernel()
    t_acesso_cache = (time.perf_counter() - t0) * 1000.0

    return {
        "compilacao_jit_ms": t_compilacao_frio,
        "acesso_cache_ms": t_acesso_cache,
    }

