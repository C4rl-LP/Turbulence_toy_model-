"""
Suíte de Benchmarks Reproduzíveis de Desempenho: CPU vs GPU.

Mede de forma reprodutiva e isolada:
1. Tempo de avaliação do campo de velocidades (por chamada e por partícula).
2. Tempo de integração das trajetórias (RK4 determinístico e Euler-Maruyama estocástico).
3. Taxa de processamento (partículas-passos por segundo).
4. Custo de transferência CPU-GPU (quando CUDA disponível).
5. Aceleração relativa (speedup) e uso de memória.
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import tracemalloc
import numpy as np

import backend
import campos
import integradores as it


def benchmark_campo(
    lista_Np: list[int] = [10, 100, 500, 1000, 2000],
    lista_n_max: list[int] = [3, 6, 9],
    n_repeticoes: int = 10,
    seed: int = 42,
    backend_name: str = "auto",
) -> list[dict]:
    """
    Executa benchmark da avaliação do campo de velocidades para diferentes
    números de partículas Np e níveis n_max.
    """
    rng = np.random.default_rng(seed)
    cuda_disp = backend.is_cuda_available()
    if backend_name == "cuda" and not cuda_disp:
        raise backend.CudaNotAvailableError("Backend CUDA solicitado mas indisponível.")
    tem_cuda = (backend_name in ("auto", "cuda")) and cuda_disp
    resultados = []

    print("\n" + "=" * 80)
    print(" 1. BENCHMARK DE AVALIAÇÃO DO CAMPO DE VELOCIDADES")
    print(f" Ambiente: {'CUDA GPU disponível' if tem_cuda else 'CPU apenas (CUDA indisponível)'}")
    print("=" * 80)
    print(f"{'Np':>6} | {'n_max':>5} | {'CPU Ref (ms)':>12} | {'CPU Opt (ms)':>12} | {'CUDA (ms)':>10} | {'Speedup CPU':>12}")
    print("-" * 80)

    for n_max in lista_n_max:
        for Np in lista_Np:
            xp = rng.uniform(-0.5, 0.5, Np)
            yp = rng.uniform(-0.5, 0.5, Np)
            t = 0.25

            # 1. CPU Referência (campo_total_podado_2)
            # Para Np muito grande em níveis altos, reduz repetições para não travar
            reps_ref = min(n_repeticoes, 5 if Np >= 1000 else n_repeticoes)
            t0 = time.perf_counter()
            for _ in range(reps_ref):
                campos.campo_total_podado_2(xp, yp, t, n_max=n_max)
            t_ref = (time.perf_counter() - t0) / reps_ref

            # 2. CPU Otimizado (campo_total_podado_vetorizado)
            # Aquecimento
            campos.campo_total_podado_vetorizado(xp, yp, t, n_max=n_max)
            t0 = time.perf_counter()
            for _ in range(n_repeticoes):
                campos.campo_total_podado_vetorizado(xp, yp, t, n_max=n_max)
            t_opt = (time.perf_counter() - t0) / n_repeticoes

            speedup_cpu = t_ref / t_opt if t_opt > 0 else 0.0

            # 3. CUDA GPU (se disponível)
            t_cuda = None
            if tem_cuda:
                import cuda_kernels
                # Aquecimento GPU
                cuda_kernels.avaliar_campo_cuda(xp, yp, t, n_max=n_max)
                backend.synchronize(backend="cuda")

                timer = backend.DeviceTimer(backend="cuda")
                with timer:
                    for _ in range(n_repeticoes):
                        cuda_kernels.avaliar_campo_cuda(xp, yp, t, n_max=n_max)
                t_cuda = timer.elapsed_time / n_repeticoes

            str_cuda = f"{t_cuda * 1000:.3f}" if t_cuda is not None else "N/A"
            print(
                f"{Np:6d} | {n_max:5d} | {t_ref * 1000:12.3f} | {t_opt * 1000:12.3f} | "
                f"{str_cuda:>10} | {speedup_cpu:11.1f}x"
            )

            res = {
                "Np": Np,
                "n_max": n_max,
                "t_cpu_ref_ms": t_ref * 1000,
                "t_cpu_opt_ms": t_opt * 1000,
                "t_cuda_ms": t_cuda * 1000 if t_cuda is not None else None,
                "speedup_cpu_opt_vs_ref": speedup_cpu,
                "speedup_cuda_vs_ref": (t_ref / t_cuda) if t_cuda else None,
            }
            resultados.append(res)

    return resultados


def benchmark_integradores(
    lista_Np: list[int] = [50, 200, 1000],
    n_max: int = 4,
    dt: float = 0.01,
    t_max: float = 0.1,
    seed: int = 42,
    backend_name: str = "auto",
) -> list[dict]:
    """
    Executa benchmark de simulação completa de trajetórias (RK4 e Euler-Maruyama).
    Mede tempo total e taxa de partículas-passos por segundo.
    """
    r0 = np.array([0.3, 0.2333])
    cuda_disp = backend.is_cuda_available()
    if backend_name == "cuda" and not cuda_disp:
        raise backend.CudaNotAvailableError("Backend CUDA solicitado mas indisponível.")
    tem_cuda = (backend_name in ("auto", "cuda")) and cuda_disp
    resultados = []

    print("\n" + "=" * 80)
    print(" 2. BENCHMARK DE INTEGRAÇÃO DE TRAJETÓRIAS (RK4 & EULER-MARUYAMA)")
    print(f" Parâmetros: n_max={n_max}, dt={dt}, t_max={t_max} (passos: {int(t_max/dt)})")
    print("=" * 80)
    print(f"{'Método':<16} | {'Np':>6} | {'Ref (s)':>10} | {'Opt (s)':>10} | {'CUDA (s)':>10} | {'Speedup':>10}")
    print("-" * 80)

    for Np in lista_Np:
        # --- RK4 ---
        # 1. RK4 Referência
        t0 = time.perf_counter()
        it.simular_particulas(
            N=Np, N_fields=n_max, dimensao_quadrado=0.01, r0=r0,
            dt=dt, t_max=t_max, func=campos.campo_total_podado_2, seed=seed
        )
        t_rk4_ref = time.perf_counter() - t0

        # 2. RK4 Otimizado
        t0 = time.perf_counter()
        it.simular_particulas(
            N=Np, N_fields=n_max, dimensao_quadrado=0.01, r0=r0,
            dt=dt, t_max=t_max, backend_name="cpu", seed=seed
        )
        t_rk4_opt = time.perf_counter() - t0

        t_rk4_cuda = None
        if tem_cuda:
            t0 = time.perf_counter()
            it.simular_particulas(
                N=Np, N_fields=n_max, dimensao_quadrado=0.01, r0=r0,
                dt=dt, t_max=t_max, backend_name="cuda", seed=seed
            )
            backend.synchronize(backend="cuda")
            t_rk4_cuda = time.perf_counter() - t0

        str_cuda_rk4 = f"{t_rk4_cuda:.3f}" if t_rk4_cuda is not None else "N/A"
        speedup_rk4 = t_rk4_ref / t_rk4_opt
        print(f"{'RK4':<16} | {Np:6d} | {t_rk4_ref:10.3f} | {t_rk4_opt:10.3f} | {str_cuda_rk4:>10} | {speedup_rk4:9.1f}x")

        # --- Euler-Maruyama ---
        # 1. SDE Referência
        t0 = time.perf_counter()
        it.simular_particulas_ruidosa(
            N=Np, N_fields=n_max, r0=r0, dt_1=dt, t_max=t_max,
            func=campos.campo_total_podado_2, seed=seed
        )
        t_sde_ref = time.perf_counter() - t0

        # 2. SDE Otimizado
        t0 = time.perf_counter()
        it.simular_particulas_ruidosa(
            N=Np, N_fields=n_max, r0=r0, dt_1=dt, t_max=t_max,
            backend_name="cpu", seed=seed
        )
        t_sde_opt = time.perf_counter() - t0

        t_sde_cuda = None
        if tem_cuda:
            t0 = time.perf_counter()
            it.simular_particulas_ruidosa(
                N=Np, N_fields=n_max, r0=r0, dt_1=dt, t_max=t_max,
                backend_name="cuda", seed=seed
            )
            backend.synchronize(backend="cuda")
            t_sde_cuda = time.perf_counter() - t0

        str_cuda_sde = f"{t_sde_cuda:.3f}" if t_sde_cuda is not None else "N/A"
        speedup_sde = t_sde_ref / t_sde_opt
        print(f"{'Euler-Maruyama':<16} | {Np:6d} | {t_sde_ref:10.3f} | {t_sde_opt:10.3f} | {str_cuda_sde:>10} | {speedup_sde:9.1f}x")

        resultados.append({
            "Np": Np,
            "t_rk4_ref": t_rk4_ref,
            "t_rk4_opt": t_rk4_opt,
            "t_rk4_cuda": t_rk4_cuda,
            "t_sde_ref": t_sde_ref,
            "t_sde_opt": t_sde_opt,
            "t_sde_cuda": t_sde_cuda,
        })

    return resultados


def benchmark_transferencias(
    lista_Np: list[int] = [10_000, 100_000, 1_000_000],
    repeticoes: int = 5,
    backend_name: str = "auto",
) -> list[dict]:
    """
    Mede a latência e largura de banda efetiva das transferências de dados:
    - Host-to-Device (CPU -> GPU)
    - Device-to-Host (GPU -> CPU)
    """
    cuda_disp = backend.is_cuda_available()
    if backend_name == "cuda" and not cuda_disp:
        raise backend.CudaNotAvailableError("Backend CUDA solicitado mas indisponível.")
    tem_cuda = (backend_name in ("auto", "cuda")) and cuda_disp

    print("\n" + "=" * 80)
    print(" 3. BENCHMARK DE TRANSFERÊNCIA DE DADOS CPU <-> GPU")
    print(f" Ambiente: {'CUDA GPU disponível' if tem_cuda else 'CPU apenas (CUDA indisponível - transferência simulada)'}")
    print("=" * 80)
    print(f"{'Np':>10} | {'Tamanho (MB)':>12} | {'H2D (ms)':>12} | {'H2D (GB/s)':>12} | {'D2H (ms)':>12} | {'D2H (GB/s)':>12}")
    print("-" * 80)

    resultados = []
    if not tem_cuda:
        print("  [INFO] Backend CUDA não está disponível neste ambiente. Medição suspensa.")
        return resultados

    import cupy as cp

    for Np in lista_Np:
        # Cada partícula possui 2 coordenadas float64 (16 bytes por partícula)
        arr_cpu = np.random.uniform(-1.0, 1.0, (Np, 2)).astype(np.float64)
        nbytes = arr_cpu.nbytes
        tamanho_mb = nbytes / (1024.0 * 1024.0)

        # 1. Warm-up
        arr_gpu = cp.asarray(arr_cpu)
        _ = arr_gpu.get()
        backend.synchronize(backend="cuda")

        # 2. Host-to-Device (CPU -> GPU)
        t_h2d_list = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            arr_gpu = cp.asarray(arr_cpu)
            backend.synchronize(backend="cuda")
            t_h2d_list.append(time.perf_counter() - t0)
        t_h2d = np.median(t_h2d_list)
        bw_h2d = (nbytes / 1e9) / t_h2d if t_h2d > 0 else 0.0

        # 3. Device-to-Host (GPU -> CPU)
        t_d2h_list = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            _ = arr_gpu.get()
            backend.synchronize(backend="cuda")
            t_d2h_list.append(time.perf_counter() - t0)
        t_d2h = np.median(t_d2h_list)
        bw_d2h = (nbytes / 1e9) / t_d2h if t_d2h > 0 else 0.0

        print(
            f"{Np:10d} | {tamanho_mb:12.2f} | {t_h2d * 1000:12.3f} | {bw_h2d:12.2f} | "
            f"{t_d2h * 1000:12.3f} | {bw_d2h:12.2f}"
        )

        resultados.append({
            "Np": Np,
            "tamanho_mb": tamanho_mb,
            "t_h2d_ms": t_h2d * 1000,
            "bw_h2d_gb_s": bw_h2d,
            "t_d2h_ms": t_d2h * 1000,
            "bw_d2h_gb_s": bw_d2h,
        })

    return resultados


def benchmark_compilacao() -> dict | None:
    """
    Mede isoladamente o tempo de compilação JIT inicial (NVRTC)
    versus recuperação em cache de memória do RawKernel.
    """
    print("\n" + "=" * 80)
    print(" 4. BENCHMARK DE COMPILAÇÃO JIT DE KERNELS CUDA (NVRTC)")
    print("=" * 80)
    if not backend.is_cuda_available():
        print("  [INFO] Backend CUDA não está disponível neste ambiente. Medição suspensa.")
        return None

    import cuda_kernels
    res = cuda_kernels.medir_tempo_compilacao_kernel()
    print(f"  Tempo de compilação JIT inicial (frio) : {res['compilacao_jit_ms']:.2f} ms")
    print(f"  Tempo de acesso ao cache (quente)       : {res['acesso_cache_ms']:.4f} ms")
    speedup = res['compilacao_jit_ms'] / res['acesso_cache_ms'] if res['acesso_cache_ms'] > 0 else 0.0
    print(f"  Speedup de cache                       : {speedup:.1f}x")
    return res


def main():
    parser = argparse.ArgumentParser(description="Suíte de benchmarks de paralelização CPU/GPU.")
    parser.add_argument(
        "--modo",
        type=str,
        default="todos",
        choices=["todos", "campo", "integradores", "transferencias", "compilacao"],
        help="Módulo a testar ('campo', 'integradores', 'transferencias', 'compilacao' ou 'todos')."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Backend computacional ('auto', 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--Np",
        type=int,
        nargs="+",
        default=None,
        help="Lista de números de partículas para benchmark do campo (ex: --Np 10 100 500)."
    )
    parser.add_argument(
        "--n_max",
        type=int,
        nargs="+",
        default=None,
        help="Lista de níveis n_max para benchmark do campo (ex: --n_max 3 6 8)."
    )
    parser.add_argument(
        "--Np_int",
        type=int,
        nargs="+",
        default=None,
        help="Lista de números de partículas para integradores (ex: --Np_int 50 200)."
    )
    parser.add_argument(
        "--n_max_int",
        type=int,
        default=4,
        help="Nível n_max para integração de trajetórias (padrão: 4)."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Passo temporal dt para integradores (padrão: 0.01)."
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=0.1,
        help="Tempo final t_max para integradores (padrão: 0.1)."
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=5,
        help="Número de repetições por medição (padrão: 5)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatória (padrão: 42)."
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="Executa versão rápida com grades reduzidas."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe apenas o plano de benchmark e configurações sem executar cálculos."
    )
    args = parser.parse_args()

    if args.rapido:
        lista_Np = args.Np if args.Np is not None else [10, 100, 500]
        lista_n_max = args.n_max if args.n_max is not None else [3, 6]
        lista_Np_int = args.Np_int if args.Np_int is not None else [50, 200]
    else:
        lista_Np = args.Np if args.Np is not None else [10, 100, 500, 1000]
        lista_n_max = args.n_max if args.n_max is not None else [3, 6, 8]
        lista_Np_int = args.Np_int if args.Np_int is not None else [50, 200, 500]

    cuda_disp, cuda_msg = backend.get_cuda_status()

    if args.dry_run:
        print("\n" + "=" * 70)
        print(" PLANO DE EXECUÇÃO DE BENCHMARK (DRY RUN - SEM CÁLCULOS)")
        print("=" * 70)
        print(f" Modo selecionado          : {args.modo}")
        print(f" Backend requisitado       : {args.backend}")
        print(f" Disponibilidade CUDA      : {'Disponível' if cuda_disp else 'Indisponível'} ({cuda_msg})")
        print(f" Repetições por ponto      : {args.repeticoes}")
        print(f" Semente aleatória         : {args.seed}")
        if args.modo in ("todos", "campo"):
            print("\n [Campo de Velocidades]")
            print(f"   - Partículas (Np)       : {lista_Np}")
            print(f"   - Níveis (n_max)        : {lista_n_max}")
        if args.modo in ("todos", "integradores"):
            print("\n [Integração de Trajetórias]")
            print(f"   - Partículas (Np_int)   : {lista_Np_int}")
            print(f"   - Nível fixo (n_max)    : {args.n_max_int}")
            print(f"   - Passo dt              : {args.dt}")
            print(f"   - Tempo t_max           : {args.t_max} (passos: {int(args.t_max / args.dt)})")
        if args.modo in ("todos", "transferencias"):
            print("\n [Transferência CPU <-> GPU]")
            print("   - Partículas            : [10000, 100000, 1000000] (160 KB a 16 MB)")
        if args.modo in ("todos", "compilacao"):
            print("\n [Compilação JIT de Kernels]")
            print("   - Compilador            : NVRTC / CuPy RawKernel (-O3, --use_fast_math)")
        print("=" * 70)
        print("Dry run finalizado. Nenhuma simulação ou benchmark foi executado.\n")
        return

    if args.modo in ("todos", "compilacao"):
        benchmark_compilacao()

    if args.modo in ("todos", "transferencias"):
        benchmark_transferencias(
            lista_Np=[10_000, 100_000, 1_000_000],
            repeticoes=args.repeticoes,
            backend_name=args.backend,
        )

    if args.modo in ("todos", "campo"):
        benchmark_campo(
            lista_Np=lista_Np,
            lista_n_max=lista_n_max,
            n_repeticoes=args.repeticoes,
            seed=args.seed,
            backend_name=args.backend,
        )

    if args.modo in ("todos", "integradores"):
        benchmark_integradores(
            lista_Np=lista_Np_int,
            n_max=args.n_max_int,
            dt=args.dt,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend,
        )


if __name__ == "__main__":
    main()
