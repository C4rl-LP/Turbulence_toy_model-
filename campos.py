import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import integradores as it
import funcoes_para_centros as fc 
import backend
from funcoes_para_centros import (
    R_1,
    O_1,
    x_1,
    lamb,
    c_padrao,
    T_1,
    alpha_padrao
)


# Campo base como função dos centros, sem usar o index

def campo_2(
    x: np.ndarray | float,
    y: np.ndarray | float,
    x_c: float,
    y_c: float,
    t: float,
    n: int,
    c: float = c_padrao,
    alpha: float = alpha_padrao
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Diferenças vetoriais (arrays)
    dx = x - x_c
    dy = y - y_c
    r2 = dx**2 + dy**2
    R2_coef = (fc.R_cufoff(n, alpha))**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2_coef
    
    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(c / ((r2[mask] / R2_coef) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy
# Campo base que utiliza os index 

def campo_2_com_index(
    x: np.ndarray | float,
    y: np.ndarray | float,
    t: float,
    n: int,
    index: list[int],
    c: float = c_padrao,
    alpha: float = alpha_padrao
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y).
    """
    x_centro_n = fc.x_centros(n, index, t)

    # Diferenças vetoriais (arrays)
    dx = x - x_centro_n[0]
    dy = y - x_centro_n[1]
    r2 = dx**2 + dy**2
    R2 = (fc.R_cufoff(n, alpha))**2

    vx= np.zeros_like(dx)
    vy = np.zeros_like(dy)

    poten = np.zeros_like(r2)

    mask = r2 < R2

    if np.any(mask):
        poten[mask] = (
            2*np.pi
            * np.exp(c / ((r2[mask] / R2) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy

# Campo total usando todos os index e não ótimo
def campo_total_correto(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos.
    """
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)
    
    for n in range(1, n_max + 1):
        for idx in fc.indices_nivel(n):

            vx, vy = campo_2_com_index(x, y, t, n, list(idx), c=c, alpha = alpha)
            Vx += vx
            Vy += vy

    return Vx, Vy
def campo_total_podado(
    xp: np.ndarray | float,
    yp: np.ndarray | float,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soma contribuições apenas dos centros ativos na árvore podada.
    Permite passar opcionalmente os parâmetros c e alpha.
    """
    xp_arr = np.atleast_1d(xp)
    yp_arr = np.atleast_1d(yp)
    Np = len(xp_arr)
    x0, y0 = 0.0, 0.0

    # Nível 1 (raiz)
    vx_total, vy_total = campo_2(xp_arr, yp_arr, x0, y0, t, 1, c=c, alpha=alpha)

    if n_max > 1:
        centros_x = [np.array([x0]) for _ in range(Np)]
        centros_y = [np.array([y0]) for _ in range(Np)]

        for n in range(1, n_max):
            R_cut2 = (fc.R_cufoff(n + 1, alpha))**2
            inv_R_cut2 = 1.0 / R_cut2
            fator_poten = 2.0 * np.pi / fc.T(n + 1)

            sub_cx_base, sub_cy_base = fc.subcentro(n, 0.0, 0.0, t)

            novos_x = []
            novos_y = []

            for p in range(Np):
                cx = centros_x[p]
                cy = centros_y[p]
                if len(cx) == 0:
                    novos_x.append(np.empty(0))
                    novos_y.append(np.empty(0))
                    continue

                scx = (cx[:, None] + sub_cx_base[None, :]).ravel()
                scy = (cy[:, None] + sub_cy_base[None, :]).ravel()

                dx = xp_arr[p] - scx
                dy = yp_arr[p] - scy
                r2 = dx*dx + dy*dy
                mask = r2 < R_cut2

                if np.any(mask):
                    scx_m = scx[mask]
                    scy_m = scy[mask]
                    novos_x.append(scx_m)
                    novos_y.append(scy_m)

                    dx_m = dx[mask]
                    dy_m = dy[mask]
                    r2_m = r2[mask]
                    poten = fator_poten * np.exp(c / ((r2_m * inv_R_cut2) - 1.0))
                    vx_total[p] += np.sum(-dy_m * poten)
                    vy_total[p] += np.sum(dx_m * poten)
                else:
                    novos_x.append(np.empty(0))
                    novos_y.append(np.empty(0))

            centros_x = novos_x
            centros_y = novos_y

    if np.isscalar(xp) and np.isscalar(yp):
        return float(vx_total[0]), float(vy_total[0])
    return vx_total, vy_total


def campo_total_podado_2(
    xp: np.ndarray | float,
    yp: np.ndarray | float,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o campo total usando uma poda segura para alpha > 0.

    Um centro pode ser expandido mesmo quando não contribui
    diretamente para o campo no ponto avaliado.
    """
    if alpha <= 0:
        raise ValueError("alpha deve ser positivo.")

    xp_arr = np.atleast_1d(xp)
    yp_arr = np.atleast_1d(yp)

    if xp_arr.shape != yp_arr.shape:
        raise ValueError("xp e yp devem ter o mesmo formato.")

    Np = len(xp_arr)
    x0, y0 = 0.0, 0.0

    # Contribuição da raiz, nível 1.
    vx_total, vy_total = campo_2(
        xp_arr, yp_arr, x0, y0, t, 1,
        c=c, alpha=alpha
    )

    # Centros candidatos à expansão, inicialmente a raiz.
    centros_x = [np.array([x0]) for _ in range(Np)]
    centros_y = [np.array([y0]) for _ in range(Np)]

    # Percorre os níveis dos pais.
    for n in range(1, n_max):
        nivel_filho = n + 1

        # Raio do suporte dos filhos.
        R_cut = fc.R_cufoff(nivel_filho, alpha)
        R_cut2 = R_cut**2
        inv_R_cut2 = 1.0 / R_cut2

        fator_poten = 2.0 * np.pi / fc.T(nivel_filho)

        # Limite que contém os suportes de todos os
        # descendentes de um centro de nível n.
        R_cut_pai = fc.R_cufoff(n, alpha)
        Rn = R_cut_pai / alpha
        Dn = (np.sqrt(2.0) + alpha / 2.0) * Rn

        # Deslocamentos dos quatro filhos em relação ao pai.
        sub_cx_base, sub_cy_base = fc.subcentro(
            n, 0.0, 0.0, t
        )

        novos_x = []
        novos_y = []

        for p in range(Np):
            cx = centros_x[p]
            cy = centros_y[p]

            if len(cx) == 0:
                novos_x.append(np.empty(0))
                novos_y.append(np.empty(0))
                continue

            # Descarta pais cuja descendência inteira
            # não pode alcançar o ponto.
            dx_pai = xp_arr[p] - cx
            dy_pai = yp_arr[p] - cy

            pode_expandir = (
                dx_pai**2 + dy_pai**2 <= Dn**2
            )

            cx = cx[pode_expandir]
            cy = cy[pode_expandir]

            if len(cx) == 0:
                novos_x.append(np.empty(0))
                novos_y.append(np.empty(0))
                continue

            # Gera todos os filhos dos pais restantes.
            scx = (
                cx[:, None] + sub_cx_base[None, :]
            ).ravel()

            scy = (
                cy[:, None] + sub_cy_base[None, :]
            ).ravel()

            dx = xp_arr[p] - scx
            dy = yp_arr[p] - scy
            r2 = dx**2 + dy**2

            # 1. Contribuição direta: dentro do suporte.
            mask_campo = r2 < R_cut2

            if np.any(mask_campo):
                dx_m = dx[mask_campo]
                dy_m = dy[mask_campo]
                r2_m = r2[mask_campo]

                poten = fator_poten * np.exp(
                    c / (r2_m * inv_R_cut2 - 1.0)
                )

                vx_total[p] += np.sum(-dy_m * poten)
                vy_total[p] += np.sum(dx_m * poten)

            # 2. Candidatos à próxima expansão.
            # Não exigimos que contribuam diretamente.
            if nivel_filho < n_max:
                D_filho = (
                    np.sqrt(2.0) + alpha / 2.0
                ) * (R_cut / alpha)

                mask_expansao = r2 <= D_filho**2

                novos_x.append(scx[mask_expansao])
                novos_y.append(scy[mask_expansao])
            else:
                novos_x.append(np.empty(0))
                novos_y.append(np.empty(0))

        centros_x = novos_x
        centros_y = novos_y

    if np.isscalar(xp) and np.isscalar(yp):
        return float(vx_total[0]), float(vy_total[0])

    return vx_total, vy_total


def campo_total_podado_vetorizado(
    xp: np.ndarray | float,
    yp: np.ndarray | float,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Avaliação otimizada e vetorizada do campo hierárquico com poda segura.

    Elimina o loop Python sobre partículas e substitui micro-alocações por
    operações vetoriais unificadas sobre todos os pares ativos (partícula, centro),
    mantendo equivalência exata com campo_total_podado_2.
    """
    if alpha <= 0:
        raise ValueError("alpha deve ser positivo.")

    is_scalar = np.isscalar(xp) and np.isscalar(yp)
    xp_arr = np.atleast_1d(backend.to_cpu(xp)).astype(np.float64)
    yp_arr = np.atleast_1d(backend.to_cpu(yp)).astype(np.float64)

    if xp_arr.shape != yp_arr.shape:
        raise ValueError("xp e yp devem ter o mesmo formato.")

    Np = len(xp_arr)
    if Np == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    # Contribuição da raiz (nível 1)
    vx_total, vy_total = campo_2(xp_arr, yp_arr, 0.0, 0.0, t, 1, c=c, alpha=alpha)

    if n_max <= 1:
        if is_scalar:
            return float(vx_total[0]), float(vy_total[0])
        return vx_total, vy_total

    # Raio envelope do nível 1
    R_cut_root = fc.R_cufoff(1, alpha)
    Rn_root = R_cut_root / alpha
    Dn_root = (np.sqrt(2.0) + alpha / 2.0) * Rn_root

    dist2_root = xp_arr**2 + yp_arr**2
    pode_root = np.where(dist2_root <= Dn_root**2)[0]

    if len(pode_root) == 0:
        if is_scalar:
            return float(vx_total[0]), float(vy_total[0])
        return vx_total, vy_total

    part_idx = pode_root
    cx = np.zeros(len(part_idx), dtype=np.float64)
    cy = np.zeros(len(part_idx), dtype=np.float64)

    for n in range(1, n_max):
        if len(part_idx) == 0:
            break

        nivel_filho = n + 1
        R_cut = fc.R_cufoff(nivel_filho, alpha)
        R_cut2 = R_cut**2
        inv_R_cut2 = 1.0 / R_cut2
        fator_poten = 2.0 * np.pi / fc.T(nivel_filho)

        sub_cx_base, sub_cy_base = fc.subcentro(n, 0.0, 0.0, t)

        # Gera os 4 filhos de cada centro pai ativo
        scx = (cx[:, None] + sub_cx_base[None, :]).ravel()
        scy = (cy[:, None] + sub_cy_base[None, :]).ravel()
        spart = np.repeat(part_idx, 4)

        dx = xp_arr[spart] - scx
        dy = yp_arr[spart] - scy
        r2 = dx**2 + dy**2

        # 1. Contribuição direta se dentro do suporte
        mask_campo = r2 < R_cut2
        if np.any(mask_campo):
            dx_m = dx[mask_campo]
            dy_m = dy[mask_campo]
            r2_m = r2[mask_campo]
            part_m = spart[mask_campo]

            poten = fator_poten * np.exp(c / (r2_m * inv_R_cut2 - 1.0))
            np.add.at(vx_total, part_m, -dy_m * poten)
            np.add.at(vy_total, part_m, dx_m * poten)

        # 2. Candidatos à próxima expansão
        if nivel_filho < n_max:
            D_filho = (np.sqrt(2.0) + alpha / 2.0) * (R_cut / alpha)
            mask_exp = r2 <= D_filho**2
            cx = scx[mask_exp]
            cy = scy[mask_exp]
            part_idx = spart[mask_exp]
        else:
            break

    if is_scalar:
        return float(vx_total[0]), float(vy_total[0])
    return vx_total, vy_total


def campo_total(
    xp: np.ndarray | float,
    yp: np.ndarray | float,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao,
    backend_name: backend.BackendName = "auto",
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Interface unificada para avaliação do campo de velocidades hierárquico.

    Despacha automaticamente ou conforme especificado para:
    - 'cpu': Avaliação vetorizada e otimizada por poda hierárquica.
    - 'cuda': Avaliação acelerada na GPU via CuPy RawKernel.
    - 'auto': Seleciona 'cuda' se disponível, senão 'cpu'.
    """
    b = backend.resolve_backend(backend_name)
    if b == "cuda":
        import cuda_kernels
        vx, vy = cuda_kernels.avaliar_campo_cuda(
            xp, yp, t, n_max=n_max, c=c, alpha=alpha
        )
        if np.isscalar(xp) and np.isscalar(yp):
            return float(vx[0]), float(vy[0])
        return vx, vy

    return campo_total_podado_vetorizado(
        xp, yp, t, n_max=n_max, c=c, alpha=alpha
    )
