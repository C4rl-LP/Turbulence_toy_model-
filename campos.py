import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from itertools import product
import integradores as it
import funcoes_para_centros as fc 
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
    x: cp.ndarray | float,
    y: cp.ndarray | float,
    x_c: float,
    y_c: float,
    t: float,
    n: int,
    c: float = c_padrao,
    alpha: float = alpha_padrao
    ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y) na GPU (CuPy).
    """
    x = cp.asarray(x)
    y = cp.asarray(y)

    # Diferenças vetoriais (arrays na GPU)
    dx = x - x_c
    dy = y - y_c
    r2 = dx**2 + dy**2
    R2_coef = (fc.R_cufoff(n, alpha))**2

    vx = cp.zeros_like(dx)
    vy = cp.zeros_like(dy)

    poten = cp.zeros_like(r2)

    mask = r2 < R2_coef
    
    if cp.any(mask):
        poten[mask] = (
            2 * cp.pi
            * cp.exp(c / ((r2[mask] / R2_coef) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy
# Campo base que utiliza os index 

def campo_2_com_index(
    x: cp.ndarray | float,
    y: cp.ndarray | float,
    t: float,
    n: int,
    index: list[int],
    c: float = c_padrao,
    alpha: float = alpha_padrao
    ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Campo vetorial induzido por um único centro
    do nível n nos pontos (x, y) na GPU (CuPy).
    """
    x_centro_n = fc.x_centros(n, index, t)
    x = cp.asarray(x)
    y = cp.asarray(y)

    # Diferenças vetoriais (arrays na GPU)
    dx = x - x_centro_n[0]
    dy = y - x_centro_n[1]
    r2 = dx**2 + dy**2
    R2 = (fc.R_cufoff(n, alpha))**2

    vx = cp.zeros_like(dx)
    vy = cp.zeros_like(dy)

    poten = cp.zeros_like(r2)

    mask = r2 < R2

    if cp.any(mask):
        poten[mask] = (
            2 * cp.pi
            * cp.exp(c / ((r2[mask] / R2) - 1))
            / fc.T(n)
        )
    vx[mask] = -dy[mask] * poten[mask]
    vy[mask] = dx[mask] * poten[mask]

    return vx, vy

# Campo total usando todos os index e não ótimo
def campo_total_correto(
    x: cp.ndarray,
    y: cp.ndarray,
    t: float,
    n_max: int = 3
    ) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos na GPU (CuPy).
    """
    x = cp.asarray(x)
    y = cp.asarray(y)
    Vx = cp.zeros_like(x)
    Vy = cp.zeros_like(y)
    
    for n in range(1, n_max + 1):
        for idx in fc.indices_nivel(n):

            vx, vy = campo_2_com_index(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy
def campo_total_podado(
    xp: cp.ndarray | float,
    yp: cp.ndarray | float,
    t: float,
    n_max: int = 3,
    c: float = c_padrao,
    alpha: float = alpha_padrao
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Soma contribuições apenas dos centros ativos na árvore podada na GPU (CuPy).
    Permite passar opcionalmente os parâmetros c e alpha.
    """
    xp_is_scalar = cp.isscalar(xp)
    yp_is_scalar = cp.isscalar(yp)

    xp_arr = cp.atleast_1d(cp.asarray(xp))
    yp_arr = cp.atleast_1d(cp.asarray(yp))
    Np = len(xp_arr)
    x0, y0 = 0.0, 0.0

    # Nível 1 (raiz)
    vx_total, vy_total = campo_2(xp_arr, yp_arr, x0, y0, t, 1, c=c, alpha=alpha)

    if n_max > 1:
        centros_x = [cp.array([x0]) for _ in range(Np)]
        centros_y = [cp.array([y0]) for _ in range(Np)]

        for n in range(1, n_max):
            R_cut2 = (fc.R_cufoff(n + 1, alpha))**2
            inv_R_cut2 = 1.0 / R_cut2
            fator_poten = 2.0 * cp.pi / fc.T(n + 1)

            sub_cx_base, sub_cy_base = fc.subcentro(n, 0.0, 0.0, t)

            novos_x = []
            novos_y = []

            for p in range(Np):
                cx = centros_x[p]
                cy = centros_y[p]
                if len(cx) == 0:
                    novos_x.append(cp.empty(0))
                    novos_y.append(cp.empty(0))
                    continue

                scx = (cx[:, None] + sub_cx_base[None, :]).ravel()
                scy = (cy[:, None] + sub_cy_base[None, :]).ravel()

                dx = xp_arr[p] - scx
                dy = yp_arr[p] - scy
                r2 = dx*dx + dy*dy
                mask = r2 < R_cut2

                if cp.any(mask):
                    scx_m = scx[mask]
                    scy_m = scy[mask]
                    novos_x.append(scx_m)
                    novos_y.append(scy_m)

                    dx_m = dx[mask]
                    dy_m = dy[mask]
                    r2_m = r2[mask]
                    poten = fator_poten * cp.exp(c / ((r2_m * inv_R_cut2) - 1.0))
                    vx_total[p] += cp.sum(-dy_m * poten)
                    vy_total[p] += cp.sum(dx_m * poten)
                else:
                    novos_x.append(cp.empty(0))
                    novos_y.append(cp.empty(0))

            centros_x = novos_x
            centros_y = novos_y

    if xp_is_scalar and yp_is_scalar:
        return float(vx_total[0]), float(vy_total[0])
    return vx_total, vy_total


if __name__ =="__main__":
    x = np.array([-.2])
    y = np.array([.1])


    '''n = 4
    # Função para simular as partículas.
    x = np.array([-.2, -.201, .1])
    y = np.array([.1, .1, .013])
    t = 0.2

    a = campo_total_otimo(x, y, t, n)
    b = campo_total_otimo_vet(x, y, t, n)
    d = campo_total_podado(x,y, t, n)
    c = campo_total_correto(x, y , t, n)
    t_linspace = np.linspace(0, lamb, 100)
    for i in t_linspace:
        print(i)
        c = campo_total_correto(x,y, i, n)
        d = campo_total_podado(x,y, i, n)
        print(np.array(c)- np.array(d))

    print(f'campo total otimo:{a}')
    print(f'campo total otimo vetorizado certo:{b}')
    print(f'campo total correto:{c}')
    print(f'campo total podado:{np.array(d)}')
    print(f'subtração do podado: {np.array(d)- np.array(c)}')'''

    
    '''a= 'estabilidade_podada'
    b ='estabilidade_com_todos_os_pontos'
    it.testar_estabilidade_estatico(
        nivel_max=8,
        N_particulas=50,
        r0=r0,
        nivel_min = 1,
        pasta_saida= a,
        funcao=campo_total_podado
    )'''
    r0 = np.array([[.3, .2333]])
    c= 'estabilidade_temporal'
    d ='estabilidade_com_todos_os_pontos_temporal'
    '''it.testar_estabilidade_temporal(
        nivel_max=4,
        N_particulas=50,
        r0=r0,
        nivel_min = 1,
        pasta_saida= d,
        funcao=campo_total_correto
    )'''
    it.testar_estabilidade_temporal(
        nivel_max=1,
        N_particulas=500,
        r0=r0,
        nivel_min = 1,
        pasta_saida= c,
        funcao=campo_total_podado,
        pasta_saida_info= 'estabilidade_podadda_info'
    )


