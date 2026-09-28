import numpy as np
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
    n_max: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Soma o campo vetorial de todos os níveis
    e todos os centros hierárquicos.
    """
    Vx = np.zeros_like(x)
    Vy = np.zeros_like(y)
    
    for n in range(1, n_max + 1):
        for idx in fc.indices_nivel(n):

            vx, vy = campo_2_com_index(x, y, t, n, list(idx))
            Vx += vx
            Vy += vy

    return Vx, Vy
def campo_total_podado(
    xp: np.ndarray | float,
    yp: np.ndarray | float,
    t: float,
    n_max: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Soma contribuições apenas dos centros
    encontrados por verificar().
    """
    xp=np.atleast_1d(xp)
    yp=np.atleast_1d(yp)
    # centro raiz n=1
    x0,y0 = 0.0,0.0
    Np = len(xp)
    # árvore podada
    centros = fc.verificar_vet(
        n_max,
        xp,yp,
        t,
        x0,y0
    )

    vx_total=0.0
    vy_total=0.0


    # nível 1 (raiz)
    vx,vy = campo_2(
        xp,yp,
        x0,y0,
        t,
        1
    )

    vx_total += vx
    vy_total += vy

    for n in range(2, n_max +1):
        for j in range(Np):
            for cx, cy in centros[n-1][j]:
                vx, vy = campo_2(xp[j], yp[j], cx, cy, t, n)
                vx_total[j] += vx 
                vy_total[j] += vy 
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


