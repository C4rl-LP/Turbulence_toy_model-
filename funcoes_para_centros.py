import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# Constantes globais
# ============================================================

R_1 = 1            # Raio característico do nível 1
T_1 = 1             # Período caracterísitoc do nível 1
O_1 = 2*np.pi            # Frequência angular base
x_1 = np.array([0, 0])  # Centro inicial (nível 1)
lamb = 2**(2/3)         # Parâmetro que determina a razão de decaimento do período

# Constantes referentes a função bump B(r/[sqrt(alpha_padrao)R_n] ) | B(x) = exp(-1/(1- x²))
alpha_padrao = np.sqrt(2.5) # Determina o corte da função 
c_padrao = 0.6              # Determina a incluinação da função     




def indices_nivel(n):
    """
    Gera os índices que identificam cada centro no nível n.
    O número de centros cresce como 4^(n-1).
    """
    if n == 1:
        return [[]]
    return list(product(range(4), repeat=n-1))


def R(n):
    """ Raio característico do nível n """
    return R_1 * 2**(1 - n)

def T(n):
    """ Período associado ao nível n """
    return T_1* lamb**(-n +1 ) 

def Omega(n):
    """ Frequência angular do nível n """
    return O_1 * lamb**(n - 1)   

def R_cufoff(n, alpha = np.sqrt(2)):
    return alpha*R(n)

def phi(n, index):
    idx = index[-1] 

    return np.pi/4 + idx * (np.pi/2)

def x_centros(n, index, t):
    """
    Calcula recursivamente a posição do centro associado
    a um índice hierárquico no nível n.
    """
    if n == 1:
        return x_1

    if len(index) != n - 1:
        raise ValueError("Erro de tamanho de array")

    # Deslocamento circular do centro atual
    diff = R(n) * np.sqrt(2) * np.array([
        np.cos(Omega(n) * t + phi(n, index)),
        np.sin(Omega(n) * t + phi(n, index))
    ])

    # Soma com o centro do nível anterior
    return diff + x_centros(n - 1, index[:-1], t)

def centros_nivel_todos(n, t):
    """
    Retorna array (N_centros, 2) com todos os centros do nível n na GPU (CuPy)
    """
    centros = cp.array([[0.0, 0.0]])  # nível 1

    for k in range(2, n+1):

        Rk = R(k) * np.sqrt(2)
        angle_base = Omega(k) * t

        # 4 possíveis fases
        fases = cp.array([
            -cp.pi/4,
             cp.pi/4,
             3*cp.pi/4,
             5*cp.pi/4
        ])

        deslocamentos = Rk * cp.column_stack([
            cp.cos(angle_base + fases),
            cp.sin(angle_base + fases)
        ])

        # Produto cartesiano vetorizado
        centros = (
            centros[:, None, :] + deslocamentos[None, :, :]
        ).reshape(-1, 2)

    return centros




def atualiza_centro(cx, cy, t, n, idx):
    """
    cx, cy: (Np,)
    idx: (Np,)
    """
    Rn = R(n)
    On = Omega(n)

    phi_val = np.pi/4 + (np.pi/2) * cp.asarray(idx)

    shift_x = cp.sqrt(2)*Rn * cp.cos(On*t + phi_val)
    shift_y = cp.sqrt(2)*Rn * cp.sin(On*t + phi_val)

    return cx + shift_x, cy + shift_y



def subcentro(n, cx_0, cy_0, t):
    """
    Retorna arrays CuPy (4,) com as coordenadas (x, y) dos 4 subcentros
    gerados no nível n+1 a partir de (cx_0, cy_0).
    """
    phi_vals = cp.pi/4 + (cp.pi/2) * cp.arange(4)
    shift_x = R(n+1) * cp.sqrt(2) * cp.cos(Omega(n+1)*t + phi_vals)
    shift_y = R(n+1) * cp.sqrt(2) * cp.sin(Omega(n+1)*t + phi_vals)
    return cx_0 + shift_x, cy_0 + shift_y       

def sondar(n, xp, yp, cx_0, cy_0, t):
    sub_cx, sub_cy = subcentro(n, cx_0, cy_0, t)
    cx_valid = []
    cy_valid = []
    for s_cx_i, s_cy_i in zip(sub_cx, sub_cy):
        dx = xp - s_cx_i
        dy = yp - s_cy_i 
        print(s_cx_i, s_cy_i)
        print((dx**2 + dy**2)) # agora 0,1,2,3
        if dx**2 + dy**2 < (R_cufoff(n+1, alpha_padrao))**2:
            cx_valid.append(s_cx_i)
            cy_valid.append(s_cy_i)
    return cx_valid, cy_valid

def verificar(N, xp, yp, t, x_0, y_0): # x_0, y_0 é  a coordenada do primeiro centro n =1
    centros = [np.array([x_0, y_0])]
    centros_vistos =  [np.array([x_0, y_0])]
    for j in range(1, N):
        sub = []
        for r_0k in centros_vistos:
            valid_x, valid_y = sondar(j, xp,yp, r_0k[0],r_0k[1],t)
            for cxi,cyi in zip(
                valid_x,
                valid_y
            ):
                sub.append((cxi,cyi))
        centros_vistos = sub

        centros.append(sub)
    return centros

def sondar_vet(n, xp, yp, cx0, cy0, t):
    xp = cp.atleast_1d(cp.asarray(xp))
    yp = cp.atleast_1d(cp.asarray(yp))

    sub_cx, sub_cy = subcentro(
        n, cx0, cy0, t
    )

    # (Np,4)
    dx = xp[:, None] - sub_cx
    dy = yp[:, None] - sub_cy

    mask = (
        dx**2 + dy**2
        < (R_cufoff(n+1, alpha_padrao))**2
    )

    return sub_cx, sub_cy, mask

def verificar_vet(N,xp,yp,t,x0,y0):

    xp=cp.atleast_1d(cp.asarray(xp))
    yp=cp.atleast_1d(cp.asarray(yp))

    Np=len(xp)

    # para cada partícula:
    ativos=[
        [(x0,y0)]
        for _ in range(Np)
    ]

    hist=[ativos.copy()]


    for n in range(1,N):

        novos=[]

        for p in range(Np):
            
            ativos_p=[]

            for cx,cy in ativos[p]:

                sub_cx,sub_cy,mask = sondar_vet(
                    n,
                    xp[p:p+1],
                    yp[p:p+1],
                    cx,
                    cy,
                    t
                )

                filhos=list(
                    zip(
                      sub_cx[mask[0]],
                      sub_cy[mask[0]]
                    )
                )

                ativos_p.extend(filhos)

            novos.append(ativos_p)

        ativos=novos
        hist.append(ativos.copy())

    return hist


if __name__ == "__main__":
    pass
