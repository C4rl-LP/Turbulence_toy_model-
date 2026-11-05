import numpy as np 
import os
from matplotlib import pyplot as plt
import funcoes_para_centros as fc
from funcoes_para_centros import R_1, O_1, lamb, x_1, T_1, alpha_padrao, c_padrao
import time 
import backend
import campos

def solve_RK4(func, r0, t0, dt, t_max, backend_name="auto", guardar_trajetoria=True):
    """
    Resolve o sistema dr/dt = func(t, r) usando RK4.

    Parâmetros:
    - func : função do campo vetorial
    - r0   : condição inicial (array)
    - t0   : tempo inicial
    - dt   : passo de tempo
    - t_max: tempo final
    - backend_name : 'cpu', 'cuda' ou 'auto'
    - guardar_trajetoria : se False, mantém apenas o estado atual e retorna
      a posição final, reduzindo o armazenamento de O(Nt * Np) para O(Np).
    """
    b = backend.resolve_backend(backend_name)
    xp = backend.get_backend_module(b)

    Nt = int((t_max - t0) / dt) + 1
    r0_dev = backend.to_device(r0, backend=b)

    t = np.zeros(Nt, dtype=float)
    if guardar_trajetoria:
        r = xp.empty((Nt, *r0_dev.shape), dtype=r0_dev.dtype)
        r[0] = r0_dev
        ri = r[0]
    else:
        r = None
        ri = r0_dev
    t[0] = t0

    for i in range(Nt - 1):
        ti = t[i]

        k1 = func(ti, ri)
        k2 = func(ti + dt / 2.0, ri + dt / 2.0 * k1)
        k3 = func(ti + dt / 2.0, ri + dt / 2.0 * k2)
        k4 = func(ti + dt, ri + dt * k3)

        ri = ri + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if guardar_trajetoria:
            r[i + 1] = ri
        t[i + 1] = ti + dt

    return t, backend.to_cpu(r if guardar_trajetoria else ri)

def solve_euler_maruyama(func, r0, t_max, dt):
    """
    Resolve

        dr = V(r,t) dt + dW

    usando o método de Euler-Maruyama.

    Parâmetros
    ----------
    func : função
        Campo de velocidade V(t, r).

    r0 : array_like, shape (2,)
        Condição inicial [x0, y0].

    t_max : float
        Tempo total da simulação.

    dt : float
        Passo temporal.

    Retorna
    -------
    t : ndarray, shape (Nt,)
        Tempos.

    r : ndarray, shape (Nt, 2)
        Trajetória.
    """

    N_steps = int(np.ceil(t_max / dt))

    # Ajusta dt para que o último ponto seja exatamente t_max
    dt_real = t_max / N_steps

    t = np.linspace(0.0, t_max, N_steps + 1)    

    r = np.empty((N_steps + 1, 2))
    r[0] = r0

    for k in range(N_steps):
        tk = t[k]

        V = func(tk, r[k])

        dW = np.sqrt(dt_real) * np.random.normal(
            size=2
        )

        r[k + 1] = (
            r[k]
            + V * dt_real
            + dW
        )

    return t, r

def simular_particulas(
    N,
    N_fields,
    dimensao_quadrado,
    r0,
    dt=0.01,
    t_max=1,
    func=None,
    temporal=False,
    backend_name="auto",
    seed=None,
    guardar_trajetoria=True,
):
    """
    Simula N partículas em um pequeno disco ao redor da posição inicial r0.
    """
    b = backend.resolve_backend(backend_name)
    xp = backend.get_backend_module(b)

    if seed is not None:
        np.random.seed(seed)
        if b == "cuda":
            xp.random.seed(seed)

    if func is None:
        func = campos.campo_total

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        t_eval = t if temporal else 0.0
        try:
            Vx, Vy = func(x, y, t_eval, N_fields, backend_name=b)
        except TypeError:
            Vx, Vy = func(x, y, t_eval, N_fields)
        return xp.column_stack((Vx, Vy))

    L = dimensao_quadrado
    R0 = L / 2.0

    theta = np.random.uniform(0, 2 * np.pi, size=N - 1)
    u = np.random.uniform(0, 1, size=N - 1)

    r_rand = R0 * np.sqrt(u)
    dx = r_rand * np.cos(theta)
    dy = r_rand * np.sin(theta)

    r0_arr = np.asarray(r0, dtype=float).reshape(1, 2)
    particle_t0 = r0_arr + np.column_stack((dx, dy))
    particle_t0 = np.vstack((r0_arr, particle_t0))

    t, r = solve_RK4(
        funcao, particle_t0, 0.0, dt, t_max,
        backend_name=b, guardar_trajetoria=guardar_trajetoria
    )
    return t, r


def simular_particulas_ruidosa(
    N,
    N_fields,
    r0,
    dt_1=0.01,
    t_max=1,
    lamb=2**(2/3),
    func=None,
    temporal=False,
    sigma0=0.0,
    backend_name="auto",
    seed=None
):
    """
    Simula N_particulas realizações da SDE:
        dr = V_N(r,t) dt + dW
    todas próximas da mesma condição inicial r0.
    """
    b = backend.resolve_backend(backend_name)
    xp = backend.get_backend_module(b)

    if seed is not None:
        np.random.seed(seed)
        if b == "cuda":
            xp.random.seed(seed)

    if func is None:
        func = campos.campo_total

    r0 = np.asarray(r0, dtype=float).reshape(1, 2)

    dt_N = dt_1 / (lamb**(N_fields - 1))
    N_steps = int(np.ceil(t_max / dt_N))
    dt_N = t_max / N_steps

    t = np.linspace(0.0, t_max, N_steps + 1)

    particle_t0 = np.tile(r0, (N, 1))
    if sigma0 > 0.0:
        particle_t0 = particle_t0 + np.random.normal(0.0, sigma0, size=(N, 2))

    particle_t0_dev = backend.to_device(particle_t0, backend=b)

    rs = xp.empty((N_steps + 1, N, 2), dtype=xp.float64)
    rs[0] = particle_t0_dev

    sqrt_dt = np.sqrt(dt_N)

    for k in range(N_steps):
        tk = t[k]
        x = rs[k, :, 0]
        y = rs[k, :, 1]
        t_eval = tk if temporal else 0.0

        try:
            Vx, Vy = func(x, y, t_eval, N_fields, backend_name=b)
        except TypeError:
            Vx, Vy = func(x, y, t_eval, N_fields)

        V = xp.column_stack((Vx, Vy))

        if b == "cuda":
            dW = sqrt_dt * xp.random.normal(size=(N, 2))
        else:
            dW = sqrt_dt * np.random.normal(size=(N, 2))

        rs[k + 1] = rs[k] + V * dt_N + dW

    return t, backend.to_cpu(rs)
