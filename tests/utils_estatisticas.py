import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Garantir importação dos módulos da raiz do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb


def calcular_estatisticas_trajetorias(t: np.ndarray, rs: np.ndarray) -> dict:
    """
    Calcula séries temporais das principais estatísticas a partir das trajetórias.

    Parâmetros
    ----------
    t : ndarray, shape (Nt,)
        Vetor de tempos.
    rs : ndarray, shape (Nt, N, 2)
        Trajetórias das N partículas nos Nt passos de tempo.

    Retorna
    -------
    dict com:
        - 't': tempos
        - 'mean_x', 'mean_y': médias de x e y
        - 'var_x', 'var_y': variâncias amostrais de x e y
        - 'cov_xy': covariância entre x e y
        - 'sigma2': variância total sigma^2(t) = < |r(t) - <r(t)>|^2 >
        - 'msd': deslocamento quadrático médio MSD(t) = < |r(t) - r(0)|^2 >
        - 'dist_media_trajetorias': distância média de cada partícula ao centroide
    """
    Nt, N, _ = rs.shape

    mean_x = np.mean(rs[:, :, 0], axis=1)
    mean_y = np.mean(rs[:, :, 1], axis=1)

    if N > 1:
        var_x = np.var(rs[:, :, 0], axis=1, ddof=1)
        var_y = np.var(rs[:, :, 1], axis=1, ddof=1)
        cov_xy = np.zeros(Nt)
        for k in range(Nt):
            cov_mat = np.cov(rs[k, :, 0], rs[k, :, 1], ddof=1)
            cov_xy[k] = cov_mat[0, 1]
    else:
        var_x = np.zeros(Nt)
        var_y = np.zeros(Nt)
        cov_xy = np.zeros(Nt)

    # sigma^2(t) = < |r(t) - <r(t)>|^2 >
    mean_vec = np.column_stack((mean_x, mean_y))  # shape (Nt, 2)
    dev = rs - mean_vec[:, None, :]  # shape (Nt, N, 2)
    sigma2 = np.mean(np.sum(dev**2, axis=2), axis=1)
    dist_centroide = np.mean(np.sqrt(np.sum(dev**2, axis=2)), axis=1)

    # MSD(t) = < |r(t) - r(0)|^2 >
    disp = rs - rs[0:1, :, :]  # shape (Nt, N, 2)
    msd = np.mean(np.sum(disp**2, axis=2), axis=1)

    return {
        't': t,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'var_x': var_x,
        'var_y': var_y,
        'cov_xy': cov_xy,
        'sigma2': sigma2,
        'msd': msd,
        'dist_media_trajetorias': dist_centroide
    }


def calcular_expoente_difusao(t: np.ndarray, msd: np.ndarray) -> np.ndarray:
    """
    Calcula a estimativa local do expoente de difusão:
        alpha(t) = d log(MSD) / d log(t)

    Utiliza diferenças finitas numéricas nos valores logarítmicos.
    Ignora t=0 para evitar log(0).
    """
    valid = (t > 0) & (msd > 0)
    t_val = t[valid]
    msd_val = msd[valid]

    if len(t_val) < 2:
        return np.zeros_like(t)

    log_t = np.log(t_val)
    log_msd = np.log(msd_val)

    dlog_msd = np.gradient(log_msd, log_t)

    # Mapeia de volta para o vetor original de tempo
    alpha = np.zeros_like(t)
    alpha[valid] = dlog_msd
    return alpha


def calcular_assimetria(x: np.ndarray) -> float:
    """Calcula a assimetria (skewness) de um conjunto de dados."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def calcular_curtose(x: np.ndarray) -> float:
    """Calcula o excesso de curtose (kurtosis - 3) de um conjunto de dados."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3.0


def calcular_estatisticas_distribuicao(data: np.ndarray) -> dict:
    """
    Calcula momento de ordem superior, quantis e estatísticas de uma distribuição 1D.
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        'var': float(np.var(data, ddof=1)) if len(data) > 1 else 0.0,
        'assimetria': float(calcular_assimetria(data)),
        'curtose': float(calcular_curtose(data)),
        'q05': float(np.quantile(data, 0.05)),
        'q25': float(np.quantile(data, 0.25)),
        'q50': float(np.quantile(data, 0.50)),
        'q75': float(np.quantile(data, 0.75)),
        'q95': float(np.quantile(data, 0.95))
    }


def gerar_histogramas_completos(
    rs_final: np.ndarray,
    title_prefix: str,
    output_dir: str,
    filename_prefix: str
):
    """
    Gera e salva histogramas 1D de x, 1D de y e 2D de (x, y).
    """
    os.makedirs(output_dir, exist_ok=True)
    x = rs_final[:, 0]
    y = rs_final[:, 1]

    # Histogramas 1D
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(x, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Densidade')
    axes[0].set_title(f'{title_prefix} - Histograma 1D (x)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(y, bins=30, density=True, color='salmon', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('Densidade')
    axes[1].set_title(f'{title_prefix} - Histograma 1D (y)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_hist1d.png'), dpi=200)
    plt.close()

    # Histograma 2D
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=40, density=True, cmap='viridis')
    plt.colorbar(label='Densidade')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{title_prefix} - Histograma 2D (x, y)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_hist2d.png'), dpi=200)
    plt.close()


def simular_estocastico_com_perturbacao(
    N: int,
    N_fields: int,
    r0: np.ndarray,
    delta_r: float,
    dt_1: float = 0.01,
    t_max: float = 1.0,
    lamb_val: float = lamb,
    func=None,
    temporal: bool = False,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simula N partículas sob dinâmica estocástica com perturbação inicial delta_r:
        dr = V_N(r, t) dt + dW
        r(0) = r0 + delta_r
    """
    rng = np.random.default_rng(seed)
    r0 = np.asarray(r0, dtype=float).reshape(-1)

    theta = rng.uniform(0, 2 * np.pi, size=N)
    u = rng.uniform(0, 1, size=N)
    r_rand = delta_r * np.sqrt(u)

    dx = r_rand * np.cos(theta)
    dy = r_rand * np.sin(theta)

    particle_t0 = r0 + np.column_stack((dx, dy))

    dt_N = dt_1 / (lamb_val ** (N_fields - 1))
    N_steps = int(np.ceil(t_max / dt_N))
    dt_N = t_max / N_steps
    t = np.linspace(0.0, t_max, N_steps + 1)

    rs = np.empty((N_steps + 1, N, 2))
    rs[0] = particle_t0

    for k in range(N_steps):
        tk = t[k]
        x = rs[k, :, 0]
        y = rs[k, :, 1]

        if func is None:
            func = campos.campo_total_podado

        if temporal:
            Vx, Vy = func(x, y, tk, N_fields)
        else:
            Vx, Vy = func(x, y, 0, N_fields)

        V = np.column_stack((Vx, Vy))
        dW = np.sqrt(dt_N) * rng.normal(size=(N, 2))
        rs[k + 1] = rs[k] + V * dt_N + dW

    return t, rs


def executar_simulacoes_paralelas(
    sim_func,
    N_total: int,
    num_chunks: int = 4,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hook preparado para substituição do loop explícito por execução paralela/vetorizada
    das realizações das partículas divididas em processos independentes.
    """
    if num_chunks <= 1:
        return sim_func(N=N_total, **kwargs)

    sub_N = N_total // num_chunks
    sobram = N_total % num_chunks
    tamanho_chunks = [sub_N + (1 if i < sobram else 0) for i in range(num_chunks)]

    def Worker(chunk_args):
        chunk_idx, n_part = chunk_args
        kw = kwargs.copy()
        if 'seed' in kw:
            kw['seed'] += chunk_idx * 100
        return sim_func(N=n_part, **kw)

    tasks = [(i, tamanho_chunks[i]) for i in range(num_chunks)]

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        results = list(executor.map(Worker, tasks))

    t_base = results[0][0]
    rs_comb = np.concatenate([r for _, r in results], axis=1)

    return t_base, rs_comb
