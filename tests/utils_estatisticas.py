import os
import sys
import numpy as np
import cupy as cp
from concurrent.futures import ProcessPoolExecutor

# Garantir importação dos módulos da raiz do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb


def calcular_estatisticas_trajetorias(t, rs) -> dict:
    """
    Calcula séries temporais das principais estatísticas a partir das trajetórias na GPU (CuPy).

    Parâmetros
    ----------
    t : ndarray, shape (Nt,)
        Vetor de tempos.
    rs : ndarray, shape (Nt, N, 2)
        Trajetórias das N partículas nos Nt passos de tempo (CuPy ou NumPy array).

    Retorna
    -------
    dict com séries temporais convertidas para NumPy para salvamento e plots:
        - 't': tempos
        - 'mean_x', 'mean_y': médias de x e y
        - 'var_x', 'var_y': variâncias amostrais de x e y
        - 'cov_xy': covariância entre x e y
        - 'sigma2': variância total sigma^2(t) = < |r(t) - <r(t)>|^2 >
        - 'msd': deslocamento quadrático médio MSD(t) = < |r(t) - r(0)|^2 >
        - 'dist_media_trajetorias': distância média de cada partícula ao centroide
    """
    rs_gpu = cp.asarray(rs)
    Nt, N, _ = rs_gpu.shape

    mean_x = cp.mean(rs_gpu[:, :, 0], axis=1)
    mean_y = cp.mean(rs_gpu[:, :, 1], axis=1)

    if N > 1:
        var_x = cp.var(rs_gpu[:, :, 0], axis=1, ddof=1)
        var_y = cp.var(rs_gpu[:, :, 1], axis=1, ddof=1)
        cov_xy = cp.zeros(Nt)
        for k in range(Nt):
            cov_mat = cp.cov(rs_gpu[k, :, 0], rs_gpu[k, :, 1], ddof=1)
            cov_xy[k] = cov_mat[0, 1]
    else:
        var_x = cp.zeros(Nt)
        var_y = cp.zeros(Nt)
        cov_xy = cp.zeros(Nt)

    # sigma^2(t) = < |r(t) - <r(t)>|^2 >
    mean_vec = cp.column_stack((mean_x, mean_y))  # shape (Nt, 2)
    dev = rs_gpu - mean_vec[:, None, :]  # shape (Nt, N, 2)
    sigma2 = cp.mean(cp.sum(dev**2, axis=2), axis=1)
    dist_centroide = cp.mean(cp.sqrt(cp.sum(dev**2, axis=2)), axis=1)

    # MSD(t) = < |r(t) - r(0)|^2 >
    disp = rs_gpu - rs_gpu[0:1, :, :]  # shape (Nt, N, 2)
    msd = cp.mean(cp.sum(disp**2, axis=2), axis=1)

    t_cpu = cp.asnumpy(t) if isinstance(t, cp.ndarray) else np.asarray(t)

    return {
        't': t_cpu,
        'mean_x': cp.asnumpy(mean_x),
        'mean_y': cp.asnumpy(mean_y),
        'var_x': cp.asnumpy(var_x),
        'var_y': cp.asnumpy(var_y),
        'cov_xy': cp.asnumpy(cov_xy),
        'sigma2': cp.asnumpy(sigma2),
        'msd': cp.asnumpy(msd),
        'dist_media_trajetorias': cp.asnumpy(dist_centroide)
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
    data_cpu = cp.asnumpy(data) if isinstance(data, cp.ndarray) else np.asarray(data)
    return {
        'mean': float(np.mean(data_cpu)),
        'std': float(np.std(data_cpu, ddof=1)) if len(data_cpu) > 1 else 0.0,
        'var': float(np.var(data_cpu, ddof=1)) if len(data_cpu) > 1 else 0.0,
        'assimetria': float(calcular_assimetria(data_cpu)),
        'curtose': float(calcular_curtose(data_cpu)),
        'q05': float(np.quantile(data_cpu, 0.05)),
        'q25': float(np.quantile(data_cpu, 0.25)),
        'q50': float(np.quantile(data_cpu, 0.50)),
        'q75': float(np.quantile(data_cpu, 0.75)),
        'q95': float(np.quantile(data_cpu, 0.95))
    }


def gerar_histogramas_completos(rs_final, title_prefix, output_dir, filename_prefix):
    """Compatibilidade: visualização explícita, fora da execução dos testes."""
    from tests.graficos_resultados import gerar_histogramas_completos as desenhar
    from tests.dados_resultados import para_numpy
    return desenhar(para_numpy(rs_final), title_prefix, output_dir, filename_prefix)



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
    Simula N partículas sob dinâmica estocástica com perturbação inicial delta_r na GPU (CuPy):
        dr = V_N(r, t) dt + dW
        r(0) = r0 + delta_r
    """
    cp.random.seed(seed)
    r0_gpu = cp.asarray(r0, dtype=float).reshape(-1)

    theta = cp.random.uniform(0, 2 * cp.pi, size=N)
    u = cp.random.uniform(0, 1, size=N)
    r_rand = delta_r * cp.sqrt(u)

    dx = r_rand * cp.cos(theta)
    dy = r_rand * cp.sin(theta)

    particle_t0 = r0_gpu + cp.column_stack((dx, dy))

    dt_N = dt_1 / (lamb_val ** (N_fields - 1))
    N_steps = int(np.ceil(t_max / dt_N))
    dt_N = t_max / N_steps
    t = np.linspace(0.0, t_max, N_steps + 1)

    rs = cp.empty((N_steps + 1, N, 2), dtype=float)
    rs[0] = particle_t0

    if func is None:
        func = campos.campo_total_podado

    for k in range(N_steps):
        tk = float(t[k])
        x = rs[k, :, 0]
        y = rs[k, :, 1]

        if temporal:
            Vx, Vy = func(x, y, tk, N_fields)
        else:
            Vx, Vy = func(x, y, 0.0, N_fields)

        V = cp.column_stack((Vx, Vy))
        dW = cp.sqrt(dt_N) * cp.random.normal(size=(N, 2))
        rs[k + 1] = rs[k] + V * dt_N + dW

    return t, rs

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
