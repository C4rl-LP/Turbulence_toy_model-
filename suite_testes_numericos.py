import os
import time
import numpy as np
import matplotlib.pyplot as plt

import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb

# ==============================================================================
# FUNÇÕES AUXILIARES DE CÁLCULO E SIMULAÇÃO
# ==============================================================================

def calcular_estatisticas_trajetorias(t: np.ndarray, rs: np.ndarray) -> dict:
    """
    Calcula estatísticas e séries temporais a partir das trajetórias.

    Parâmetros
    ----------
    t : ndarray, shape (Nt,)
        Vetor de tempos da simulação.
    rs : ndarray, shape (Nt, N, 2)
        Trajetórias das N partículas nos Nt instantes de tempo.

    Retorna
    -------
    dict : Dicionário contendo as séries temporais:
        - 't': tempos
        - 'mean_x', 'mean_y': médias temporais em x e y
        - 'var_x', 'var_y': variâncias amostrais em x e y
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

    # Distância média das trajetórias em relação ao centroide
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
    Simula N partículas sob dinâmica estocástica (Euler-Maruyama) com perturbação inicial:
        dr = V_N(r, t) dt + dW
        r(0) = r0 + delta_r
    """
    rng = np.random.default_rng(seed)
    r0 = np.asarray(r0, dtype=float).reshape(-1)

    # Distribuição inicial uniforme em um círculo de raio delta_r ao redor de r0
    theta = rng.uniform(0, 2 * np.pi, size=N)
    u = rng.uniform(0, 1, size=N)
    r_rand = delta_r * np.sqrt(u)

    dx = r_rand * np.cos(theta)
    dy = r_rand * np.sin(theta)

    particle_t0 = r0 + np.column_stack((dx, dy))  # shape (N, 2)

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


# ==============================================================================
# TESTE 1: TESTE DE CONVERGÊNCIA TEMPORAL
# ==============================================================================

def teste_1_convergencia_temporal(
    r0=np.array([0.3, 0.2333]),
    N_fields: int = 3,
    t_max: float = 1.0,
    N_particulas: int = 500,
    lista_dt: list[float] = None,
    func=None,
    temporal: bool = False,
    output_dir: str = "resultados_teste_1",
    seed: int = 42
):
    """
    1. Teste de convergência temporal para o simulador estocástico:
       dr = V_N(r,t) dt + dW

    Muda dt mantendo fixos N_fields, r0, t_max, N_particulas.
    Calcula média, variância, covariância, sigma^2 total e distribuição 2D final.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    dt_base = 0.01 * (fc.T(1) / (lamb ** (N_fields - 1)))
    if lista_dt is None:
        lista_dt = [dt_base, dt_base / 2.0, dt_base / 4.0, dt_base / 8.0]

    resultados = {
        'dt': [],
        'mean_x': [],
        'mean_y': [],
        'var_x': [],
        'var_y': [],
        'cov_xy': [],
        'sigma2': [],
        'distribuicoes_finais': []
    }

    for idx, dt in enumerate(lista_dt):
        np.random.seed(seed + idx)
        # simular_particulas_ruidosa calcula dt_N = dt_1 / lamb^(N_fields - 1).
        # Para forçar o passo ser dt, ajustamos dt_1:
        dt_1_ajustado = dt * (lamb ** (N_fields - 1))

        ts, rs = it.simular_particulas_ruidosa(
            N=N_particulas,
            N_fields=N_fields,
            r0=r0,
            dt_1=dt_1_ajustado,
            t_max=t_max,
            lamb=lamb,
            func=func,
            temporal=temporal
        )

        rs_final = rs[-1, :, :]
        mean = np.mean(rs_final, axis=0)
        var = np.var(rs_final, axis=0, ddof=1)
        cov_mat = np.cov(rs_final.T, ddof=1)
        cov_xy = cov_mat[0, 1] if N_particulas > 1 else 0.0
        sigma2 = np.mean(np.sum((rs_final - mean)**2, axis=1))

        resultados['dt'].append(dt)
        resultados['mean_x'].append(mean[0])
        resultados['mean_y'].append(mean[1])
        resultados['var_x'].append(var[0])
        resultados['var_y'].append(var[1])
        resultados['cov_xy'].append(cov_xy)
        resultados['sigma2'].append(sigma2)
        resultados['distribuicoes_finais'].append(rs_final)

        # Histograma 2D final para este dt
        plt.figure(figsize=(6, 5))
        plt.hist2d(rs_final[:, 0], rs_final[:, 1], bins=40, density=True, cmap='viridis')
        plt.colorbar(label='Densidade de probabilidade')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Distribuição 2D final (dt = {dt:.5e})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribuicao_2d_dt_{idx+1}.png'), dpi=200)
        plt.close()

    # Salvar dados numéricos
    np.savez_compressed(
        os.path.join(output_dir, "convergencia_temporal.npz"),
        dt=np.array(resultados['dt']),
        mean_x=np.array(resultados['mean_x']),
        mean_y=np.array(resultados['mean_y']),
        var_x=np.array(resultados['var_x']),
        var_y=np.array(resultados['var_y']),
        cov_xy=np.array(resultados['cov_xy']),
        sigma2=np.array(resultados['sigma2']),
        distribuicoes_finais=np.array(resultados['distribuicoes_finais'], dtype=object)
    )

    # Gráficos das estatísticas vs dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    dts = np.array(resultados['dt'])

    axes[0, 0].plot(dts, resultados['mean_x'], 'o-', label='<x>')
    axes[0, 0].plot(dts, resultados['mean_y'], 's-', label='<y>')
    axes[0, 0].set_xlabel('dt')
    axes[0, 0].set_ylabel('Média Final')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title('Média final vs dt')

    axes[0, 1].plot(dts, resultados['var_x'], 'o-', label='Var[x]')
    axes[0, 1].plot(dts, resultados['var_y'], 's-', label='Var[y]')
    axes[0, 1].set_xlabel('dt')
    axes[0, 1].set_ylabel('Variância Final')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_title('Variância final vs dt')

    axes[1, 0].plot(dts, resultados['cov_xy'], 'd-r', label='Cov[x,y]')
    axes[1, 0].set_xlabel('dt')
    axes[1, 0].set_ylabel('Covariância')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_title('Covariância vs dt')

    axes[1, 1].plot(dts, resultados['sigma2'], '^-g', label=r'$\sigma^2$ total')
    axes[1, 1].set_xlabel('dt')
    axes[1, 1].set_ylabel(r'$\sigma^2$')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_title(r'$\sigma^2$ total vs dt')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'estatisticas_vs_dt.png'), dpi=200)
    plt.close()

    return resultados


# ==============================================================================
# TESTE 2: TESTE ESPECÍFICO DA ESCALA TEMPORAL COM N (N_fields)
# ==============================================================================

def teste_2_escala_temporal_N(
    r0=np.array([0.3, 0.2333]),
    N_min: int = 1,
    N_max: int = 6,
    dt_1: float = 0.03 * fc.T(1),
    t_max: float = 1.0,
    N_particulas: int = 300,
    func=None,
    temporal: bool = False,
    output_dir: str = "resultados_teste_2",
    seed: int = 42
):
    """
    2. Teste da regra dt_N = dt_1 / lambda^(N_fields - 1) para N_fields de N_min até N_max.
    Registra: dt_N, passos temporais, tempo computacional, média, variância, sigma^2, covariância.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    n_fields_list = list(range(N_min, N_max + 1))

    dados = {
        'N_fields': [],
        'dt_N': [],
        'num_passos': [],
        'tempo_computacional': [],
        'mean_x': [],
        'mean_y': [],
        'var_x': [],
        'var_y': [],
        'cov_xy': [],
        'sigma2': []
    }

    for n_idx, n in enumerate(n_fields_list):
        np.random.seed(seed + n_idx)

        t_inicio = time.perf_counter()
        ts, rs = it.simular_particulas_ruidosa(
            N=N_particulas,
            N_fields=n,
            r0=r0,
            dt_1=dt_1,
            t_max=t_max,
            lamb=lamb,
            func=func,
            temporal=temporal
        )
        t_fim = time.perf_counter()

        tempo_comp = t_fim - t_inicio
        dt_N_calc = dt_1 / (lamb ** (n - 1))
        num_steps = len(ts) - 1

        rs_final = rs[-1, :, :]
        mean = np.mean(rs_final, axis=0)
        var = np.var(rs_final, axis=0, ddof=1)
        cov_mat = np.cov(rs_final.T, ddof=1)
        cov_xy = cov_mat[0, 1] if N_particulas > 1 else 0.0
        sigma2 = np.mean(np.sum((rs_final - mean)**2, axis=1))

        dados['N_fields'].append(n)
        dados['dt_N'].append(dt_N_calc)
        dados['num_passos'].append(num_steps)
        dados['tempo_computacional'].append(tempo_comp)
        dados['mean_x'].append(mean[0])
        dados['mean_y'].append(mean[1])
        dados['var_x'].append(var[0])
        dados['var_y'].append(var[1])
        dados['cov_xy'].append(cov_xy)
        dados['sigma2'].append(sigma2)

    # Salvar em .npz
    np.savez_compressed(
        os.path.join(output_dir, "escala_temporal_N.npz"),
        N_fields=np.array(dados['N_fields']),
        dt_N=np.array(dados['dt_N']),
        num_passos=np.array(dados['num_passos']),
        tempo_computacional=np.array(dados['tempo_computacional']),
        mean_x=np.array(dados['mean_x']),
        mean_y=np.array(dados['mean_y']),
        var_x=np.array(dados['var_x']),
        var_y=np.array(dados['var_y']),
        cov_xy=np.array(dados['cov_xy']),
        sigma2=np.array(dados['sigma2'])
    )

    # Salvar também em CSV para fácil navegação
    header = "N_fields,dt_N,num_passos,tempo_computacional,mean_x,mean_y,var_x,var_y,cov_xy,sigma2"
    matrix_data = np.column_stack([
        dados['N_fields'], dados['dt_N'], dados['num_passos'], dados['tempo_computacional'],
        dados['mean_x'], dados['mean_y'], dados['var_x'], dados['var_y'],
        dados['cov_xy'], dados['sigma2']
    ])
    np.savetxt(os.path.join(output_dir, "escala_temporal_N.csv"), matrix_data, delimiter=",", header=header, comments="")

    # Gerar gráficos separados
    N_arr = np.array(dados['N_fields'])

    # 1. dt_N vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['dt_N'], 'o-b')
    plt.yscale('log')
    plt.xlabel('N (N_fields)')
    plt.ylabel('dt_N (escala log)')
    plt.title('Passo temporal dt_N vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dt_N_vs_N.png'), dpi=200)
    plt.close()

    # 2. Número de passos vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['num_passos'], 's-g')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Número de passos temporais')
    plt.title('Número de passos vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_passos_vs_N.png'), dpi=200)
    plt.close()

    # 3. sigma² vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['sigma2'], '^-r')
    plt.xlabel('N (N_fields)')
    plt.ylabel(r'$\sigma^2$ total final')
    plt.title(r'$\sigma^2$ total final vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_vs_N.png'), dpi=200)
    plt.close()

    # 4. Variância em x e y vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['var_x'], 'o-', label='Var[x]')
    plt.plot(N_arr, dados['var_y'], 's-', label='Var[y]')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Variância final')
    plt.title('Variância final em x e y vs N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variancia_x_y_vs_N.png'), dpi=200)
    plt.close()

    # 5. Custo computacional vs N
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['tempo_computacional'], 'd-m')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Tempo de execução (segundos)')
    plt.title('Custo computacional vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'custo_computacional_vs_N.png'), dpi=200)
    plt.close()

    return dados


# ==============================================================================
# TESTE 3: TESTE DE ESTOCASTICIDADE ESPONTÂNEA
# ==============================================================================

def teste_3_estocasticidade_espontanea(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3, 4],
    lista_delta_r: list[float] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    t_max: float = 1.0,
    N_particulas: int = 300,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = "resultados_teste_3",
    seed: int = 42
):
    """
    3. Teste de estocasticidade espontânea no limite delta_r -> 0.
    Investiga a dispersão estatística final para diferentes amplitudes de perturbação inicial.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    resultados_globais = {}

    for n_idx, n in enumerate(lista_N_fields):
        sub_dir = os.path.join(output_dir, f"N_fields_{n}")
        os.makedirs(sub_dir, exist_ok=True)

        res_n = {
            'delta_r': [],
            'mean_x': [],
            'mean_y': [],
            'var_x': [],
            'var_y': [],
            'sigma2': [],
            'dispersao_media': [],
            'distribuicoes_finais': []
        }

        for d_idx, delta_r in enumerate(lista_delta_r):
            # Utiliza a função determinística com tamanho de nuvem 2*delta_r
            # ou simulação estocástica com perturbação inicial.
            ts, rs = simular_estocastico_com_perturbacao(
                N=N_particulas,
                N_fields=n,
                r0=r0,
                delta_r=delta_r,
                dt_1=dt_1,
                t_max=t_max,
                lamb_val=lamb,
                func=func,
                temporal=temporal,
                seed=seed + n_idx * 100 + d_idx
            )

            rs_final = rs[-1, :, :]
            mean = np.mean(rs_final, axis=0)
            var = np.var(rs_final, axis=0, ddof=1)
            sigma2 = np.mean(np.sum((rs_final - mean)**2, axis=1))

            # Separação/dispersão média entre trajetórias
            dev = rs_final - mean
            disp_media = np.mean(np.sqrt(np.sum(dev**2, axis=1)))

            res_n['delta_r'].append(delta_r)
            res_n['mean_x'].append(mean[0])
            res_n['mean_y'].append(mean[1])
            res_n['var_x'].append(var[0])
            res_n['var_y'].append(var[1])
            res_n['sigma2'].append(sigma2)
            res_n['dispersao_media'].append(disp_media)
            res_n['distribuicoes_finais'].append(rs_final)

            # Histograma/Scatter da distribuição final para cada delta_r
            plt.figure(figsize=(6, 5))
            plt.scatter(rs_final[:, 0], rs_final[:, 1], alpha=0.5, s=10, c='blue', label=f'Final (dr={delta_r:.1e})')
            plt.scatter(r0[0], r0[1], c='red', marker='x', s=80, label='r0')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Distribuição final (N_fields={n}, dr={delta_r:.1e})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(sub_dir, f'distribuicao_delta_r_{d_idx+1}.png'), dpi=200)
            plt.close()

        resultados_globais[f'N_{n}'] = res_n

        # Salvar npz para N_fields = n
        np.savez_compressed(
            os.path.join(sub_dir, f"estocasticidade_espontanea_N_{n}.npz"),
            delta_r=np.array(res_n['delta_r']),
            mean_x=np.array(res_n['mean_x']),
            mean_y=np.array(res_n['mean_y']),
            var_x=np.array(res_n['var_x']),
            var_y=np.array(res_n['var_y']),
            sigma2=np.array(res_n['sigma2']),
            dispersao_media=np.array(res_n['dispersao_media']),
            distribuicoes_finais=np.array(res_n['distribuicoes_finais'], dtype=object)
        )

        # Gráficos em função de delta_r para o nível n
        deltas = np.array(res_n['delta_r'])

        plt.figure(figsize=(6, 5))
        plt.plot(deltas, res_n['dispersao_media'], 'o-b')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\delta r$ inicial')
        plt.ylabel('Dispersão média final')
        plt.title(fr'Dispersão final vs $\delta r$ (N_fields={n})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, 'dispersao_vs_delta_r.png'), dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(deltas, res_n['var_x'], 'o-', label='Var[x]')
        plt.plot(deltas, res_n['var_y'], 's-', label='Var[y]')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\delta r$ inicial')
        plt.ylabel('Variância final')
        plt.title(fr'Variância final vs $\delta r$ (N_fields={n})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, 'variancia_vs_delta_r.png'), dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.plot(deltas, res_n['sigma2'], '^-r')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\delta r$ inicial')
        plt.ylabel(r'$\sigma^2$ total final')
        plt.title(fr'$\sigma^2$ final vs $\delta r$ (N_fields={n})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, 'sigma2_vs_delta_r.png'), dpi=200)
        plt.close()

    return resultados_globais


# ==============================================================================
# TESTE 4: SEPARAR RUÍDO INICIAL DE RUÍDO DINÂMICO
# ==============================================================================

def teste_4_separar_ruidos(
    r0=np.array([0.3, 0.2333]),
    N_fields: int = 3,
    delta_r: float = 1e-3,
    t_max: float = 1.0,
    N_particulas: int = 500,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = "resultados_teste_4",
    seed: int = 42
):
    """
    4. Compara três cenários de ruído:
       A) Determinístico com perturbação inicial: dr/dt = V_N(r,t), r(0) = r0 + delta_r
       B) Estocástico sem perturbação inicial: dr = V_N(r,t) dt + dW, r(0) = r0
       C) Estocástico com perturbação inicial: dr = V_N(r,t) dt + dW, r(0) = r0 + delta_r
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    dt_N = dt_1 / (lamb ** (N_fields - 1))

    # --- Caso A: Determinístico + perturbação inicial ---
    np.random.seed(seed)
    L = 2.0 * delta_r
    ts_A, rs_A = it.simular_particulas(
        N=N_particulas,
        N_fields=N_fields,
        dimensao_quadrado=L,
        r0=r0,
        dt=dt_N,
        t_max=t_max,
        func=func,
        temporal=temporal
    )
    stats_A = calcular_estatisticas_trajetorias(ts_A, rs_A)

    # --- Caso B: Estocástico sem perturbação inicial ---
    np.random.seed(seed + 1)
    ts_B, rs_B = it.simular_particulas_ruidosa(
        N=N_particulas,
        N_fields=N_fields,
        r0=r0,
        dt_1=dt_1,
        t_max=t_max,
        lamb=lamb,
        func=func,
        temporal=temporal
    )
    stats_B = calcular_estatisticas_trajetorias(ts_B, rs_B)

    # --- Caso C: Estocástico com perturbação inicial ---
    ts_C, rs_C = simular_estocastico_com_perturbacao(
        N=N_particulas,
        N_fields=N_fields,
        r0=r0,
        delta_r=delta_r,
        dt_1=dt_1,
        t_max=t_max,
        lamb_val=lamb,
        func=func,
        temporal=temporal,
        seed=seed + 2
    )
    stats_C = calcular_estatisticas_trajetorias(ts_C, rs_C)

    # Salvar resultados
    np.savez_compressed(
        os.path.join(output_dir, "comparacao_ruidos.npz"),
        t_A=ts_A, rs_A=rs_A,
        t_B=ts_B, rs_B=rs_B,
        t_C=ts_C, rs_C=rs_C,
        sigma2_A=stats_A['sigma2'], sigma2_B=stats_B['sigma2'], sigma2_C=stats_C['sigma2'],
        var_x_A=stats_A['var_x'], var_x_B=stats_B['var_x'], var_x_C=stats_C['var_x'],
        var_y_A=stats_A['var_y'], var_y_B=stats_B['var_y'], var_y_C=stats_C['var_y'],
        dist_A=stats_A['dist_media_trajetorias'],
        dist_B=stats_B['dist_media_trajetorias'],
        dist_C=stats_C['dist_media_trajetorias']
    )

    # Gráficos comparativos

    # 1. Variância em x e y vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['var_x'], label='A (Det + pert. inic.) - Var[x]', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['var_x'], label='B (Estoc. sem pert.) - Var[x]', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['var_x'], label='C (Estoc. + pert.) - Var[x]', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel('Variância em x')
    plt.title('Comparação de Variância Var[x](t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_variancia_x.png'), dpi=200)
    plt.close()

    # 2. sigma² total vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['sigma2'], label='A (Det + pert. inic.)', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['sigma2'], label='B (Estoc. sem pert.)', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['sigma2'], label='C (Estoc. + pert.)', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel(r'$\sigma^2(t)$ total')
    plt.title(r'Comparação de $\sigma^2(t)$ Total')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_sigma2.png'), dpi=200)
    plt.close()

    # 3. Distância média entre trajetórias vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['dist_media_trajetorias'], label='A (Det + pert. inic.)', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['dist_media_trajetorias'], label='B (Estoc. sem pert.)', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['dist_media_trajetorias'], label='C (Estoc. + pert.)', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel('Distância média ao centroide')
    plt.title('Distância média entre trajetórias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_distancia_trajetorias.png'), dpi=200)
    plt.close()

    # 4. Distribuições finais comparadas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    axes[0].scatter(rs_A[-1, :, 0], rs_A[-1, :, 1], alpha=0.5, s=12, color='blue')
    axes[0].set_title('A: Det. + perturbação')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(rs_B[-1, :, 0], rs_B[-1, :, 1], alpha=0.5, s=12, color='green')
    axes[1].set_title('B: Estocástico puro')
    axes[1].set_xlabel('x')
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(rs_C[-1, :, 0], rs_C[-1, :, 1], alpha=0.5, s=12, color='red')
    axes[2].set_title('C: Estoc. + perturbação')
    axes[2].set_xlabel('x')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_distribuicoes_finais.png'), dpi=200)
    plt.close()

    return {'stats_A': stats_A, 'stats_B': stats_B, 'stats_C': stats_C}


# ==============================================================================
# TESTE 5: ESTATÍSTICAS AO LONGO DO TEMPO
# ==============================================================================

def teste_5_estatisticas_temporais(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3, 4],
    t_max: float = 1.0,
    N_particulas: int = 500,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = "resultados_teste_5",
    seed: int = 42
):
    """
    5. Estatísticas ao longo do tempo para cada instante t_k:
       <x(t)>, <y(t)>, Var[x(t)], Var[y(t)], Cov[x,y](t), sigma^2(t), MSD(t).
    Repete e compara para diferentes N_fields.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    series_temporais_n = {}

    for n_idx, n in enumerate(lista_N_fields):
        np.random.seed(seed + n_idx)

        ts, rs = it.simular_particulas_ruidosa(
            N=N_particulas,
            N_fields=n,
            r0=r0,
            dt_1=dt_1,
            t_max=t_max,
            lamb=lamb,
            func=func,
            temporal=temporal
        )

        stats = calcular_estatisticas_trajetorias(ts, rs)
        series_temporais_n[n] = stats

        # Salvar série temporal individual do nível n
        np.savez_compressed(
            os.path.join(output_dir, f"series_temporais_N_{n}.npz"),
            t=ts,
            mean_x=stats['mean_x'],
            mean_y=stats['mean_y'],
            var_x=stats['var_x'],
            var_y=stats['var_y'],
            cov_xy=stats['cov_xy'],
            sigma2=stats['sigma2'],
            msd=stats['msd']
        )

    # Criar gráficos comparativos das séries temporais entre níveis N_fields
    figuras_info = [
        ('mean_x', 'Média <x>(t)', 'media_x_vs_t.png'),
        ('mean_y', 'Média <y>(t)', 'media_y_vs_t.png'),
        ('var_x', 'Variância Var[x](t)', 'variancia_x_vs_t.png'),
        ('var_y', 'Variância Var[y](t)', 'variancia_y_vs_t.png'),
        ('cov_xy', 'Covariância Cov[x,y](t)', 'covariancia_vs_t.png'),
        ('sigma2', r'$\sigma^2(t)$ Total', 'sigma2_vs_t.png'),
        ('msd', 'MSD(t)', 'msd_vs_t.png')
    ]

    for chave, titulo, filename in figuras_info:
        plt.figure(figsize=(7, 5))
        for n in lista_N_fields:
            st = series_temporais_n[n]
            plt.plot(st['t'], st[chave], label=f'N_fields = {n}')
        plt.xlabel('Tempo t')
        plt.ylabel(titulo)
        plt.title(f'{titulo} vs t')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=200)
        plt.close()

    return series_temporais_n


# ==============================================================================
# CONFIGURAÇÃO GERAL E EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    print("Suíte de testes numéricos criada com sucesso!")
    print("Módulos de teste disponíveis:")
    print(" - teste_1_convergencia_temporal(...)")
    print(" - teste_2_escala_temporal_N(...)")
    print(" - teste_3_estocasticidade_espontanea(...)")
    print(" - teste_4_separar_ruidos(...)")
    print(" - teste_5_estatisticas_temporais(...)")
    print("\nAVISO: Os testes não foram executados automaticamente conforme solicitado.")
