import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def executar_teste_convergencia_N(
    r0=np.array([0.3, 0.2333]),
    N_min: int = 1,
    N_max: int = 6,
    dt_1: float = 0.03 * fc.T(1),
    t_max: float = 1.0,
    N_particulas: int = 300,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "convergencia_N"),
    seed: int = 42
) -> dict:
    """
    2. Teste da regra de escala temporal dt_N = dt_1 / lambda^(N_fields - 1).
    Varia N_fields de N_min até N_max e avalia custo computacional, dt_N e estatísticas.
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

    print("=" * 60)
    print("EXECUTANDO TESTE DE ESCALA TEMPORAL COM N (N_fields)")
    print("=" * 60)

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

        stats = calcular_estatisticas_trajetorias(ts, rs)

        dados['N_fields'].append(n)
        dados['dt_N'].append(dt_N_calc)
        dados['num_passos'].append(num_steps)
        dados['tempo_computacional'].append(tempo_comp)
        dados['mean_x'].append(stats['mean_x'][-1])
        dados['mean_y'].append(stats['mean_y'][-1])
        dados['var_x'].append(stats['var_x'][-1])
        dados['var_y'].append(stats['var_y'][-1])
        dados['cov_xy'].append(stats['cov_xy'][-1])
        dados['sigma2'].append(stats['sigma2'][-1])

    # Salvar em .npz
    np.savez_compressed(
        os.path.join(output_dir, "convergencia_N_dados.npz"),
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

    # Salvar em CSV
    header = "N_fields,dt_N,num_passos,tempo_computacional,mean_x,mean_y,var_x,var_y,cov_xy,sigma2"
    matrix_data = np.column_stack([
        dados['N_fields'], dados['dt_N'], dados['num_passos'], dados['tempo_computacional'],
        dados['mean_x'], dados['mean_y'], dados['var_x'], dados['var_y'],
        dados['cov_xy'], dados['sigma2']
    ])
    np.savetxt(os.path.join(output_dir, "convergencia_N_resumo.csv"), matrix_data, delimiter=",", header=header, comments="")

    # Gerar gráficos separados
    N_arr = np.array(dados['N_fields'])

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

    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['num_passos'], 's-g')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Número de passos temporais')
    plt.title('Número de passos vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_passos_vs_N.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['sigma2'], '^-r')
    plt.xlabel('N (N_fields)')
    plt.ylabel(r'$\sigma^2$ total final')
    plt.title(r'$\sigma^2$ total final vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_vs_N.png'), dpi=200)
    plt.close()

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

    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['tempo_computacional'], 'd-m')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Tempo de execução (segundos)')
    plt.title('Custo computacional vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'custo_computacional_vs_N.png'), dpi=200)
    plt.close()

    print("\nRESUMO QUANTITATIVO DE ESCALA TEMPORAL COM N:")
    print(f"{'N':<5}{'dt_N':<15}{'Passos':<10}{'Tempo(s)':<12}{'sigma2':<12}")
    print("-" * 55)
    for i in range(len(N_arr)):
        print(f"{N_arr[i]:<5}{dados['dt_N'][i]:<15.5e}{dados['num_passos'][i]:<10}{dados['tempo_computacional'][i]:<12.4f}{dados['sigma2'][i]:<12.4e}")
    print("=" * 60 + "\n")

    return dados


if __name__ == "__main__":
    print("Módulo test_convergencia_N.py pronto para ser chamado.")
