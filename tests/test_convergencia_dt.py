import os
import sys
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def executar_teste_convergencia_dt(
    r0=np.array([0.3, 0.2333]),
    N_fields: int = 3,
    t_max: float = 1.0,
    N_particulas: int = 500,
    lista_dt: list[float] = None,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "convergencia_dt"),
    seed: int = 42
) -> dict:
    """
    1. Teste de convergência temporal para o simulador estocástico:
       dr = V_N(r,t) dt + dW

    Muda o passo dt mantendo fixos N_fields, r0, t_max, N_particulas.
    Saves: dados brutos .npz, resumo .csv, gráficos em resultados/convergencia_dt/.
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

    print("=" * 60)
    print("EXECUTANDO TESTE DE CONVERGÊNCIA TEMPORAL (dt)")
    print("=" * 60)

    for idx, dt in enumerate(lista_dt):
        np.random.seed(seed + idx)
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

        rs_final = cp.asnumpy(rs[-1, :, :])
        stats = calcular_estatisticas_trajetorias(ts, rs)

        resultados['dt'].append(dt)
        resultados['mean_x'].append(stats['mean_x'][-1])
        resultados['mean_y'].append(stats['mean_y'][-1])
        resultados['var_x'].append(stats['var_x'][-1])
        resultados['var_y'].append(stats['var_y'][-1])
        resultados['cov_xy'].append(stats['cov_xy'][-1])
        resultados['sigma2'].append(stats['sigma2'][-1])
        resultados['distribuicoes_finais'].append(rs_final)

        # Gráfico de Histograma 2D final para este dt
        plt.figure(figsize=(6, 5))
        plt.hist2d(rs_final[:, 0], rs_final[:, 1], bins=40, density=True, cmap='viridis')
        plt.colorbar(label='Densidade')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Distribuição 2D final (dt = {dt:.5e})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribuicao_2d_dt_{idx+1}.png'), dpi=200)
        plt.close()

    # Salvar dados brutos
    np.savez_compressed(
        os.path.join(output_dir, "convergencia_dt_dados.npz"),
        dt=np.array(resultados['dt']),
        mean_x=np.array(resultados['mean_x']),
        mean_y=np.array(resultados['mean_y']),
        var_x=np.array(resultados['var_x']),
        var_y=np.array(resultados['var_y']),
        cov_xy=np.array(resultados['cov_xy']),
        sigma2=np.array(resultados['sigma2']),
        distribuicoes_finais=np.array(resultados['distribuicoes_finais'], dtype=object)
    )

    # Gráfico comparativo vs dt
    dts = np.array(resultados['dt'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

    # Resumo quantitativo
    print("\nRESUMO QUANTITATIVO DE CONVERGÊNCIA TEMPORAL (dt):")
    print(f"{'dt':<15}{'<x>':<12}{'<y>':<12}{'Var[x]':<12}{'Var[y]':<12}{'sigma2':<12}")
    print("-" * 75)
    for i in range(len(dts)):
        print(f"{dts[i]:<15.5e}{resultados['mean_x'][i]:<12.4f}{resultados['mean_y'][i]:<12.4f}"
              f"{resultados['var_x'][i]:<12.4e}{resultados['var_y'][i]:<12.4e}{resultados['sigma2'][i]:<12.4e}")
    print("=" * 60 + "\n")

    return resultados


if __name__ == "__main__":
    print("Módulo test_convergencia_dt.py pronto para ser chamado.")
