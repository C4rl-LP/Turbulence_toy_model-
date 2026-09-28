import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def executar_teste_comparacao_simuladores(
    r0=np.array([0.3, 0.2333]),
    N_fields: int = 3,
    delta_r: float = 1e-3,
    t_max: float = 1.0,
    N_particulas: int = 500,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "comparacao"),
    seed: int = 42
) -> dict:
    """
    7. Comparação entre simulador determinístico (simular_particulas) e estocástico (simular_particulas_ruidosa)
    para o mesmo r0, N_fields, t_max e nuvem inicial de partículas.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    dt_N = dt_1 / (lamb ** (N_fields - 1))
    L = 2.0 * delta_r

    print("=" * 60)
    print("EXECUTANDO COMPARAÇÃO ENTRE SIMULADOR DETERMINÍSTICO E ESTOCÁSTICO")
    print("=" * 60)

    # 1. Simulador Determinístico (simular_particulas)
    np.random.seed(seed)
    ts_det, rs_det = it.simular_particulas(
        N=N_particulas,
        N_fields=N_fields,
        dimensao_quadrado=L,
        r0=r0,
        dt=dt_N,
        t_max=t_max,
        func=func,
        temporal=temporal
    )
    stats_det = calcular_estatisticas_trajetorias(ts_det, rs_det)

    # 2. Simulador Estocástico (simular_particulas_ruidosa)
    np.random.seed(seed)
    ts_estoc, rs_estoc = it.simular_particulas_ruidosa(
        N=N_particulas,
        N_fields=N_fields,
        r0=r0,
        dt_1=dt_1,
        t_max=t_max,
        lamb=lamb,
        func=func,
        temporal=temporal
    )
    stats_estoc = calcular_estatisticas_trajetorias(ts_estoc, rs_estoc)

    # Salvar dados brutos
    np.savez_compressed(
        os.path.join(output_dir, "comparacao_simuladores_dados.npz"),
        ts_det=ts_det, rs_det=rs_det,
        ts_estoc=ts_estoc, rs_estoc=rs_estoc,
        stats_det=stats_det,
        stats_estoc=stats_estoc
    )

    # Gráficos sobrepostos:

    # A) Trajetórias comparadas
    plt.figure(figsize=(7, 7))
    for i in range(min(50, N_particulas)):
        plt.plot(rs_det[:, i, 0], rs_det[:, i, 1], 'b-', alpha=0.3, lw=0.8, label='Determinístico' if i == 0 else "")
        plt.plot(rs_estoc[:, i, 0], rs_estoc[:, i, 1], 'r-', alpha=0.3, lw=0.8, label='Estocástico' if i == 0 else "")
    plt.scatter(r0[0], r0[1], c='black', marker='x', s=100, label='r0')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Trajetórias Sobrepostas (N_fields = {N_fields})')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajetorias_comparadas.png'), dpi=200)
    plt.close()

    # B) Distribuição final comparada
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(rs_det[-1, :, 0], rs_det[-1, :, 1], color='blue', alpha=0.5, s=15)
    axes[0].set_title('Distribuição Final - Determinístico')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(rs_estoc[-1, :, 0], rs_estoc[-1, :, 1], color='red', alpha=0.5, s=15)
    axes[1].set_title('Distribuição Final - Estocástico')
    axes[1].set_xlabel('x')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribuicao_final_comparada.png'), dpi=200)
    plt.close()

    # C) Médias sobrepostas vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_det, stats_det['mean_x'], 'b--', label='Det <x>')
    plt.plot(ts_det, stats_det['mean_y'], 'b:', label='Det <y>')
    plt.plot(ts_estoc, stats_estoc['mean_x'], 'r-', label='Estoc <x>')
    plt.plot(ts_estoc, stats_estoc['mean_y'], 'r-.', label='Estoc <y>')
    plt.xlabel('Tempo t')
    plt.ylabel('Média da Posição')
    plt.title('Comparação de Médias da Posição')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'medias_comparadas.png'), dpi=200)
    plt.close()

    # D) Variância sobreposta vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_det, stats_det['var_x'], 'b--', label='Det Var[x]')
    plt.plot(ts_det, stats_det['var_y'], 'b:', label='Det Var[y]')
    plt.plot(ts_estoc, stats_estoc['var_x'], 'r-', label='Estoc Var[x]')
    plt.plot(ts_estoc, stats_estoc['var_y'], 'r-.', label='Estoc Var[y]')
    plt.xlabel('Tempo t')
    plt.ylabel('Variância')
    plt.title('Comparação de Variâncias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variancias_comparadas.png'), dpi=200)
    plt.close()

    # E) sigma² total sobreposto vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_det, stats_det['sigma2'], 'b--', label='Determinístico')
    plt.plot(ts_estoc, stats_estoc['sigma2'], 'r-', label='Estocástico')
    plt.xlabel('Tempo t')
    plt.ylabel(r'$\sigma^2(t)$ Total')
    plt.title(r'Comparação de $\sigma^2(t)$ Total')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_comparado.png'), dpi=200)
    plt.close()

    # F) MSD sobreposto vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts_det, stats_det['msd'], 'b--', label='Determinístico')
    plt.plot(ts_estoc, stats_estoc['msd'], 'r-', label='Estocástico')
    plt.xlabel('Tempo t')
    plt.ylabel('MSD(t)')
    plt.title('Comparação de MSD(t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'msd_comparado.png'), dpi=200)
    plt.close()

    print("\nRESUMO QUANTITATIVO DA COMPARAÇÃO DE SIMULADORES (t_max):")
    print(f"{'Modelo':<18}{'<x>':<12}{'<y>':<12}{'Var[x]':<12}{'sigma2':<12}{'MSD':<12}")
    print("-" * 78)
    print(f"{'Determinístico':<18}{stats_det['mean_x'][-1]:<12.4f}{stats_det['mean_y'][-1]:<12.4f}"
          f"{stats_det['var_x'][-1]:<12.4e}{stats_det['sigma2'][-1]:<12.4e}{stats_det['msd'][-1]:<12.4e}")
    print(f"{'Estocástico':<18}{stats_estoc['mean_x'][-1]:<12.4f}{stats_estoc['mean_y'][-1]:<12.4f}"
          f"{stats_estoc['var_x'][-1]:<12.4e}{stats_estoc['sigma2'][-1]:<12.4e}{stats_estoc['msd'][-1]:<12.4e}")
    print("=" * 60 + "\n")

    return {'det': stats_det, 'estoc': stats_estoc}


if __name__ == "__main__":
    print("Módulo test_comparacao_simuladores.py pronto para ser chamado.")
