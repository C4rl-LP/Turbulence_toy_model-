import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import simular_estocastico_com_perturbacao, calcular_estatisticas_trajetorias


def executar_teste_estocasticidade(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3, 4],
    lista_delta_r: list[float] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    t_max: float = 1.0,
    N_particulas: int = 300,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "estocasticidade"),
    seed: int = 42
) -> dict:
    """
    3. Teste de estocasticidade espontânea no limite delta_r -> 0.
    Investiga se existe dispersão estatística não trivial sem afirmar conclusões físicas automaticamente.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    resultados_globais = {}

    print("=" * 60)
    print("EXECUTANDO TESTE DE ESTOCASTICIDADE ESPONTÂNEA")
    print("=" * 60)

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
            stats = calcular_estatisticas_trajetorias(ts, rs)

            dev = rs_final - np.mean(rs_final, axis=0)
            disp_media = np.mean(np.sqrt(np.sum(dev**2, axis=1)))

            res_n['delta_r'].append(delta_r)
            res_n['mean_x'].append(stats['mean_x'][-1])
            res_n['mean_y'].append(stats['mean_y'][-1])
            res_n['var_x'].append(stats['var_x'][-1])
            res_n['var_y'].append(stats['var_y'][-1])
            res_n['sigma2'].append(stats['sigma2'][-1])
            res_n['dispersao_media'].append(disp_media)
            res_n['distribuicoes_finais'].append(rs_final)

            # Histograma 2D / Scatter
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

        np.savez_compressed(
            os.path.join(sub_dir, f"estocasticidade_N_{n}_dados.npz"),
            delta_r=np.array(res_n['delta_r']),
            mean_x=np.array(res_n['mean_x']),
            mean_y=np.array(res_n['mean_y']),
            var_x=np.array(res_n['var_x']),
            var_y=np.array(res_n['var_y']),
            sigma2=np.array(res_n['sigma2']),
            dispersao_media=np.array(res_n['dispersao_media']),
            distribuicoes_finais=np.array(res_n['distribuicoes_finais'], dtype=object)
        )

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

        print(f"\nRESUMO QUANTITATIVO DE ESTOCASTICIDADE (N_fields = {n}):")
        print(f"{'delta_r':<15}{'Dispersão':<15}{'sigma2':<15}")
        print("-" * 45)
        for i in range(len(deltas)):
            print(f"{deltas[i]:<15.5e}{res_n['dispersao_media'][i]:<15.5e}{res_n['sigma2'][i]:<15.5e}")

    print("=" * 60 + "\n")
    return resultados_globais


if __name__ == "__main__":
    print("Módulo test_estocasticidade.py pronto para ser chamado.")
