import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
import integradores as it
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def executar_teste_robustez(
    r0=np.array([0.3, 0.2333]),
    N_fields: int = 3,
    lista_N_particulas: list[int] = [100, 300, 1000, 3000],
    t_max: float = 1.0,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "robustez"),
    seed: int = 42
) -> dict:
    """
    10. Teste de robustez estatística:
        Analisa a convergência e os erros relativos das estimativas de média, variância, sigma^2 e MSD
        variando o número de realizações (N_particulas).
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    dados_robustez = {
        'N_particulas': lista_N_particulas,
        'mean_x': [],
        'mean_y': [],
        'var_x': [],
        'var_y': [],
        'sigma2': [],
        'msd': []
    }

    print("=" * 60)
    print("EXECUTANDO TESTE DE ROBUSTEZ ESTATÍSTICA (CONVERGÊNCIA COM N_particulas)")
    print("=" * 60)

    for idx, n_part in enumerate(lista_N_particulas):
        np.random.seed(seed + idx)

        ts, rs = it.simular_particulas_ruidosa(
            N=n_part,
            N_fields=N_fields,
            r0=r0,
            dt_1=dt_1,
            t_max=t_max,
            lamb=lamb,
            func=func,
            temporal=temporal
        )

        stats = calcular_estatisticas_trajetorias(ts, rs)

        dados_robustez['mean_x'].append(stats['mean_x'][-1])
        dados_robustez['mean_y'].append(stats['mean_y'][-1])
        dados_robustez['var_x'].append(stats['var_x'][-1])
        dados_robustez['var_y'].append(stats['var_y'][-1])
        dados_robustez['sigma2'].append(stats['sigma2'][-1])
        dados_robustez['msd'].append(stats['msd'][-1])

    # Salvar dados brutos
    salvar_npz(
        os.path.join(output_dir, "robustez_estatistica_dados.npz"),
        N_particulas=np.array(lista_N_particulas),
        mean_x=np.array(dados_robustez['mean_x']),
        mean_y=np.array(dados_robustez['mean_y']),
        var_x=np.array(dados_robustez['var_x']),
        var_y=np.array(dados_robustez['var_y']),
        sigma2=np.array(dados_robustez['sigma2']),
        msd=np.array(dados_robustez['msd'])
    )

    n_pts = np.array(lista_N_particulas)


    print("\nRESUMO QUANTITATIVO DE ROBUSTEZ ESTATÍSTICA (t_max):")
    print(f"{'N_particulas':<15}{'<x>':<12}{'Var[x]':<15}{'sigma2':<15}{'MSD':<15}")
    print("-" * 72)
    for i in range(len(n_pts)):
        print(f"{n_pts[i]:<15}{dados_robustez['mean_x'][i]:<12.4f}{dados_robustez['var_x'][i]:<15.4e}"
              f"{dados_robustez['sigma2'][i]:<15.4e}{dados_robustez['msd'][i]:<15.4e}")
    print("=" * 60 + "\n")

    return dados_robustez


if __name__ == "__main__":
    print("Módulo test_robustez_estatistica.py pronto para ser chamado.")
