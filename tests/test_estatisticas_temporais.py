import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
import integradores as it
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def executar_teste_estatisticas_temporais(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3],
    t_max: float = 1.0,
    N_particulas: int = 50,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "estatisticas_temporais"),
    seed: int = 42
) -> dict:
    """
    5. Estatísticas ao longo do tempo para cada instante t_k:
       <x(t)>, <y(t)>, Var[x(t)], Var[y(t)], Cov[x,y](t), sigma^2(t), MSD(t).
    Repete para diferentes N_fields e salva todas as séries temporais.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    series_temporais_n = {}

    print("=" * 60)
    print("EXECUTANDO TESTE DE ESTATÍSTICAS AO LONGO DO TEMPO")
    print("=" * 60)

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

        salvar_npz(
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

    print("\nRESUMO QUANTITATIVO DAS SÉRIES TEMPORAIS (t_max):")
    print(f"{'N':<5}{'<x>(t_max)':<15}{'Var[x](t_max)':<15}{'sigma2(t_max)':<15}{'MSD(t_max)':<15}")
    print("-" * 65)
    for n in lista_N_fields:
        st = series_temporais_n[n]
        print(f"{n:<5}{st['mean_x'][-1]:<15.4f}{st['var_x'][-1]:<15.4e}{st['sigma2'][-1]:<15.4e}{st['msd'][-1]:<15.4e}")
    print("=" * 60 + "\n")

    return series_temporais_n


if __name__ == "__main__":
    print("Módulo test_estatisticas_temporais.py pronto para ser chamado.")
