import os
import sys
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
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

    # Conversão para CPU para salvar e plotar
    rs_det_cpu = cp.asnumpy(rs_det)
    rs_estoc_cpu = cp.asnumpy(rs_estoc)

    # Salvar dados brutos
    salvar_npz(
        os.path.join(output_dir, "comparacao_simuladores_dados.npz"),
        ts_det=ts_det, rs_det=rs_det_cpu,
        ts_estoc=ts_estoc, rs_estoc=rs_estoc_cpu,
        stats_det=stats_det,
        stats_estoc=stats_estoc
    )


    # A) Trajetórias comparadas

    # B) Distribuição final comparada

    # C) Médias sobrepostas vs t

    # D) Variância sobreposta vs t

    # E) sigma² total sobreposto vs t

    # F) MSD sobreposto vs t

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
