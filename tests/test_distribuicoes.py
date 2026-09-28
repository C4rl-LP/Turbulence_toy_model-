import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import integradores as it
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import (
    calcular_estatisticas_distribuicao,
    gerar_histogramas_completos
)


def executar_teste_distribuicoes(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 3, 5],
    lista_dt: list[float] = [0.02, 0.005],
    t_max: float = 1.0,
    N_particulas: int = 1000,
    func=None,
    output_dir: str = os.path.join("resultados", "distribuicoes"),
    seed: int = 42
) -> dict:
    """
    9. Análise de distribuições completas:
       Gera histogramas 1D de x, 1D de y, 2D de (x,y) e calcula assimetria, curtose e quantis.
       Compara distribuições entre N_fields, dt, condições iniciais e simuladores (det vs estoc).
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    relatorio_estatistico = {}

    print("=" * 60)
    print("EXECUTANDO ANÁLISE DE DISTRIBUIÇÕES COMPLETAS (1D, 2D, ASSIMETRIA, CURTOSE, QUANTIS)")
    print("=" * 60)

    # 1. Comparação entre N_fields
    for n_idx, n in enumerate(lista_N_fields):
        np.random.seed(seed + n_idx)

        ts, rs = it.simular_particulas_ruidosa(
            N=N_particulas,
            N_fields=n,
            r0=r0,
            dt_1=0.01,
            t_max=t_max,
            lamb=lamb,
            func=func
        )

        rs_final = rs[-1, :, :]
        x_final = rs_final[:, 0]
        y_final = rs_final[:, 1]

        stats_x = calcular_estatisticas_distribuicao(x_final)
        stats_y = calcular_estatisticas_distribuicao(y_final)

        key = f"N_fields_{n}"
        relatorio_estatistico[key] = {
            'stats_x': stats_x,
            'stats_y': stats_y,
            'rs_final': rs_final
        }

        gerar_histogramas_completos(
            rs_final,
            title_prefix=f'N_fields = {n}',
            output_dir=output_dir,
            filename_prefix=f'distribuicao_N_{n}'
        )

    # 2. Comparação entre Simulador Determinístico vs Estocástico
    np.random.seed(seed)
    _, rs_det = it.simular_particulas(
        N=N_particulas,
        N_fields=3,
        dimensao_quadrado=1e-3,
        r0=r0,
        dt=0.01,
        t_max=t_max,
        func=func
    )
    gerar_histogramas_completos(
        rs_det[-1, :, :],
        title_prefix='Simulador Determinístico (N=3)',
        output_dir=output_dir,
        filename_prefix='distribuicao_deterministic'
    )

    relatorio_estatistico['deterministic'] = {
        'stats_x': calcular_estatisticas_distribuicao(rs_det[-1, :, 0]),
        'stats_y': calcular_estatisticas_distribuicao(rs_det[-1, :, 1]),
        'rs_final': rs_det[-1, :, :]
    }

    # Salvar resultados brutos e resumo em tabela .csv
    np.savez_compressed(
        os.path.join(output_dir, "distribuicoes_dados.npz"),
        relatorio=relatorio_estatistico
    )

    # Imprimir tabela detalhada de estatísticas de ordem superior
    print("\nTABELA ESTATÍSTICA DE DISTRIBUIÇÃO (Eixo X):")
    print(f"{'Caso':<20}{'Média':<12}{'DesvPad':<12}{'Assimetria':<12}{'Curtose':<12}{'Q50(Mediana)':<12}")
    print("-" * 80)
    for k, v in relatorio_estatistico.items():
        sx = v['stats_x']
        print(f"{k:<20}{sx['mean']:<12.4f}{sx['std']:<12.4e}{sx['assimetria']:<12.4f}{sx['curtose']:<12.4f}{sx['q50']:<12.4f}")

    print("\nTABELA ESTATÍSTICA DE DISTRIBUIÇÃO (Eixo Y):")
    print(f"{'Caso':<20}{'Média':<12}{'DesvPad':<12}{'Assimetria':<12}{'Curtose':<12}{'Q50(Mediana)':<12}")
    print("-" * 80)
    for k, v in relatorio_estatistico.items():
        sy = v['stats_y']
        print(f"{k:<20}{sy['mean']:<12.4f}{sy['std']:<12.4e}{sy['assimetria']:<12.4f}{sy['curtose']:<12.4f}{sy['q50']:<12.4f}")
    print("=" * 60 + "\n")

    return relatorio_estatistico


if __name__ == "__main__":
    print("Módulo test_distribuicoes.py pronto para ser chamado.")
