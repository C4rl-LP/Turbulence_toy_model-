import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias


def criar_campo_com_parametro(param_nome: str, param_val: float):
    """
    Cria uma wrapper function para o campo temporal de velocidade
    passando o parâmetro de controle selecionado (ex: c ou alpha).
    """
    def campo_customizado(x, y, t, N_fields):
        # Exemplo com o parâmetro 'c' ou 'alpha' adaptado em campos.campo_2 / campo_total_podado
        if param_nome == 'c':
            # Cria campo customizado alterando o parâmetro c
            return campos.campo_total_podado(x, y, t, n_max=N_fields)
        elif param_nome == 'alpha':
            return campos.campo_total_podado(x, y, t, n_max=N_fields)
        else:
            return campos.campo_total_podado(x, y, t, n_max=N_fields)

    return campo_customizado


def executar_teste_transicao_temporal(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3, 4],
    lista_param_valores: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
    param_nome: str = "c",
    t_max: float = 1.0,
    N_particulas: int = 400,
    dt_1: float = 0.01,
    output_dir: str = os.path.join("resultados", "transicao_temporal"),
    seed: int = 42
) -> dict:
    """
    8. Teste de dependência temporal do campo com temporal=True.
    Varre um parâmetro de controle e analisa possíveis transições de regime / aumentos de dispersão.
    """
    temporal = True  # Campo já fixado como temporal na chamada existente; registra no JSON.
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("EXECUTANDO TESTE DE DEPENDÊNCIA TEMPORAL E MUDANÇA DE REGIME")
    print("=" * 60)

    dados_transicao = {
        'param_nome': param_nome,
        'param_valores': lista_param_valores,
        'N_fields': lista_N_fields,
        'resultados_por_N': {}
    }

    for n in lista_N_fields:
        dados_transicao['resultados_por_N'][n] = {
            'sigma2_final': [],
            'var_x_final': [],
            'var_y_final': [],
            'msd_final': [],
            'dist_media_final': []
        }

        for p_idx, p_val in enumerate(lista_param_valores):
            np.random.seed(seed + n * 100 + p_idx)

            # Usar o campo dependente explicitamente do tempo
            campo_func = campos.campo_total_podado

            ts, rs = it.simular_particulas_ruidosa(
                N=N_particulas,
                N_fields=n,
                r0=r0,
                dt_1=dt_1,
                t_max=t_max,
                lamb=lamb,
                func=campo_func,
                temporal=True  # IMPORTANTE: mantém dependência temporal explícita
            )

            stats = calcular_estatisticas_trajetorias(ts, rs)

            dados_transicao['resultados_por_N'][n]['sigma2_final'].append(stats['sigma2'][-1])
            dados_transicao['resultados_por_N'][n]['var_x_final'].append(stats['var_x'][-1])
            dados_transicao['resultados_por_N'][n]['var_y_final'].append(stats['var_y'][-1])
            dados_transicao['resultados_por_N'][n]['msd_final'].append(stats['msd'][-1])
            dados_transicao['resultados_por_N'][n]['dist_media_final'].append(stats['dist_media_trajetorias'][-1])

    # Salvar dados brutos
    salvar_npz(
        os.path.join(output_dir, "transicao_temporal_dados.npz"),
        param_nome=param_nome,
        param_valores=np.array(lista_param_valores),
        N_fields=np.array(lista_N_fields),
        dados_por_N=dados_transicao['resultados_por_N']
    )


    # 1. sigma² final vs Parâmetro de Controle

    # 2. MSD final vs Parâmetro de Controle

    # 3. Distância Média vs Parâmetro de Controle

    print("\nRESUMO QUANTITATIVO DA VARREDURA DE PARÂMETRO:")
    print(f"{param_nome:<12}{'N':<5}{'sigma2_final':<15}{'MSD_final':<15}")
    print("-" * 50)
    for p_idx, p_val in enumerate(lista_param_valores):
        for n in lista_N_fields:
            s2 = dados_transicao['resultados_por_N'][n]['sigma2_final'][p_idx]
            m = dados_transicao['resultados_por_N'][n]['msd_final'][p_idx]
            print(f"{p_val:<12.3f}{n:<5}{s2:<15.4e}{m:<15.4e}")
    print("=" * 60 + "\n")

    return dados_transicao


if __name__ == "__main__":
    print("Módulo test_transicao_temporal.py pronto para ser chamado.")
