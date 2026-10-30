import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
import integradores as it
import campos
from funcoes_para_centros import lamb
from tests.utils_estatisticas import calcular_estatisticas_trajetorias, calcular_expoente_difusao


def executar_teste_difusao(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields: list[int] = [1, 2, 3, 4],
    t_max: float = 1.0,
    N_particulas: int = 500,
    dt_1: float = 0.01,
    func=None,
    temporal: bool = False,
    output_dir: str = os.path.join("resultados", "difusao"),
    seed: int = 42
) -> dict:
    """
    6. Teste de difusão:
       Analisa MSD(t) = <|r(t) - r(0)|^2> e calcula a estimativa local do expoente:
           alpha(t) = d log(MSD) / d log(t)
       sem impor previamente que o processo seja difusivo (alpha=1).
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func = campos.campo_total_podado

    resultados_difusao = {}

    print("=" * 60)
    print("EXECUTANDO TESTE DE DIFUSÃO E EXPOENTE ANÔMALO alpha(t)")
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
        msd = stats['msd']
        alpha_t = calcular_expoente_difusao(ts, msd)

        resultados_difusao[n] = {
            't': ts,
            'msd': msd,
            'alpha_t': alpha_t
        }

        # Salvar dados numéricos brutos por nível N
        salvar_npz(
            os.path.join(output_dir, f"difusao_N_{n}_dados.npz"),
            t=ts,
            msd=msd,
            alpha_t=alpha_t
        )

    print("\nRESUMO QUANTITATIVO DO EXPOENTE DIFUSIVO alpha(t):")
    print(f"{'N':<5}{'MSD(t_max)':<15}{'alpha_medio (t > t_max/2)':<25}")
    print("-" * 50)
    for n in lista_N_fields:
        res = resultados_difusao[n]
        valid_half = res['t'] > (t_max / 2)
        alpha_medio = np.mean(res['alpha_t'][valid_half]) if np.any(valid_half) else 0.0
        print(f"{n:<5}{res['msd'][-1]:<15.4e}{alpha_medio:<25.4f}")
    print("=" * 60 + "\n")

    return resultados_difusao


if __name__ == "__main__":
    print("Módulo test_difusao.py pronto para ser chamado.")
