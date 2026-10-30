import os
import sys
import time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.dados_resultados import salvar_npz
import integradores as it
import campos
import funcoes_para_centros as fc
from funcoes_para_centros import (
    R_1,
    T_1,
    lamb,
    alpha_padrao,
    c_padrao
)
from tests.utils_estatisticas import (
    calcular_estatisticas_trajetorias
)


def executar_teste_estocasticidade(
    r0=np.array([0.3, 0.2333]),
    lista_N_fields=(1, 2, 3),
    t_max: float = 1.0,
    N_particulas: int = 300,
    dt_ref: float = None,
    func=None,
    temporal: bool = False,
    output_dir: str = None,
    delta_r0 = 0.1,
    seed: int = 42
) -> dict:
    """
    Teste numérico de estocasticidade espontânea sob o campo determinístico.

    Para cada nível n in lista_N_fields:
      - Distribui N_particulas uniformemente em área num disco centrado em r0 com raio
            delta_r_n = ||r0|| * R(n)
        usando a função it.simular_particulas (com dimensao_quadrado = 2 * delta_r_n).
      - Evolui as partículas deterministicamente via RK4 (sem ruído estocástico).
      - Acompanha a evolução temporal das partículas e calcula:
            * Posição média r_medio(t)
            * Variâncias var_x(t), var_y(t)
            * Covariância cov_xy(t)
            * Dispersão quadrática média sigma²(t) = (1/N) * sum_i ||r_i(t) - r_medio(t)||²
      - Gera arquivos de dados numéricos (.npz), estatísticas formatadas (Informações.txt)
        para visualização posterior com tests/plot_resultados.py.
    """
    if output_dir is None:
        if temporal:
            output_dir =os.path.join("resultados", "estocasticidade_temporal")
        else:
            output_dir = os.path.join("resultados", "estocasticidade_estatico")
    os.makedirs(output_dir, exist_ok=True)

    pasta_info = os.path.join(output_dir, "informacoes")
    os.makedirs(pasta_info, exist_ok=True)

    if func is None:
        func = campos.campo_total_podado

    r0 = np.asarray(r0, dtype=float).reshape(2)
    norm_r0 = float(np.linalg.norm(r0))

    if seed is not None:
        np.random.seed(seed)

    nome_do_txt_info = os.path.join(pasta_info, "Informações.txt")
    with open(nome_do_txt_info, "w", encoding="utf-8") as f:
        f.write("Características\n\n")
        f.write(f"R_1 = {R_1}, T_1 = {T_1}, lambda = {lamb}\n")
        f.write(f"alpha de corte = {alpha_padrao}, Fator c = {c_padrao}\n")
        f.write(f"r_0 = {r0.tolist()}, Número de partículas: {N_particulas}\n")
        f.write(f"Tempo total (t_max): {t_max}\n")
        f.write("Dinâmica: determinística (solve_RK4 via it.simular_particulas)\n")
        f.write("Condição inicial: disco uniforme de raio delta_r_n = ||r0|| * R(n)\n\n")
        f.write("Estatísticas:\n\n")

    resultados_globais = {}

    print("=" * 70)
    print("TESTE NUMÉRICO DE ESTOCASTICIDADE ESPONTÂNEA (DETERMINÍSTICO)")
    print("=" * 70)
    print(f"Condição inicial de referência r0: {r0}")
    print(f"||r0||: {norm_r0:.6e}")
    print(f"Número de partículas: {N_particulas}")
    print(f"Tempo de simulação t_max: {t_max}")
    print(f"Campo dependente do tempo: {temporal}")
    print("Perturbação inicial: delta_r_n = ||r0|| * R(n)")
    print("-" * 70)

    for n in lista_N_fields:
        t0_n = time.perf_counter()

        sub_dir = os.path.join(output_dir, f"N_fields_{n}")
        os.makedirs(sub_dir, exist_ok=True)

        R_n = float(fc.R(n))

        dimensao_quadrado = delta_r0 * R_n

        if dt_ref is None:
            dt_n = 0.01 * float(fc.T(n))
        else:
            dt_n = dt_ref

        print(f"\n[Nível n = {n}]")
        print(f"  R({n}) = {R_n:.6e}")
        print(f"  delta_r_{n} = {dimensao_quadrado:.6e}")
        print(f"  dt = {dt_n:.6e}")

        # -------------------------------------------------------------
        # 1. Simulação com a função oficial do projeto (determinística)
        # -------------------------------------------------------------
        ts, rs = it.simular_particulas(
            N=N_particulas,
            N_fields=n,
            dimensao_quadrado=dimensao_quadrado,
            r0=r0,
            dt=dt_n,
            t_max=t_max,
            func=func,
            temporal=temporal
        )

        t1_n = time.perf_counter()

        # -------------------------------------------------------------
        # 2. Estatísticas temporais e dispersão em relação ao centroide
        # -------------------------------------------------------------
        stats = calcular_estatisticas_trajetorias(ts, rs)

        # NumPy, Matplotlib e os arquivos de saída recebem arrays na CPU.
        rs = cp.asnumpy(rs)
        rs_final = rs[-1, :, :]
        mean_final = np.mean(rs_final, axis=0)
        var_final = np.var(rs_final, axis=0, ddof=1)
        cov_final = np.cov(rs_final.T, ddof=1)

        dev_final = rs_final - mean_final
        sigma2_final = float(np.mean(np.sum(dev_final**2, axis=1)))
        dispersao_media_final = float(np.mean(np.sqrt(np.sum(dev_final**2, axis=1))))

        # -------------------------------------------------------------
        # 3. Registro no arquivo de informações (seguindo integradores.py)
        # -------------------------------------------------------------
        with open(nome_do_txt_info, "a", encoding="utf-8") as f:
            f.write(f"Nível: {n}, delta_r_n = {dimensao_quadrado:.6e}, Tamanho da nuvem inicial (L) = {dimensao_quadrado:.6e}\n")
            f.write(f"Média: {mean_final}, Variância: {var_final}\n")
            f.write(f"Sigma^2 raio = {sigma2_final:.6e}\n")
            f.write("Matriz de covariância:\n")
            f.write(f"[[{cov_final[0, 0]:.4e}, {cov_final[0, 1]:.4e}],\n")
            f.write(f" [{cov_final[1, 0]:.4e}, {cov_final[1, 1]:.4e}]]\n\n")
            f.write("=-=" * 10 + "\n")

        # -------------------------------------------------------------
        # 4. Salvar dados numéricos completos
        # -------------------------------------------------------------
        salvar_npz(
            os.path.join(sub_dir, f"estocasticidade_nivel_{n}_dados.npz"),
            N_fields=n,
            R_n=R_n,
            delta_r_n=dimensao_quadrado,
            dimensao_quadrado=dimensao_quadrado,
            t=ts,
            rs=rs,
            mean_x=stats["mean_x"],
            mean_y=stats["mean_y"],
            var_x=stats["var_x"],
            var_y=stats["var_y"],
            cov_xy=stats["cov_xy"],
            sigma2=stats["sigma2"],
            msd=stats["msd"],
            mean_final=mean_final,
            var_final=var_final,
            cov_final=cov_final,
            sigma2_final=sigma2_final,
            dispersao_media_final=dispersao_media_final
        )


        print(f"  Tempo decorrido: {t1_n - t0_n:.2f} s")
        print(f"  sigma² final: {sigma2_final:.6e}")
        print(f"  Dispersão média final: {dispersao_media_final:.6e}")
        print(f"  Média final: [{mean_final[0]:.4f}, {mean_final[1]:.4f}]")

        resultados_globais[f"N_{n}"] = {
            "N_fields": n,
            "R_n": R_n,
            "delta_r_n": dimensao_quadrado,
            "ts": ts,
            "mean_x": stats["mean_x"],
            "mean_y": stats["mean_y"],
            "var_x": stats["var_x"],
            "var_y": stats["var_y"],
            "cov_xy": stats["cov_xy"],
            "sigma2": stats["sigma2"],
            "msd": stats["msd"],
            "mean_final": mean_final,
            "var_final": var_final,
            "cov_final": cov_final,
            "sigma2_final": sigma2_final,
            "dispersao_media_final": dispersao_media_final,
            "tempo_execucao": t1_n - t0_n
        }


    niveis = np.array(list(lista_N_fields))
    deltas = np.array([resultados_globais[f"N_{n}"]["delta_r_n"] for n in niveis])
    sigma2_finais = np.array([resultados_globais[f"N_{n}"]["sigma2_final"] for n in niveis])
    dispersoes_finais = np.array([resultados_globais[f"N_{n}"]["dispersao_media_final"] for n in niveis])


    # Resumo global compactado
    salvar_npz(
        os.path.join(output_dir, "resumo_global.npz"),
        niveis=niveis,
        delta_r_n=deltas,
        sigma2_final=sigma2_finais,
        dispersao_media_final=dispersoes_finais
    )

    print("\n" + "=" * 75)
    print("RESUMO GLOBAL DE ESTOCASTICIDADE ESPONTÂNEA")
    print("=" * 75)
    print(f"{'Nível (n)':<12}{'delta_r_n':<18}{'sigma² final':<20}{'Dispersão média':<20}")
    print("-" * 75)
    for i, n in enumerate(niveis):
        print(f"{n:<12}{deltas[i]:<18.6e}{sigma2_finais[i]:<20.6e}{dispersoes_finais[i]:<20.6e}")
    print("=" * 75)
    print(f"Dados e estatísticas salvos em: {output_dir}\n")

    return resultados_globais


if __name__ == "__main__":
    executar_teste_estocasticidade(temporal=False)
