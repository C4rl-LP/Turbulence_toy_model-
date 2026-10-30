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


def criar_campo_parametrizado(c_val: float, alpha_val: float):
    """
    Retorna uma função de velocidade que avalia a superposição hierárquica
    de vórtices configurando os parâmetros c (inclinação da função bump)
    e alpha (fator de corte de raio de influência: R_cutoff = alpha * R(n)).
    Reutiliza a função campos.campo_total_podado com a árvore podada.
    """
    def campo(x, y, t, N_fields):
        return campos.campo_total_podado(
            xp=x,
            yp=y,
            t=t,
            n_max=N_fields,
            c=c_val,
            alpha=alpha_val
        )

    return campo


def executar_teste_estocasticidade_parametros(
    N_fields: int = 6,
    r0=np.array([0.3, 0.2333]),
    lista_alpha=None,
    lista_c=None,
    N_particulas: int = 500,
    t_max: float = 0.5,
    dt_ref: float = None,
    temporal: bool = False,
    num_amostras: int = 1,
    delta_r0: float = .1,
    output_dir: str =None,
    seed: int = 42
) -> dict:
    """
    Investiga a dependência da estocasticidade espontânea nos parâmetros
    alpha_cut e c do campo de velocidades determinístico, para um nível N_fields fixo.

    Parâmetros:
    -----------
    N_fields : int
        Nível do modelo mantido fixo durante a varredura (ex: N=3).
    r0 : array_like, shape (2,)
        Condição inicial de referência central.
    lista_alpha : iterable de floats ou None
        Valores de alpha_cut a serem investigados.
    lista_c : iterable de floats ou None
        Valores do parâmetro c a serem investigados.
    N_particulas : int
        Número de partículas por simulação.
    t_max : float
        Tempo máximo de integração temporal.
    dt_ref : float ou None
        Passo temporal de integração. Se None, adota 0.01 * fc.T(N_fields).
    temporal : bool
        Se True, campo depende explicitamente de t; se False, campo com centros em t=0.
    num_amostras : int
        Número de repetições com sementes diferentes para avaliar robustez estatística.
    output_dir : str
        Diretório dos dados numéricos e metadados; imagens são geradas separadamente.
    seed : int
        Semente inicial para reprodutibilidade.
    """

    if output_dir is None:
        nivel = f'em_nivel_{N_fields}'
        if temporal:
            output_dir =  os.path.join("resultados", "estocasticidade_parametros_temporal", nivel)
        else:
            output_dir =  os.path.join("resultados", "estocasticidade_parametros_estatico", nivel)

    os.makedirs(output_dir, exist_ok=True)
    pasta_info = os.path.join(output_dir, "informacoes")
    os.makedirs(pasta_info, exist_ok=True)

    if lista_alpha is None:
        lista_alpha = [ float(alpha_padrao), 1.8, 2.0, 2.2, 2.4]
    if lista_c is None:
        lista_c = [0.2, 0.4, float(c_padrao), 0.8, 1.0]

    lista_alpha = [float(a) for a in lista_alpha]
    lista_c = [float(c) for c in lista_c]

    r0 = np.asarray(r0, dtype=float).reshape(2)
    norm_r0 = float(np.linalg.norm(r0))

    R_N = float(fc.R(N_fields))
    delta_r = float(norm_r0 * R_N)
    dimensao_quadrado = delta_r0 * R_N

    if dt_ref is None:
        dt = 0.01 * float(fc.T(N_fields))
    else:
        dt = dt_ref

    nome_do_txt_info = os.path.join(pasta_info, "Informações.txt")
    with open(nome_do_txt_info, "w", encoding="utf-8") as f:
        f.write("CARACTERÍSTICAS DA VARREDURA NO ESPAÇO DE PARÂMETROS (alpha_cut, c)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Nível fixo N_fields: {N_fields}\n")
        f.write(f"R_1 = {R_1}, T_1 = {T_1}, lambda = {lamb}\n")
        f.write(f"alpha_padrao = {alpha_padrao:.4f}, c_padrao = {c_padrao:.4f}\n")
        f.write(f"r_0 = {r0.tolist()}, ||r_0|| = {norm_r0:.6e}\n")
        f.write(f"delta_r = ||\\delta r0|| * R(N) = {dimensao_quadrado:.6e}\n")
        f.write(f"Tamanho do quadrado/diâmetro L = {dimensao_quadrado:.6e}\n")
        f.write(f"Número de partículas: {N_particulas}\n")
        f.write(f"Tempo de integração t_max: {t_max}, dt = {dt:.6e}\n")
        f.write(f"Campo dependente do tempo: {temporal}\n")
        f.write(f"Número de amostras por ponto: {num_amostras}\n")
        f.write("Dinâmica: determinística RK4 (via it.simular_particulas)\n\n")
        f.write(f"Valores testados de alpha: {lista_alpha}\n")
        f.write(f"Valores testados de c: {lista_c}\n\n")
        f.write("ESTATÍSTICAS DETALHADAS POR COMBINAÇÃO:\n")
        f.write("-" * 70 + "\n\n")

    print("=" * 75)
    print("VARREDURA DE ESTOCASTICIDADE ESPONTÂNEA NO ESPAÇO (alpha_cut, c)")
    print("=" * 75)
    print(f"Nível fixo N_fields = {N_fields} | delta_r = {dimensao_quadrado:.6e}")
    print(f"Posição inicial de referência r0: {r0}")
    print(f"Número de partículas: {N_particulas} | t_max: {t_max} | dt: {dt:.6e}")
    print(f"Grade: {len(lista_alpha)} valores de alpha x {len(lista_c)} valores de c ({len(lista_alpha)*len(lista_c)} pontos)")
    print("-" * 75)

    n_alpha = len(lista_alpha)
    n_c = len(lista_c)

    matriz_sigma2_final = np.zeros((n_c, n_alpha))
    matriz_sigma2_desvio = np.zeros((n_c, n_alpha))
    matriz_disp_media_final = np.zeros((n_c, n_alpha))
    matriz_norm_sigma2 = np.zeros((n_c, n_alpha))

    dados_pontos = {}

    t_inicio_total = time.perf_counter()

    for idx_c, c_val in enumerate(lista_c):
        for idx_a, alpha_val in enumerate(lista_alpha):
            chave = f"alpha_{alpha_val:.4f}_c_{c_val:.4f}"
            pasta_ponto = os.path.join(output_dir, f"alpha_{alpha_val:.3f}_c_{c_val:.3f}")
            os.makedirs(pasta_ponto, exist_ok=True)

            campo_func = criar_campo_parametrizado(c_val=c_val, alpha_val=alpha_val)

            amostras_sigma2 = []
            amostras_disp = []
            amostras_mean = []
            amostras_var = []
            amostras_cov = []

            ts_ref = None
            rs_ref = None
            stats_ref = None

            t0_ponto = time.perf_counter()

            for s_idx in range(num_amostras):
                seed_amostra = seed + (idx_c * n_alpha + idx_a) * 100 + s_idx
                np.random.seed(seed_amostra)

                ts, rs = it.simular_particulas(
                    N=N_particulas,
                    N_fields=N_fields,
                    dimensao_quadrado=dimensao_quadrado,
                    r0=r0,
                    dt=dt,
                    t_max=t_max,
                    func=campo_func,
                    temporal=temporal
                )

                stats = calcular_estatisticas_trajetorias(ts, rs)
                # Converter uma vez para as agregações, arquivos e gráficos na CPU.
                rs = cp.asnumpy(rs)
                rs_final = rs[-1, :, :]
                mean_f = np.mean(rs_final, axis=0)
                var_f = np.var(rs_final, axis=0, ddof=1)
                cov_f = np.cov(rs_final.T, ddof=1)
                dev_f = rs_final - mean_f
                s2_f = float(np.mean(np.sum(dev_f**2, axis=1)))
                disp_f = float(np.mean(np.sqrt(np.sum(dev_f**2, axis=1))))

                amostras_sigma2.append(s2_f)
                amostras_disp.append(disp_f)
                amostras_mean.append(mean_f)
                amostras_var.append(var_f)
                amostras_cov.append(cov_f)

                if s_idx == 0:
                    ts_ref = ts
                    rs_ref = rs
                    stats_ref = stats

            t1_ponto = time.perf_counter()

            sigma2_medio = float(np.mean(amostras_sigma2))
            sigma2_std = float(np.std(amostras_sigma2, ddof=1)) if num_amostras > 1 else 0.0
            disp_media = float(np.mean(amostras_disp))
            norm_sigma2 = sigma2_medio / (dimensao_quadrado**2)

            matriz_sigma2_final[idx_c, idx_a] = sigma2_medio
            matriz_sigma2_desvio[idx_c, idx_a] = sigma2_std
            matriz_disp_media_final[idx_c, idx_a] = disp_media
            matriz_norm_sigma2[idx_c, idx_a] = norm_sigma2

            mean_final_ref = np.mean(amostras_mean, axis=0)
            var_final_ref = np.mean(amostras_var, axis=0)
            cov_final_ref = np.mean(amostras_cov, axis=0)

            print(
                f"[alpha = {alpha_val:6.3f}, c = {c_val:5.2f}] -> "
                f"sigma² = {sigma2_medio:10.4e} | norm(sigma²/delta_r²) = {norm_sigma2:8.2f} | "
                f"disp = {disp_media:10.4e} ({t1_ponto - t0_ponto:5.2f}s)"
            )

            with open(nome_do_txt_info, "a", encoding="utf-8") as f:
                f.write(f"alpha_cut = {alpha_val:.4f}, c = {c_val:.4f}\n")
                f.write(f"Média final: {mean_final_ref}, Variância final: {var_final_ref}\n")
                f.write(f"Sigma² final: {sigma2_medio:.6e} (std: {sigma2_std:.4e})\n")
                f.write(f"Dispersão normalizada sigma²/delta_r²: {norm_sigma2:.4f}\n")
                f.write("Matriz de covariância:\n")
                f.write(f"[[{cov_final_ref[0, 0]:.4e}, {cov_final_ref[0, 1]:.4e}],\n")
                f.write(f" [{cov_final_ref[1, 0]:.4e}, {cov_final_ref[1, 1]:.4e}]]\n")
                f.write("=-=" * 10 + "\n")

            # Salvar dados do ponto
            salvar_npz(
                os.path.join(pasta_ponto, "dados_ponto.npz"),
                alpha_cut=alpha_val,
                c=c_val,
                N_fields=N_fields,
                delta_r=delta_r,
                t=ts_ref,
                rs_ref=rs_ref,
                sigma2_t=stats_ref["sigma2"],
                msd_t=stats_ref["msd"],
                var_x=stats_ref["var_x"],
                var_y=stats_ref["var_y"],
                cov_xy=stats_ref["cov_xy"],
                mean_final=mean_final_ref,
                var_final=var_final_ref,
                cov_final=cov_final_ref,
                sigma2_final=sigma2_medio,
                disp_media_final=disp_media,
                norm_sigma2=norm_sigma2
            )


            dados_pontos[chave] = {
                "alpha": alpha_val,
                "c": c_val,
                "sigma2_final": sigma2_medio,
                "sigma2_std": sigma2_std,
                "disp_media_final": disp_media,
                "norm_sigma2": norm_sigma2
            }

    t_fim_total = time.perf_counter()
    print("-" * 75)
    print(f"Varredura concluída em {t_fim_total - t_inicio_total:.2f} s.")


    # 1. Mapa de Calor de sigma² final no plano (alpha_cut, c)

    # 2. Mapa de Calor da Dispersão Normalizada sigma² / delta_r² (escala log)

    # 3. Evolução temporal de sigma²(t) para curvas representativas

    # 4. Cortes unidimensionais: sigma² vs c para diferentes alpha

    # 5. Cortes unidimensionais: sigma² vs alpha para diferentes c

    # Salva o resumo global numérico da varredura
    salvar_npz(
        os.path.join(output_dir, "resumo_varredura.npz"),
        N_fields=N_fields,
        r0=r0,
        delta_r=dimensao_quadrado,
        lista_alpha=np.array(lista_alpha),
        lista_c=np.array(lista_c),
        matriz_sigma2_final=matriz_sigma2_final,
        matriz_sigma2_desvio=matriz_sigma2_desvio,
        matriz_disp_media_final=matriz_disp_media_final,
        matriz_norm_sigma2=matriz_norm_sigma2
    )

    print("\n" + "=" * 75)
    print("RESUMO SINTÉTICO DA VARREDURA:")
    print("=" * 75)
    print(f"Menor sigma² final observado: {np.min(matriz_sigma2_final):.6e}")
    print(f"Maior sigma² final observado: {np.max(matriz_sigma2_final):.6e}")
    print(f"Razão max/min de sigma² final: {np.max(matriz_sigma2_final)/max(np.min(matriz_sigma2_final), 1e-15):.2f}")
    print(f"Menor razão sigma²/delta_r²:  {np.min(matriz_norm_sigma2):.4f}")
    print(f"Maior razão sigma²/delta_r²:  {np.max(matriz_norm_sigma2):.4f}")
    print("=" * 75)
    print(f"Todos os resultados foram salvos em: {output_dir}\n")

    return {
        "N_fields": N_fields,
        "delta_r": dimensao_quadrado,
        "lista_alpha": lista_alpha,
        "lista_c": lista_c,
        "matriz_sigma2_final": matriz_sigma2_final,
        "matriz_disp_media_final": matriz_disp_media_final,
        "matriz_norm_sigma2": matriz_norm_sigma2,
        "dados_pontos": dados_pontos
    }


if __name__ == "__main__":
    executar_teste_estocasticidade_parametros(N_fields=7, temporal=True)
