"""Rotinas de gráficos extraídas dos drivers; somente NumPy e Matplotlib.

A seleção e a leitura de arquivos ficam em plot_resultados.py.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def comparacao_simuladores_trajetorias_comparadas_1(N_fields, N_particulas, output_dir, r0, rs_det_cpu, rs_estoc_cpu):
    plt.figure(figsize=(7, 7))
    for i in range(min(50, N_particulas)):
        plt.plot(rs_det_cpu[:, i, 0], rs_det_cpu[:, i, 1], 'b-', alpha=0.3, lw=0.8, label='Determinístico' if i == 0 else "")
        plt.plot(rs_estoc_cpu[:, i, 0], rs_estoc_cpu[:, i, 1], 'r-', alpha=0.3, lw=0.8, label='Estocástico' if i == 0 else "")
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

def comparacao_simuladores_distribuicao_final_comparada_2(output_dir, rs_det_cpu, rs_estoc_cpu):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(rs_det_cpu[-1, :, 0], rs_det_cpu[-1, :, 1], color='blue', alpha=0.5, s=15)
    axes[0].set_title('Distribuição Final - Determinístico')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(rs_estoc_cpu[-1, :, 0], rs_estoc_cpu[-1, :, 1], color='red', alpha=0.5, s=15)
    axes[1].set_title('Distribuição Final - Estocástico')
    axes[1].set_xlabel('x')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribuicao_final_comparada.png'), dpi=200)
    plt.close()

def comparacao_simuladores_medias_comparadas_3(output_dir, stats_det, stats_estoc, ts_det, ts_estoc):
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

def comparacao_simuladores_variancias_comparadas_4(output_dir, stats_det, stats_estoc, ts_det, ts_estoc):
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

def comparacao_simuladores_sigma2_comparado_5(output_dir, stats_det, stats_estoc, ts_det, ts_estoc):
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

def comparacao_simuladores_msd_comparado_6(output_dir, stats_det, stats_estoc, ts_det, ts_estoc):
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

def convergencia_N_dt_N_vs_N_1(N_arr, dados, output_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['dt_N'], 'o-b')
    plt.yscale('log')
    plt.xlabel('N (N_fields)')
    plt.ylabel('dt_N (escala log)')
    plt.title('Passo temporal dt_N vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dt_N_vs_N.png'), dpi=200)
    plt.close()

def convergencia_N_num_passos_vs_N_2(N_arr, dados, output_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['num_passos'], 's-g')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Número de passos temporais')
    plt.title('Número de passos vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'num_passos_vs_N.png'), dpi=200)
    plt.close()

def convergencia_N_sigma2_vs_N_3(N_arr, dados, output_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['sigma2'], '^-r')
    plt.xlabel('N (N_fields)')
    plt.ylabel(r'$\sigma^2$ total final')
    plt.title(r'$\sigma^2$ total final vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_vs_N.png'), dpi=200)
    plt.close()

def convergencia_N_variancia_x_y_vs_N_4(N_arr, dados, output_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['var_x'], 'o-', label='Var[x]')
    plt.plot(N_arr, dados['var_y'], 's-', label='Var[y]')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Variância final')
    plt.title('Variância final em x e y vs N')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variancia_x_y_vs_N.png'), dpi=200)
    plt.close()

def convergencia_N_custo_computacional_vs_N_5(N_arr, dados, output_dir):
    plt.figure(figsize=(6, 5))
    plt.plot(N_arr, dados['tempo_computacional'], 'd-m')
    plt.xlabel('N (N_fields)')
    plt.ylabel('Tempo de execução (segundos)')
    plt.title('Custo computacional vs N')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'custo_computacional_vs_N.png'), dpi=200)
    plt.close()

def convergencia_dt_distribuicao_2d_dt__1(dt, idx, output_dir, rs_final):
    plt.figure(figsize=(6, 5))
    plt.hist2d(rs_final[:, 0], rs_final[:, 1], bins=40, density=True, cmap='viridis')
    plt.colorbar(label='Densidade')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Distribuição 2D final (dt = {dt:.5e})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribuicao_2d_dt_{idx+1}.png'), dpi=200)
    plt.close()

def convergencia_dt_estatisticas_vs_dt_2(dts, output_dir, resultados):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(dts, resultados['mean_x'], 'o-', label='<x>')
    axes[0, 0].plot(dts, resultados['mean_y'], 's-', label='<y>')
    axes[0, 0].set_xlabel('dt')
    axes[0, 0].set_ylabel('Média Final')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_title('Média final vs dt')

    axes[0, 1].plot(dts, resultados['var_x'], 'o-', label='Var[x]')
    axes[0, 1].plot(dts, resultados['var_y'], 's-', label='Var[y]')
    axes[0, 1].set_xlabel('dt')
    axes[0, 1].set_ylabel('Variância Final')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_title('Variância final vs dt')

    axes[1, 0].plot(dts, resultados['cov_xy'], 'd-r', label='Cov[x,y]')
    axes[1, 0].set_xlabel('dt')
    axes[1, 0].set_ylabel('Covariância')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_title('Covariância vs dt')

    axes[1, 1].plot(dts, resultados['sigma2'], '^-g', label=r'$\sigma^2$ total')
    axes[1, 1].set_xlabel('dt')
    axes[1, 1].set_ylabel(r'$\sigma^2$')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_title(r'$\sigma^2$ total vs dt')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'estatisticas_vs_dt.png'), dpi=200)
    plt.close()

def estatisticas_temporais_figura_1(chave, filename, lista_N_fields, output_dir, series_temporais_n, titulo):
    plt.figure(figsize=(7, 5))
    for n in lista_N_fields:
        st = series_temporais_n[n]
        plt.plot(st['t'], st[chave], label=f'N_fields = {n}')
    plt.xlabel('Tempo t')
    plt.ylabel(titulo)
    plt.title(f'{titulo} vs t')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

def estocasticidade_trajetorias_nivel__1(N_particulas, n, r0, rs, sub_dir):
    plt.figure(figsize=(6, 6))
    cores = plt.cm.viridis(np.linspace(0, 1, N_particulas))

    for i in range(N_particulas):
        plt.plot(
            rs[:, i, 0],
            rs[:, i, 1],
            lw=1,
            alpha=0.4,
            color=cores[i]
        )
        plt.scatter(
            rs[0, i, 0],
            rs[0, i, 1],
            facecolors="none",
            edgecolors=cores[i],
            s=30,
            linewidths=1.2
        )
        plt.scatter(
            rs[-1, i, 0],
            rs[-1, i, 1],
            color=cores[i],
            s=30,
            zorder=3
        )

    plt.scatter(
        r0[0],
        r0[1],
        c="red",
        s=100,
        marker="x",
        linewidths=2,
        label=r"ponto $r_0$"
    )
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Estabilidade / Estocasticidade em torno de r0 — nível n = {n}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(sub_dir, f"trajetorias_nivel_{n}_em_x{r0[0]:.3f}_y{r0[1]:.2f}.png"),
        dpi=200
    )
    plt.close()

def estocasticidade_hist2d_2(n, pasta_info, r0, rs_final):
    plt.figure(figsize=(6, 6))
    plt.hist2d(
        rs_final[:, 0],
        rs_final[:, 1],
        bins=50,
        density=True,
        cmap="viridis"
    )
    plt.colorbar(label="densidade")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.title(f"Histograma 2D — nível {n}")
    plt.tight_layout()
    nome_hist = os.path.join(
        pasta_info,
        f"hist2d_nivel_{n}_x{r0[0]:.3f}_y{r0[1]:.2f}.png"
    )
    plt.savefig(nome_hist, dpi=200)
    plt.close()

def estocasticidade_sigma2_vs_tempo_3(n, stats, sub_dir, ts):
    plt.figure(figsize=(7, 5))
    plt.plot(ts, stats["sigma2"], "b-", lw=1.5, label=r"$\sigma^2(t)$")
    plt.xlabel("Tempo")
    plt.ylabel(r"$\sigma^2(t)$")
    plt.title(f"Evolução temporal da dispersão quadrática (n={n})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "sigma2_vs_tempo.png"), dpi=200)
    plt.close()

def estocasticidade_distribuicao_inicial_final_4(dimensao_quadrado, mean_final, n, r0, rs, rs_final, sub_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(rs[0, :, 0], rs[0, :, 1], s=12, alpha=0.5, label=rf"Inicial ($\delta r_n = {dimensao_quadrado:.2e}$)")
    plt.scatter(rs_final[:, 0], rs_final[:, 1], s=12, alpha=0.6, label="Final")
    plt.scatter(r0[0], r0[1], c="red", marker="x", s=100, linewidths=2, label=r"$r_0$")
    plt.scatter(mean_final[0], mean_final[1], c="black", marker="+", s=100, linewidths=2, label=r"$\bar{r}_{final}$")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Distribuição inicial e final (n={n})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sub_dir, "distribuicao_inicial_final.png"), dpi=200)
    plt.close()

def estocasticidade_sigma2_vs_nivel_5(niveis, output_dir, sigma2_finais):
    plt.figure(figsize=(7, 5))
    plt.plot(niveis, sigma2_finais, "o-", color="darkblue", lw=1.8, markersize=6)
    plt.xlabel(r"Nível $n$")
    plt.ylabel(r"$\sigma^2(t_{\max})$ final")
    plt.title(r"Dispersão quadrática final $\sigma^2(t_{\max})$ vs Nível $n$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sigma2_vs_nivel.png"), dpi=200)
    plt.close()

def estocasticidade_sigma2_vs_delta_r_6(deltas, output_dir, sigma2_finais):
    plt.figure(figsize=(7, 5))
    plt.plot(deltas, sigma2_finais, "s-", color="crimson", lw=1.8, markersize=6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Tamanho inicial $\delta r_n = \|r_0\| R(n)$")
    plt.ylabel(r"$\sigma^2(t_{\max})$ final")
    plt.title(r"$\sigma^2(t_{\max})$ vs $\delta r_n$ inicial (escala log-log)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sigma2_vs_delta_r.png"), dpi=200)
    plt.close()

def estocasticidade_dispersao_media_vs_nivel_7(dispersoes_finais, niveis, output_dir):
    plt.figure(figsize=(7, 5))
    plt.plot(niveis, dispersoes_finais, "^-", color="teal", lw=1.8, markersize=6)
    plt.xlabel(r"Nível $n$")
    plt.ylabel("Dispersão média final")
    plt.title(r"Dispersão média final vs Nível $n$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dispersao_media_vs_nivel.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_trajetorias_1(N_particulas, alpha_val, c_val, pasta_ponto, r0, rs_ref):
    plt.figure(figsize=(6, 6))
    cores = plt.cm.viridis(np.linspace(0, 1, N_particulas))
    for p_i in range(N_particulas):
        plt.plot(rs_ref[:, p_i, 0], rs_ref[:, p_i, 1], lw=0.8, alpha=0.35, color=cores[p_i])
        plt.scatter(rs_ref[0, p_i, 0], rs_ref[0, p_i, 1], facecolors="none", edgecolors=cores[p_i], s=25, linewidths=1.0)
        plt.scatter(rs_ref[-1, p_i, 0], rs_ref[-1, p_i, 1], color=cores[p_i], s=25, zorder=3)
    plt.scatter(r0[0], r0[1], c="red", s=100, marker="x", linewidths=2, label=r"$r_0$")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajetórias (alpha={alpha_val:.3f}, c={c_val:.2f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_ponto, "trajetorias.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_distribuicao_inicial_final_2(alpha_val, c_val, delta_r, mean_final_ref, pasta_ponto, r0, rs_ref):
    plt.figure(figsize=(6, 6))
    plt.scatter(rs_ref[0, :, 0], rs_ref[0, :, 1], s=12, alpha=0.5, label=rf"Inicial ($\delta r = {delta_r:.2e}$)")
    plt.scatter(rs_ref[-1, :, 0], rs_ref[-1, :, 1], s=12, alpha=0.6, label="Final")
    plt.scatter(r0[0], r0[1], c="red", marker="x", s=100, linewidths=2, label=r"$r_0$")
    plt.scatter(mean_final_ref[0], mean_final_ref[1], c="black", marker="+", s=100, linewidths=2, label=r"$\bar{r}_{final}$")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Distribuição inicial vs final (alpha={alpha_val:.3f}, c={c_val:.2f})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_ponto, "distribuicao_inicial_final.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_mapa_calor_sigma2_final_3(N_fields, lista_alpha, lista_c, matriz_sigma2_final, output_dir):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        matriz_sigma2_final,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[min(lista_alpha), max(lista_alpha), min(lista_c), max(lista_c)]
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\sigma^2(t_{\max})$ final")
    plt.xlabel(r"$\alpha_{\mathrm{cut}}$ (raio de corte $R_{\mathrm{cutoff}} = \alpha R_n$)")
    plt.ylabel(r"$c$ (inclinação da função bump)")
    plt.title(rf"Dispersão final $\sigma^2(t_{{\max}})$ no plano $(\alpha, c)$ (N={N_fields})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mapa_calor_sigma2_final.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_mapa_calor_sigma2_normalizado_4(N_fields, lista_alpha, lista_c, matriz_norm_sigma2, output_dir):
    plt.figure(figsize=(8, 6))
    log_norm = np.log10(np.maximum(matriz_norm_sigma2, 1e-12))
    im = plt.imshow(
        log_norm,
        origin="lower",
        aspect="auto",
        cmap="plasma",
        extent=[min(lista_alpha), max(lista_alpha), min(lista_c), max(lista_c)]
    )
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\log_{10}(\sigma^2 / \delta r^2)$")
    plt.xlabel(r"$\alpha_{\mathrm{cut}}$")
    plt.ylabel(r"$c$")
    plt.title(rf"Razão de dispersão $\log_{10}(\sigma^2 / \delta r^2)$ (N={N_fields})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mapa_calor_sigma2_normalizado.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_evolucao_temporal_sigma2_selecionados_5(N_fields, historico_temporal, lista_alpha, lista_c, output_dir):
    plt.figure(figsize=(9, 6))
    for c_sel in [lista_c[0], lista_c[len(lista_c)//2], lista_c[-1]]:
        for a_sel in [lista_alpha[0], lista_alpha[len(lista_alpha)//2], lista_alpha[-1]]:
            chave = f"alpha_{a_sel:.4f}_c_{c_sel:.4f}"
            if chave in historico_temporal:
                d = historico_temporal[chave]
                plt.plot(d["t"], d["sigma2"], lw=1.5, label=rf"$\alpha$={a_sel:.2f}, $c$={c_sel:.2f}")

    plt.xlabel("Tempo")
    plt.ylabel(r"$\sigma^2(t)$")
    plt.title(rf"Evolução temporal de $\sigma^2(t)$ para parâmetros selecionados (N={N_fields})")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evolucao_temporal_sigma2_selecionados.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_corte_sigma2_vs_c_6(lista_alpha, lista_c, matriz_sigma2_final, output_dir):
    plt.figure(figsize=(8, 5))
    for idx_a, alpha_val in enumerate(lista_alpha):
        plt.plot(lista_c, matriz_sigma2_final[:, idx_a], "o-", lw=1.5, label=rf"$\alpha$={alpha_val:.2f}")
    plt.xlabel(r"$c$")
    plt.ylabel(r"$\sigma^2(t_{\max})$ final")
    plt.title(r"Cortes unidimensionais: $\sigma^2$ final vs $c$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corte_sigma2_vs_c.png"), dpi=200)
    plt.close()

def estocasticidade_parametros_corte_sigma2_vs_alpha_7(lista_alpha, lista_c, matriz_sigma2_final, output_dir):
    plt.figure(figsize=(8, 5))
    for idx_c, c_val in enumerate(lista_c):
        plt.plot(lista_alpha, matriz_sigma2_final[idx_c, :], "s-", lw=1.5, label=rf"$c$={c_val:.2f}")
    plt.xlabel(r"$\alpha_{\mathrm{cut}}$")
    plt.ylabel(r"$\sigma^2(t_{\max})$ final")
    plt.title(r"Cortes unidimensionais: $\sigma^2$ final vs $\alpha_{\mathrm{cut}}$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corte_sigma2_vs_alpha.png"), dpi=200)
    plt.close()

def robustez_estatistica_convergencia_robustez_1(dados_robustez, n_pts, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Média vs N_particulas
    axes[0, 0].plot(n_pts, dados_robustez['mean_x'], 'o-b', label='<x>')
    axes[0, 0].plot(n_pts, dados_robustez['mean_y'], 's-r', label='<y>')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Número de partículas N_particulas')
    axes[0, 0].set_ylabel('Média final')
    axes[0, 0].set_title('Convergência da Média vs N_particulas')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Variância vs N_particulas
    axes[0, 1].plot(n_pts, dados_robustez['var_x'], 'o-b', label='Var[x]')
    axes[0, 1].plot(n_pts, dados_robustez['var_y'], 's-r', label='Var[y]')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Número de partículas N_particulas')
    axes[0, 1].set_ylabel('Variância final')
    axes[0, 1].set_title('Convergência da Variância vs N_particulas')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. sigma² vs N_particulas
    axes[1, 0].plot(n_pts, dados_robustez['sigma2'], '^-g', label=r'$\sigma^2$ total')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Número de partículas N_particulas')
    axes[1, 0].set_ylabel(r'$\sigma^2$ final')
    axes[1, 0].set_title(r'Convergência de $\sigma^2$ vs N_particulas')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. MSD vs N_particulas
    axes[1, 1].plot(n_pts, dados_robustez['msd'], 'd-m', label='MSD')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Número de partículas N_particulas')
    axes[1, 1].set_ylabel('MSD final')
    axes[1, 1].set_title('Convergência do MSD vs N_particulas')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergencia_robustez.png'), dpi=200)
    plt.close()

def transicao_temporal_sigma2_vs_parametro_1(dados_transicao, lista_N_fields, lista_param_valores, output_dir, param_nome):
    plt.figure(figsize=(7, 5))
    for n in lista_N_fields:
        sigma2_vals = dados_transicao['resultados_por_N'][n]['sigma2_final']
        plt.plot(lista_param_valores, sigma2_vals, 'o-', label=f'N_fields = {n}')
    plt.xlabel(f'Parâmetro de controle ({param_nome})')
    plt.ylabel(r'$\sigma^2$ total final')
    plt.title(fr'Dispersão $\sigma^2$ vs Parâmetro {param_nome}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma2_vs_parametro.png'), dpi=200)
    plt.close()

def transicao_temporal_msd_vs_parametro_2(dados_transicao, lista_N_fields, lista_param_valores, output_dir, param_nome):
    plt.figure(figsize=(7, 5))
    for n in lista_N_fields:
        msd_vals = dados_transicao['resultados_por_N'][n]['msd_final']
        plt.plot(lista_param_valores, msd_vals, 's-', label=f'N_fields = {n}')
    plt.xlabel(f'Parâmetro de controle ({param_nome})')
    plt.ylabel('MSD final')
    plt.title(f'MSD final vs Parâmetro {param_nome}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'msd_vs_parametro.png'), dpi=200)
    plt.close()

def transicao_temporal_distancia_vs_parametro_3(dados_transicao, lista_N_fields, lista_param_valores, output_dir, param_nome):
    plt.figure(figsize=(7, 5))
    for n in lista_N_fields:
        dist_vals = dados_transicao['resultados_por_N'][n]['dist_media_final']
        plt.plot(lista_param_valores, dist_vals, '^-', label=f'N_fields = {n}')
    plt.xlabel(f'Parâmetro de controle ({param_nome})')
    plt.ylabel('Distância média ao centroide')
    plt.title(f'Dispersão espacial vs Parâmetro {param_nome}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distancia_vs_parametro.png'), dpi=200)
    plt.close()

def estabilidade_trajetorias_1(N_particulas, estocastico, n, pasta_saida, r0_array, rs_cpu):
    plt.figure(figsize=(6, 6))

    cores = plt.cm.viridis(
        np.linspace(0, 1, N_particulas)
    )

    for i in range(N_particulas):

        # Trajetória
        plt.plot(
            rs_cpu[:, i, 0],
            rs_cpu[:, i, 1],
            lw=1,
            alpha=0.4,
            color=cores[i]
        )

        # Ponto inicial
        plt.scatter(
            rs_cpu[0, i, 0],
            rs_cpu[0, i, 1],
            facecolors="none",
            edgecolors=cores[i],
            s=30,
            linewidths=1.5
        )

        # Ponto final
        plt.scatter(
            rs_cpu[-1, i, 0],
            rs_cpu[-1, i, 1],
            color=cores[i],
            s=30,
            zorder=3
        )

    # Ponto central
    plt.scatter(
        r0_array[0],
        r0_array[1],
        c="red",
        s=100,
        marker="x",
        linewidths=2,
        label="ponto r0"
    )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    if estocastico:
        plt.title(
            f"Dinâmica estocástica — nível n = {n}"
        )
    else:
        plt.title(
            f"Estabilidade em torno de r0 — nível n = {n}"
        )

    plt.legend()
    plt.grid(alpha=0.3)

    nome_arquivo = os.path.join(
        pasta_saida,
        f"estabilidade_nivel_{n}"
        f"_em_x{r0_array[0]:.3f}"
        f"_y{r0_array[1]:.2f}.png"
    )

    plt.savefig(nome_arquivo, dpi=200)
    plt.close()

def estabilidade_trajetorias_temporal_2(N_particulas, estocastico, n, pasta_saida, r0_array, rs_cpu):
    plt.figure(figsize=(6, 6))

    cores = plt.cm.viridis(
        np.linspace(0, 1, N_particulas)
    )

    for i in range(N_particulas):

        plt.plot(
            rs_cpu[:, i, 0],
            rs_cpu[:, i, 1],
            lw=1,
            alpha=0.4,
            color=cores[i]
        )

        plt.scatter(
            rs_cpu[0, i, 0],
            rs_cpu[0, i, 1],
            facecolors="none",
            edgecolors=cores[i],
            s=30,
            linewidths=1.5
        )

        plt.scatter(
            rs_cpu[-1, i, 0],
            rs_cpu[-1, i, 1],
            color=cores[i],
            s=30,
            zorder=3
        )

    plt.scatter(
        r0_array[0],
        r0_array[1],
        c="red",
        s=100,
        marker="x",
        linewidths=2,
        label="ponto r0"
    )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")

    if estocastico:
        plt.title(
            f"Dinâmica estocástica — nível n = {n}"
        )
    else:
        plt.title(
            f"Estabilidade em torno de r0 — nível n = {n}"
        )

    plt.legend()
    plt.grid(alpha=0.3)

    nome_arquivo = os.path.join(
        pasta_saida,
        f"estabilidade_nivel_{n}"
        f"_em_x{r0_array[0]:.3f}"
        f"_y{r0_array[1]:.2f}em.png"
    )

    plt.savefig(
        nome_arquivo,
        dpi=200
    )

    plt.close()

def estabilidade_hist2d_3(estocastico, n, pasta_saida_info, r0_array, rs_final_cpu):
    plt.figure(figsize=(6, 6))

    x = rs_final_cpu[:, 0]
    y = rs_final_cpu[:, 1]

    plt.hist2d(
        x,
        y,
        bins=50,
        density=True
    )

    plt.colorbar(
        label="densidade"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    if estocastico:
        plt.title(
            f"Distribuição final — dinâmica estocástica — n={n}"
        )
    else:
        plt.title(
            f"Histograma 2D — nível {n}"
        )

    plt.axis("equal")

    nome_hist = os.path.join(
        pasta_saida_info,
        f"hist2d_nivel_{n}"
        f"_x{r0_array[0]:.3f}"
        f"_y{r0_array[1]:.2f}.png"
    )

    plt.savefig(
        nome_hist,
        dpi=200
    )

    plt.close()

def legado_distribuicao_delta_r__1(d_idx, delta_r, n, r0, rs_final, sub_dir):
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

def legado_dispersao_vs_delta_r_2(deltas, n, res_n, sub_dir):
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

def legado_variancia_vs_delta_r_3(deltas, n, res_n, sub_dir):
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

def legado_sigma2_vs_delta_r_4(deltas, n, res_n, sub_dir):
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

def legado_comparacao_variancia_x_5(output_dir, stats_A, stats_B, stats_C, ts_A, ts_B, ts_C):
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['var_x'], label='A (Det + pert. inic.) - Var[x]', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['var_x'], label='B (Estoc. sem pert.) - Var[x]', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['var_x'], label='C (Estoc. + pert.) - Var[x]', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel('Variância em x')
    plt.title('Comparação de Variância Var[x](t)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_variancia_x.png'), dpi=200)
    plt.close()

def legado_comparacao_sigma2_6(output_dir, stats_A, stats_B, stats_C, ts_A, ts_B, ts_C):
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['sigma2'], label='A (Det + pert. inic.)', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['sigma2'], label='B (Estoc. sem pert.)', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['sigma2'], label='C (Estoc. + pert.)', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel(r'$\sigma^2(t)$ total')
    plt.title(r'Comparação de $\sigma^2(t)$ Total')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_sigma2.png'), dpi=200)
    plt.close()

def legado_comparacao_distancia_trajetorias_7(output_dir, stats_A, stats_B, stats_C, ts_A, ts_B, ts_C):
    plt.figure(figsize=(8, 5))
    plt.plot(ts_A, stats_A['dist_media_trajetorias'], label='A (Det + pert. inic.)', color='blue', linestyle='--')
    plt.plot(ts_B, stats_B['dist_media_trajetorias'], label='B (Estoc. sem pert.)', color='green', linestyle='-.')
    plt.plot(ts_C, stats_C['dist_media_trajetorias'], label='C (Estoc. + pert.)', color='red', linestyle='-')
    plt.xlabel('Tempo t')
    plt.ylabel('Distância média ao centroide')
    plt.title('Distância média entre trajetórias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_distancia_trajetorias.png'), dpi=200)
    plt.close()

def legado_comparacao_distribuicoes_finais_8(output_dir, rs_A, rs_B, rs_C):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    axes[0].scatter(rs_A[-1, :, 0], rs_A[-1, :, 1], alpha=0.5, s=12, color='blue')
    axes[0].set_title('A: Det. + perturbação')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(rs_B[-1, :, 0], rs_B[-1, :, 1], alpha=0.5, s=12, color='green')
    axes[1].set_title('B: Estocástico puro')
    axes[1].set_xlabel('x')
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(rs_C[-1, :, 0], rs_C[-1, :, 1], alpha=0.5, s=12, color='red')
    axes[2].set_title('C: Estoc. + perturbação')
    axes[2].set_xlabel('x')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacao_distribuicoes_finais.png'), dpi=200)
    plt.close()

def gerar_histogramas_completos(
    rs_final: np.ndarray,
    title_prefix: str,
    output_dir: str,
    filename_prefix: str
):
    """
    Gera e salva histogramas 1D de x, 1D de y e 2D de (x, y).
    """
    rs_cpu = np.asarray(rs_final)
    os.makedirs(output_dir, exist_ok=True)
    x = rs_cpu[:, 0]
    y = rs_cpu[:, 1]

    # Histogramas 1D
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(x, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Densidade')
    axes[0].set_title(f'{title_prefix} - Histograma 1D (x)')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(y, bins=30, density=True, color='salmon', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('Densidade')
    axes[1].set_title(f'{title_prefix} - Histograma 1D (y)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_hist1d.png'), dpi=200)
    plt.close()

    # Histograma 2D
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=40, density=True, cmap='viridis')
    plt.colorbar(label='Densidade')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{title_prefix} - Histograma 2D (x, y)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_hist2d.png'), dpi=200)
    plt.close()

# Família, função e categorias disponíveis.
REGISTRO = [
    ('comparacao_simuladores', comparacao_simuladores_trajetorias_comparadas_1, ('trajetorias',)),
    ('comparacao_simuladores', comparacao_simuladores_distribuicao_final_comparada_2, ('distribuicoes',)),
    ('comparacao_simuladores', comparacao_simuladores_medias_comparadas_3, ('estatisticas',)),
    ('comparacao_simuladores', comparacao_simuladores_variancias_comparadas_4, ('estatisticas',)),
    ('comparacao_simuladores', comparacao_simuladores_sigma2_comparado_5, ('dispersao',)),
    ('comparacao_simuladores', comparacao_simuladores_msd_comparado_6, ('difusao',)),
    ('convergencia_N', convergencia_N_dt_N_vs_N_1, ('convergencia',)),
    ('convergencia_N', convergencia_N_num_passos_vs_N_2, ('convergencia',)),
    ('convergencia_N', convergencia_N_sigma2_vs_N_3, ('dispersao',)),
    ('convergencia_N', convergencia_N_variancia_x_y_vs_N_4, ('estatisticas',)),
    ('convergencia_N', convergencia_N_custo_computacional_vs_N_5, ('convergencia',)),
    ('convergencia_dt', convergencia_dt_distribuicao_2d_dt__1, ('distribuicoes',)),
    ('convergencia_dt', convergencia_dt_estatisticas_vs_dt_2, ('estatisticas',)),
    ('estatisticas_temporais', estatisticas_temporais_figura_1, ('estatisticas',)),
    ('estocasticidade', estocasticidade_trajetorias_nivel__1, ('trajetorias',)),
    ('estocasticidade', estocasticidade_hist2d_2, ('distribuicoes',)),
    ('estocasticidade', estocasticidade_sigma2_vs_tempo_3, ('dispersao',)),
    ('estocasticidade', estocasticidade_distribuicao_inicial_final_4, ('distribuicoes',)),
    ('estocasticidade', estocasticidade_sigma2_vs_nivel_5, ('dispersao',)),
    ('estocasticidade', estocasticidade_sigma2_vs_delta_r_6, ('dispersao',)),
    ('estocasticidade', estocasticidade_dispersao_media_vs_nivel_7, ('dispersao',)),
    ('estocasticidade_parametros', estocasticidade_parametros_trajetorias_1, ('trajetorias',)),
    ('estocasticidade_parametros', estocasticidade_parametros_distribuicao_inicial_final_2, ('distribuicoes',)),
    ('estocasticidade_parametros', estocasticidade_parametros_mapa_calor_sigma2_final_3, ('mapas', 'dispersao')),
    ('estocasticidade_parametros', estocasticidade_parametros_mapa_calor_sigma2_normalizado_4, ('mapas', 'dispersao')),
    ('estocasticidade_parametros', estocasticidade_parametros_evolucao_temporal_sigma2_selecionados_5, ('dispersao',)),
    ('estocasticidade_parametros', estocasticidade_parametros_corte_sigma2_vs_c_6, ('dispersao',)),
    ('estocasticidade_parametros', estocasticidade_parametros_corte_sigma2_vs_alpha_7, ('dispersao',)),
    ('robustez_estatistica', robustez_estatistica_convergencia_robustez_1, ('estatisticas',)),
    ('transicao_temporal', transicao_temporal_sigma2_vs_parametro_1, ('dispersao',)),
    ('transicao_temporal', transicao_temporal_msd_vs_parametro_2, ('difusao',)),
    ('transicao_temporal', transicao_temporal_distancia_vs_parametro_3, ('estatisticas',)),
    ('estabilidade', estabilidade_trajetorias_1, ('trajetorias',)),
    ('estabilidade', estabilidade_trajetorias_temporal_2, ('trajetorias',)),
    ('estabilidade', estabilidade_hist2d_3, ('distribuicoes',)),
    ('legado', legado_distribuicao_delta_r__1, ('distribuicoes',)),
    ('legado', legado_dispersao_vs_delta_r_2, ('dispersao',)),
    ('legado', legado_variancia_vs_delta_r_3, ('estatisticas',)),
    ('legado', legado_sigma2_vs_delta_r_4, ('dispersao',)),
    ('legado', legado_comparacao_variancia_x_5, ('estatisticas',)),
    ('legado', legado_comparacao_sigma2_6, ('dispersao',)),
    ('legado', legado_comparacao_distancia_trajetorias_7, ('estatisticas',)),
    ('legado', legado_comparacao_distribuicoes_finais_8, ('distribuicoes',)),
]
