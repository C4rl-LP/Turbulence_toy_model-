"""Reconstrói os gráficos estatísticos de uma varredura de parâmetros salva em NPZ."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plotar(arquivo_resumo, pasta_saida=None):
    arquivo_resumo = Path(arquivo_resumo)
    destino = Path(pasta_saida) if pasta_saida else arquivo_resumo.parent / "graficos"
    destino.mkdir(parents=True, exist_ok=True)

    with np.load(arquivo_resumo) as dados:
        nivel = int(dados["N_fields"])
        alphas = dados["lista_alpha"]
        valores_c = dados["lista_c"]
        sigma2 = dados["matriz_sigma2_final"]
        norm_sigma2 = dados["matriz_norm_sigma2"]

    def salvar_mapa(matriz, nome, titulo, legenda, cmap):
        fig, ax = plt.subplots(figsize=(8, 6))
        imagem = ax.pcolormesh(alphas, valores_c, matriz, shading="nearest", cmap=cmap)
        fig.colorbar(imagem, ax=ax, label=legenda)
        ax.set(xlabel=r"$\alpha_{\mathrm{cut}}$", ylabel=r"$c$", title=titulo)
        fig.tight_layout()
        fig.savefig(destino / nome, dpi=200)
        plt.close(fig)

    salvar_mapa(
        sigma2,
        "mapa_calor_sigma2_final.png",
        rf"Dispersão final $\sigma^2(t_{{\max}})$ (N={nivel})",
        r"$\sigma^2(t_{\max})$ final",
        "viridis",
    )
    salvar_mapa(
        np.log10(np.maximum(norm_sigma2, 1e-12)),
        "mapa_calor_sigma2_normalizado.png",
        rf"Dispersão normalizada $\log_{{10}}(\sigma^2/\delta r^2)$ (N={nivel})",
        r"$\log_{10}(\sigma^2 / \delta r^2)$",
        "plasma",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx_alpha, alpha in enumerate(alphas):
        ax.plot(valores_c, sigma2[:, idx_alpha], "o-", label=rf"$\alpha$={alpha:.2f}")
    ax.set(xlabel=r"$c$", ylabel=r"$\sigma^2(t_{\max})$", title=r"Dispersão final vs. $c$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "corte_sigma2_vs_c.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx_c, c_val in enumerate(valores_c):
        ax.plot(alphas, sigma2[idx_c, :], "s-", label=rf"$c$={c_val:.2f}")
    ax.set(xlabel=r"$\alpha_{\mathrm{cut}}$", ylabel=r"$\sigma^2(t_{\max})$", title=r"Dispersão final vs. $\alpha_{\mathrm{cut}}$")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "corte_sigma2_vs_alpha.png", dpi=200)
    plt.close(fig)

    selecionados_c = [valores_c[i] for i in sorted({0, len(valores_c) // 2, len(valores_c) - 1})]
    selecionados_alpha = [alphas[i] for i in sorted({0, len(alphas) // 2, len(alphas) - 1})]
    fig, ax = plt.subplots(figsize=(9, 6))
    raiz = arquivo_resumo.parent
    for c_val in selecionados_c:
        for alpha in selecionados_alpha:
            ponto = raiz / f"alpha_{alpha:.3f}_c_{c_val:.3f}" / "dados_ponto.npz"
            if not ponto.exists():
                continue
            with np.load(ponto) as dados:
                ax.plot(
                    dados["t"], dados["sigma2_t"], lw=1.5,
                    label=rf"$\alpha$={alpha:.2f}, $c$={c_val:.2f}",
                )
    ax.set(xlabel="Tempo", ylabel=r"$\sigma^2(t)$", title=rf"Evolução temporal de $\sigma^2(t)$ (N={nivel})")
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(destino / "evolucao_temporal_sigma2_selecionados.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resumo", help="arquivo resumo_varredura.npz")
    parser.add_argument("--saida", help="pasta de saída para os gráficos")
    args = parser.parse_args()
    plotar(args.resumo, args.saida)


if __name__ == "__main__":
    main()
