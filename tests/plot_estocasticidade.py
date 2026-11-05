"""Gera figuras a partir dos NPZ salvos por test_estocasticidade."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plotar(arquivo, pasta_saida=None):
    arquivo = Path(arquivo)
    destino = Path(pasta_saida) if pasta_saida else arquivo.parent / "graficos"
    destino.mkdir(parents=True, exist_ok=True)
    with np.load(arquivo) as d:
        t, rs = d["t"], d["rs"]
        nivel = int(d["N_fields"])
        sigma2 = d["sigma2"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(rs[:, :, 0], rs[:, :, 1], alpha=0.35, lw=0.7)
        ax.scatter(rs[0, :, 0], rs[0, :, 1], s=10, label="início")
        ax.scatter(rs[-1, :, 0], rs[-1, :, 1], s=10, label="fim")
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=f"Trajetórias, N={nivel}")
        ax.legend(); fig.tight_layout(); fig.savefig(destino / f"trajetorias_N{nivel}.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6))
        densidade, x_edges, y_edges, imagem = ax.hist2d(
            rs[-1, :, 0], rs[-1, :, 1], bins=50, cmap="inferno", density=True
        )
        fig.colorbar(imagem, ax=ax, label="densidade de partículas")
        ax.set(aspect="equal", xlabel="x", ylabel="y", title=f"Distribuição final, N={nivel}")
        fig.tight_layout(); fig.savefig(destino / f"distribuicao_final_N{nivel}.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(t, sigma2); ax.set(xlabel="tempo", ylabel="sigma²", title=f"Dispersão, N={nivel}")
        fig.tight_layout(); fig.savefig(destino / f"sigma2_N{nivel}.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivos", nargs="+", help="arquivos .npz de nível")
    parser.add_argument("--saida")
    args = parser.parse_args()
    for arquivo in args.arquivos:
        plotar(arquivo, args.saida)


if __name__ == "__main__":
    main()
