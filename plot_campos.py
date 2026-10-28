from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import campos 



def plot_campo(t=0.0, N_fields=3, L=5,x_0=  0, y_0= 0, resolucao=200, func=None):

    x = np.linspace(-L+x_0, L+x_0, resolucao)
    y = np.linspace(-L+y_0, L +y_0, resolucao)

    X, Y = np.meshgrid(x, y)

    # achata a malha em lista de pontos e envia para GPU
    xp = cp.asarray(X.ravel())
    yp = cp.asarray(Y.ravel())

    # calcula o campo vetorial na GPU
    Vx, Vy = func(xp, yp, t, N_fields)

    # converte de volta para CPU para o matplotlib streamplot
    Vx = cp.asnumpy(Vx).reshape(X.shape)
    Vy = cp.asnumpy(Vy).reshape(Y.shape)

    V = np.sqrt(Vx**2 + Vy**2)

    plt.figure(figsize=(8,8))

    plt.streamplot(
        X, Y,
        Vx, Vy,
        color=V,
        cmap="viridis",
        density=2.5
    )

    plt.colorbar(label='|V|')
    plt.axis('equal')
    plt.show()
t = 0.0
N = 8
x0 = 0
y0 = 0


#plot_campo(t=t, N_fields=N, L=1.4,x_0 = x0, y_0 =y0, resolucao=200, func=campos.campo_total_podado)
#plot_campo(t=t, N_fields=N, L=1.4,x_0 = x0, y_0 =y0,resolucao=200, func=campos.campo_total_podado)

def salvar_por_nivel(
    Nmax,
    t=0.0,
    L=.25,
    x_0=.3,
    y_0=.3,
    resolucao=200,
    func=None,
    pasta="campos",
    N_min = 1
):

    import os
    os.makedirs(pasta, exist_ok=True)

    for N in range(N_min , Nmax+1):

        x = np.linspace(-L+x_0, L+x_0, resolucao)
        y = np.linspace(-L+y_0, L+y_0, resolucao)

        X, Y = np.meshgrid(x,y)

        xp = cp.asarray(X.ravel())
        yp = cp.asarray(Y.ravel())

        Vx, Vy = func(xp, yp, t, N)

        Vx = cp.asnumpy(Vx).reshape(X.shape)
        Vy = cp.asnumpy(Vy).reshape(Y.shape)

        V = np.sqrt(Vx**2 + Vy**2)

        fig, ax = plt.subplots(figsize=(8,8))

        stream = ax.streamplot(
            X,Y,
            Vx,Vy,
            color=V,
            cmap="viridis",
            density=2.5
        )

        fig.colorbar(stream.lines,label="|V|")

        ax.set_title(f"Nível {N}")
        ax.set_aspect('equal')

        nome = f"{pasta}/campo_nivel_{N:02d}.png"

        plt.tight_layout()
        plt.savefig(nome,dpi=300)
        plt.close(fig)

        print(f"salvo: {nome}")

salvar_por_nivel(
    Nmax=20,
    t=0,
    func=campos.campo_total_podado,
    N_min= 9
)

