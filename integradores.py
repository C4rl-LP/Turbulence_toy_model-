import numpy as np 
import os
from matplotlib import pyplot as plt
import funcoes_para_centros as fc
from funcoes_para_centros import R_1, O_1, lamb, x_1, T_1, alpha_padrao, c_padrao
import time 

def solve_RK4(func, r0, t0, dt, t_max):
    """
    Resolve o sistema dr/dt = func(t, r) usando RK4.

    Parâmetros:
    - func : função do campo vetorial
    - r0   : condição inicial (array)
    - t0   : tempo inicial
    - dt   : passo de tempo
    - t_max: tempo final
    """
    Nt = int((t_max - t0)/dt) + 1
    r = np.zeros((Nt, *r0.shape))
    t = np.zeros(Nt)

    r[0] = r0
    t[0] = t0

    for i in range(Nt - 1):
        ti = t[i]
        ri = r[i]

        k1 = func(ti, ri)
        k2 = func(ti + dt/2, ri + dt/2*k1)
        k3 = func(ti + dt/2, ri + dt/2*k2)
        k4 = func(ti + dt,   ri + dt*k3)

        r[i+1] = ri + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = ti + dt

    return t, r

def solve_euler_maruyama(func, r0, t_max, dt):
    """
    Resolve

        dr = V(r,t) dt + dW

    usando o método de Euler-Maruyama.

    Parâmetros
    ----------
    func : função
        Campo de velocidade V(t, r).

    r0 : array_like, shape (2,)
        Condição inicial [x0, y0].

    t_max : float
        Tempo total da simulação.

    dt : float
        Passo temporal.

    Retorna
    -------
    t : ndarray, shape (Nt,)
        Tempos.

    r : ndarray, shape (Nt, 2)
        Trajetória.
    """

    N_steps = int(np.ceil(t_max / dt))

    # Ajusta dt para que o último ponto seja exatamente t_max
    dt_real = t_max / N_steps

    t = np.linspace(0.0, t_max, N_steps + 1)    

    r = np.empty((N_steps + 1, 2))
    r[0] = r0

    for k in range(N_steps):
        tk = t[k]

        V = func(tk, r[k])

        dW = np.sqrt(dt_real) * np.random.normal(
            size=2
        )

        r[k + 1] = (
            r[k]
            + V * dt_real
            + dW
        )

    return t, r

def simular_particulas(N, N_fields, dimensao_quadrado, r0, dt=0.01, t_max=1, func = None, temporal = False):
    """
    Simula N partículas em um pequeno quadrado
    ao redor da posição inicial r0.
    """

    def funcao(t, r):
        x = r[:, 0]
        y = r[:, 1]
        if temporal:
            Vx, Vy = func(x, y, t, N_fields)
        else:
            Vx, Vy = func(x, y, 0, N_fields)
        return np.column_stack((Vx, Vy))

    L = dimensao_quadrado

    # Distribuição inicial uniforme em um círculo de raio L/2
    R0 = L / 2

    theta = np.random.uniform(0, 2*np.pi, size=N-1)
    u = np.random.uniform(0, 1, size=N-1)

    r = R0 * np.sqrt(u)

    dx = r * np.cos(theta)
    dy = r * np.sin(theta)

    particle_t0 = r0 + np.column_stack((dx, dy))

    # adiciona a partícula central exatamente em r0
    particle_t0 = np.vstack((r0,particle_t0))

    t, r = solve_RK4(funcao, particle_t0, 0.0, dt, t_max)
    return t, r



def simular_particulas_ruidosa(
    N,
    N_fields,
    r0,
    dt_1=0.01,
    t_max=1,
    lamb=2**(2/3),
    func=None,
    temporal=False
):
    """
    Simula N_particulas realizações da SDE

        dr = V_N(r,t) dt + dW

    todas próximas da mesma condição inicial r0.

    Parâmetros
    ----------
    N : int
        Número de realizações/partículas.

    N_fields : int
        Número de níveis do campo V_N.

    r0 : array_like, shape (2,)
        Condição inicial central.

    dt_1 : float
        Passo temporal correspondente ao nível N_fields = 1.

    t_max : float
        Tempo total da simulação.

    lamb : float
        Fator de escala temporal.

    func : função
        Campo de velocidade.

        Deve retornar:
            Vx, Vy = func(x, y, t, N_fields)

    temporal : bool
        Se True, utiliza o tempo t no campo.
        Se False, utiliza t = 0.

    Retorna
    -------
    t : ndarray, shape (Nt,)
        Tempos.

    rs : ndarray, shape (Nt, N, 2)
        Trajetórias das N realizações.
    """

    r0 = np.asarray(r0, dtype=float)

    # ---------------------------------------------------------
    # Passo temporal correspondente ao nível N_fields
    # ---------------------------------------------------------

    dt_N = dt_1 / lamb**(N_fields - 1)

    N_steps = int(np.ceil(t_max / dt_N))

    # Ajuste para terminar exatamente em t_max
    dt_N = t_max / N_steps

    t = np.linspace(0.0, t_max, N_steps + 1)

    # ---------------------------------------------------------
    # Condições iniciais
    # ---------------------------------------------------------

    particle_t0 = np.tile(r0, (N, 1))


    # ---------------------------------------------------------
    # Armazenamento das trajetórias
    # ---------------------------------------------------------

    rs = np.empty((N_steps + 1, N, 2))

    rs[0] = particle_t0

    # ---------------------------------------------------------
    # Euler-Maruyama
    # ---------------------------------------------------------

    for k in range(N_steps):

        tk = t[k]

        x = rs[k, :, 0]
        y = rs[k, :, 1]

        if temporal:
            Vx, Vy = func(
                x,
                y,
                tk,
                N_fields
            )
        else:
            Vx, Vy = func(
                x,
                y,
                0,
                N_fields
            )

        V = np.column_stack((Vx, Vy))

        # Wiener independente para cada realização
        dW = np.sqrt(dt_N) * np.random.normal(
            size=(N, 2)
        )

        # Euler-Maruyama
        rs[k + 1] = (
            rs[k]
            + V * dt_N
            + dW
        )

    return t, rs




def testar_estabilidade_estatico(
    nivel_max,
    N_particulas,
    r0,
    pasta_saida="estabilidade_2",
    nivel_min=1,
    funcao=None,
    simulador=None,
    sigma0=0.0
):
    """
    Testa a estabilidade do ponto r0 para diferentes níveis do campo.

    Por padrão, utiliza a função determinística `simular_particulas`.

    Para utilizar a dinâmica estocástica, passe:

        simulador=simular_particulas_ruidosa

    Parâmetros adicionais
    ---------------------
    simulador : função, opcional
        Função responsável pela simulação.

        Se None:
            utiliza simular_particulas.

        Pode ser:
            simular_particulas
            simular_particulas_ruidosa

    sigma0 : float
        Desvio padrão da perturbação inicial utilizado pela
        simular_particulas_ruidosa.

        Não altera o comportamento determinístico.
    """

    os.makedirs(pasta_saida, exist_ok=True)

    # Mantém o comportamento antigo caso nenhum simulador seja informado
    if simulador is None:
        simulador = simular_particulas

    for n in range(nivel_min, nivel_max + 1):

        t0 = time.perf_counter()

        print(f"Testando estabilidade para n = {n}")

        # ---------------------------------------------------------
        # Simulação determinística
        # ---------------------------------------------------------

        if simulador is simular_particulas:

            L = fc.R(n) / 8
            dt = 0.03 * fc.T(n)
            t_max = lamb / 2

            ts, rs = simulador(
                N=N_particulas,
                N_fields=n,
                dimensao_quadrado=L,
                r0=r0,
                dt=dt,
                t_max=t_max,
                func=funcao
            )

        # ---------------------------------------------------------
        # Simulação estocástica
        # ---------------------------------------------------------

        else:

            # dt_1 corresponde ao nível n = 1
            dt_1 = 0.03 * fc.T(n)

            t_max = lamb / 2

            ts, rs = simulador(
                N=N_particulas,
                N_fields=n,
                r0=np.asarray(r0).reshape(-1),
                dt_1=dt_1,
                t_max=t_max,
                lamb=lamb,
                func=funcao,
                sigma0=sigma0
            )

            # Para manter uma escala de referência para os gráficos
            L = fc.R(n) / 8

        # rs tem shape:
        #
        # (Nt, N_particulas, 2)

        t1 = time.perf_counter()

        plt.figure(figsize=(6, 6))

        cores = plt.cm.viridis(
            np.linspace(0, 1, N_particulas)
        )

        for i in range(N_particulas):

            # Trajetória
            plt.plot(
                rs[:, i, 0],
                rs[:, i, 1],
                lw=1,
                alpha=0.4,
                color=cores[i]
            )

            # Ponto inicial
            plt.scatter(
                rs[0, i, 0],
                rs[0, i, 1],
                facecolors="none",
                edgecolors=cores[i],
                s=30,
                linewidths=1.5
            )

            # Ponto final
            plt.scatter(
                rs[-1, i, 0],
                rs[-1, i, 1],
                color=cores[i],
                s=30,
                zorder=3
            )

        # Ponto central
        r0_array = np.asarray(r0).reshape(-1)

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

        if simulador is simular_particulas_ruidosa:
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

        print(f"Tempo demorado: {t1 - t0}")
        print(f"Imagem salva em: {nome_arquivo}")

def testar_estabilidade_temporal(
    nivel_max,
    N_particulas,
    r0,
    pasta_saida="estabilidade_2",
    pasta_saida_info="estabilidade_podadda_info",
    nivel_min=1,
    funcao=None,
    temporal=True,
    simulador=None,
    sigma0=0.0
):
    """
    Testa a estabilidade do ponto r0 para diferentes níveis do campo.

    Por padrão utiliza `simular_particulas`.

    Para utilizar a dinâmica estocástica:

        simulador=simular_particulas_ruidosa

    A função calcula, para cada nível:

        média
        variância
        matriz de covariância
        sigma² total

    e também salva o histograma 2D da distribuição final.

    A estrutura de `rs` permanece:

        rs.shape = (Nt, N_particulas, 2)
    """

    # -------------------------------------------------------------
    # Compatibilidade com o código antigo
    # -------------------------------------------------------------

    if simulador is None:
        simulador = simular_particulas

    os.makedirs(pasta_saida, exist_ok=True)
    os.makedirs(pasta_saida_info, exist_ok=True)

    nome_do_txt_info = os.path.join(
        pasta_saida_info,
        "Informações.txt"
    )

    with open(nome_do_txt_info, "a") as f:

        f.write("\n")
        f.write("Características\n")
        f.write("\n")

        f.write(
            f"R_1 = {R_1}, "
            f"T_1 = {T_1}, "
            f"lambda = {lamb}\n"
        )

        f.write(
            f"alpha de corte = {alpha_padrao}, "
            f"Fator c = {c_padrao}\n"
        )

        f.write(
            f"r_0 = {r0}, "
            f"Número de partículas: {N_particulas}\n"
        )

        if simulador is simular_particulas_ruidosa:
            f.write(
                "Dinâmica: Euler-Maruyama\n"
            )
            f.write(
                f"sigma0 = {sigma0}\n"
            )
            f.write(
                "dt_n = dt_1 / lambda^(n-1)\n"
            )

        else:
            f.write(
                "Dinâmica: determinística\n"
            )

        f.write("Estatísticas:\n")
        f.write("\n")

    nivel_info = []

    # -------------------------------------------------------------
    # Loop sobre os níveis
    # -------------------------------------------------------------

    for n in range(nivel_min, nivel_max + 1):

        t0 = time.perf_counter()

        print(f"Testando estabilidade para n = {n}")

        # ---------------------------------------------------------
        # Determinístico
        # ---------------------------------------------------------

        if simulador is simular_particulas:

            L = fc.R(n) / 16
            dt = 0.03 * fc.T(n)
            t_max = lamb / 2

            ts, rs = simulador(
                N=N_particulas,
                N_fields=n,
                dimensao_quadrado=L,
                r0=r0,
                dt=dt,
                t_max=t_max,
                func=funcao,
                temporal=temporal
            )

        # ---------------------------------------------------------
        # Estocástico
        # ---------------------------------------------------------

        else:

            L = fc.R(n) / 16

            # Passo correspondente ao nível n = 1
            dt_1 = 0.03 * fc.T(1)

            t_max = lamb / 2

            ts, rs = simulador(
                N=N_particulas,
                N_fields=n,
                r0=np.asarray(r0).reshape(-1),
                dt_1=dt_1,
                t_max=t_max,
                lamb=lamb,
                func=funcao,
                temporal=temporal,
                sigma0=sigma0
            )

        # ---------------------------------------------------------
        # Distribuição final
        # ---------------------------------------------------------

        rs_final = rs[-1, :, :]

        mean = np.mean(
            rs_final,
            axis=0
        )

        var = np.var(
            rs_final,
            axis=0,
            ddof=1
        )

        matriz_covarianca = np.cov(
            rs_final.T,
            ddof=1
        )

        sigma2_total = np.mean(
            np.sum(
                (rs_final - mean)**2,
                axis=1
            )
        )

        dic_info = {
            "Media": mean,
            "Varianca": var,
            "Matriz de covariança": matriz_covarianca,
            "sigma total": sigma2_total,
            "L": L
        }

        nivel_info.append(dic_info)

        # ---------------------------------------------------------
        # Salva informações
        # ---------------------------------------------------------

        with open(nome_do_txt_info, "a") as f:

            f.write(
                f"Nível: {n}, "
                f"Tamanho da núvem inicial = {L}\n"
            )

            f.write(
                f"Média: {mean}, "
                f"Variância: {var}\n"
            )

            cov = matriz_covarianca

            f.write(
                f"Sigma^2 raio = {sigma2_total}\n"
            )

            f.write(
                "Matriz de covariância:\n"
            )

            f.write(
                f"[[{cov[0,0]:.4e}, "
                f"{cov[0,1]:.4e}],\n"
            )

            f.write(
                f" [{cov[1,0]:.4e}, "
                f"{cov[1,1]:.4e}]]\n"
            )

            f.write("\n")
            f.write("=-=" * 10)
            f.write("\n")

        # ---------------------------------------------------------
        # Gráfico das trajetórias
        # ---------------------------------------------------------

        t1 = time.perf_counter()

        plt.figure(figsize=(6, 6))

        cores = plt.cm.viridis(
            np.linspace(0, 1, N_particulas)
        )

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
                linewidths=1.5
            )

            plt.scatter(
                rs[-1, i, 0],
                rs[-1, i, 1],
                color=cores[i],
                s=30,
                zorder=3
            )

        r0_array = np.asarray(r0).reshape(-1)

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

        if simulador is simular_particulas_ruidosa:
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

        print(
            f"Tempo demorado: "
            f"{t1 - t0}"
        )

        print(
            f"Imagem salva em: "
            f"{nome_arquivo}"
        )

        # ---------------------------------------------------------
        # Histograma 2D da distribuição final
        # ---------------------------------------------------------

        plt.figure(figsize=(6, 6))

        x = rs_final[:, 0]
        y = rs_final[:, 1]

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

        if simulador is simular_particulas_ruidosa:
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

        print(
            f"Histograma salvo em: "
            f"{nome_hist}"
        )

    return nivel_info

