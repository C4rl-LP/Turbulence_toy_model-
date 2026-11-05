import time
import campos 
import os




import numpy as np


def testar_poda(
    campo_bruto,
    campo_podado,
    alphas,
    t=0.0,
    n_max=6,
    c=0.4,
    n_pontos=1000,
    limites=(-1.0, 1.0),
    pontos=None,
    atol=1e-10,
    rtol=1e-8,
    seed=42,
    verbose=True,
):
    """
    Compara o campo bruto com o campo podado para vários alphas.

    Os dois campos devem aceitar os argumentos:
        campo(xp, yp, t, n_max=n_max, c=c, alpha=alpha)

    Parâmetros
    ----------
    campo_bruto, campo_podado : callable
        Funções de cálculo do campo.
    alphas : iterable
        Valores de alpha a testar.
    t : float
        Instante de avaliação.
    n_max : int
        Nível máximo da hierarquia.
    c : float
        Parâmetro de suavidade.
    n_pontos : int
        Número de pontos aleatórios, se pontos=None.
    limites : tuple
        Intervalo (mínimo, máximo) para x e y.
    pontos : array (n_pontos, 2), opcional
        Pontos específicos para avaliar.
    atol, rtol : float
        Tolerâncias absoluta e relativa.
    seed : int
        Semente aleatória.

    Retorna
    -------
    resultados : list[dict]
        Métricas e resultado do teste para cada alpha.
    """

    rng = np.random.default_rng(seed)

    if pontos is None:
        pontos = rng.uniform(
            limites[0], limites[1], size=(n_pontos, 2)
        )
    else:
        pontos = np.asarray(pontos, dtype=float)
        if pontos.ndim != 2 or pontos.shape[1] != 2:
            raise ValueError("pontos deve ter formato (N, 2).")

    xp = pontos[:, 0]
    yp = pontos[:, 1]

    resultados = []

    for alpha in alphas:
        # Calcula os dois campos nos mesmos pontos.
        vx_b, vy_b = campo_bruto(
            xp, yp, t,
            n_max=n_max, c=c, alpha=alpha
        )

        vx_p, vy_p = campo_podado(
            xp, yp, t,
            n_max=n_max, c=c, alpha=alpha
        )

        # Converte para arrays, inclusive se a função
        # retornar um escalar inesperadamente.
        vx_b = np.broadcast_to(np.asarray(vx_b), xp.shape)
        vy_b = np.broadcast_to(np.asarray(vy_b), xp.shape)
        vx_p = np.broadcast_to(np.asarray(vx_p), xp.shape)
        vy_p = np.broadcast_to(np.asarray(vy_p), xp.shape)

        bruto = np.column_stack((vx_b, vy_b))
        podado = np.column_stack((vx_p, vy_p))

        # Erro vetorial em cada ponto.
        erro = np.linalg.norm(podado - bruto, axis=1)
        norma_bruto = np.linalg.norm(bruto, axis=1)

        erro_max = np.max(erro)
        erro_medio = np.mean(erro)

        # Erro relativo por ponto; evita divisão por zero.
        denominador = np.maximum(norma_bruto, atol)
        erro_relativo = erro / denominador
        erro_rel_max = np.max(erro_relativo)

        # Critério de aceitação componente a componente.
        tolerancia = atol + rtol * np.abs(bruto)
        passou = np.all(np.abs(podado - bruto) <= tolerancia)

        resultado = {
            "alpha": float(alpha),
            "n_pontos": len(pontos),
            "erro_maximo": float(erro_max),
            "erro_medio": float(erro_medio),
            "erro_relativo_maximo": float(erro_rel_max),
            "passou": bool(passou),
        }
        resultados.append(resultado)

        if verbose:
            status = "PASSOU" if passou else "FALHOU"
            print(
                f"alpha={alpha:.6g} | {status} | "
                f"erro max={erro_max:.3e} | "
                f"erro médio={erro_medio:.3e} | "
                f"erro relativo max={erro_rel_max:.3e}"
            )

    return resultados


def teste_tempo_soluçao():
    # Comparar tempo dos campos
    xp=0.13
    yp=0.41

    nmax=20
    t=0.0

    Nrep=100

    t0=time.perf_counter()



    for _ in range(Nrep):
        campos.campo_total_podado_2(
            xp,yp,t,nmax
        )

    t1=time.perf_counter()

    tempo_podado=t1-t0

    # -------- campo completo ----------
    t0=time.perf_counter()

    for _ in range(Nrep):
        campos.campo_total_podado(
            xp,yp,t,nmax
        )

    t1=time.perf_counter()

    tempo_full=t1-t0


    # -------- campo podado ----------



    print("Completo :",tempo_full)
    print("Podado   :",tempo_podado)

    print(
    "speedup =",
    tempo_full/tempo_podado
    )

    pasta = os.path.dirname(os.path.abspath(__file__))
    arquivo = os.path.join(pasta,'Teste_temporal.txt')

    with open(arquivo, "a") as f:

        f.write("Teste \n")
        f.write(f"ponto: ({xp},{yp})    N = {nmax}   t = {t}     Repetições = {Nrep} \n")
        f.write("=-"*10)
        f.write("\n")
        f.write(f"Completo : {tempo_full}\n" )
        f.write(f"Podado : {tempo_podado} \n")
        f.write(f"speedup = {tempo_full/tempo_podado}\n")
        f.write("=-"*10)
        f.write("\n")

'''
alphas = [
    0.2,
    0.5,
    1.0,
    1.2,
    np.sqrt(2),
    1.5,
    2.0,
    2.5,
]
for _ in range(10):
    resultados = testar_poda(
        campo_bruto=campos.campo_total_correto,
        campo_podado=campos.campo_total_podado_2,
        alphas=alphas,
        t=0.3,
        n_max=6,
        c=0.4,
        n_pontos=1000,
        seed=None,
    )
'''
if __name__ == "__main__":
    sla = teste_tempo_soluçao()
