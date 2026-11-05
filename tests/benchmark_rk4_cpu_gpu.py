"""Benchmarka o ganho de integrar um lote de partículas com RK4 na GPU.

Execute com, por exemplo:
    python -m tests.benchmark_rk4_cpu_gpu --n-particles 100 1000 10000 \
        --levels 3 5 --dt 0.01 --t-max 0.1
"""

import argparse
import statistics
import time

import numpy as np

import backend
import campos
import integradores


def executar(n_particulas, n_niveis, dt, t_max, repeticoes, alpha, c):
    if not backend.is_cuda_available():
        raise RuntimeError(f"CUDA indisponível: {backend.get_cuda_status()[1]}")

    # O mesmo lote inicial é passado sem alterações aos dois backends.
    rng = np.random.default_rng(2026)
    r0 = rng.uniform(-0.01, 0.01, size=(n_particulas, 2))

    def integrar(dispositivo):
        def campo(t, r):
            vx, vy = campos.campo_total(
                r[:, 0], r[:, 1], t, n_niveis,
                alpha=alpha, c=c, backend_name=dispositivo,
            )
            xp = backend.get_backend_module(dispositivo)
            return xp.column_stack((vx, vy))

        return integradores.solve_RK4(
            campo, r0, 0.0, dt, t_max, backend_name=dispositivo
        )[1]

    # Aquecimento include compilação JIT e inicialização de contexto.
    trajetoria_gpu = integrar("cuda")
    backend.synchronize("cuda")
    trajetoria_cpu = integrar("cpu")

    erro_max = float(np.max(np.abs(trajetoria_gpu - trajetoria_cpu)))
    tempos = {}
    for dispositivo in ("cpu", "cuda"):
        amostras = []
        for _ in range(repeticoes):
            inicio = time.perf_counter()
            integrar(dispositivo)
            backend.synchronize(dispositivo)
            amostras.append(time.perf_counter() - inicio)
        tempos[dispositivo] = statistics.median(amostras)

    tamanho_trajetoria = trajetoria_cpu.nbytes
    return {
        "particulas": n_particulas,
        "niveis": n_niveis,
        "passos": len(trajetoria_cpu) - 1,
        "cpu_s": tempos["cpu"],
        "gpu_s": tempos["cuda"],
        "speedup": tempos["cpu"] / tempos["cuda"],
        "erro_max_abs": erro_max,
        "trajetoria_MiB": tamanho_trajetoria / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-particles", nargs="+", type=int,
                        default=[100, 1000, 10000, 100000])
    parser.add_argument("--levels", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=campos.alpha_padrao)
    parser.add_argument("--c", type=float, default=campos.c_padrao)
    args = parser.parse_args()

    if not backend.is_cuda_available():
        parser.error(f"CUDA indisponível: {backend.get_cuda_status()[1]}")

    print("partículas níveis passos CPU(s) GPU(s) speedup erro_max trajetórias(MiB)")
    for n in args.n_particles:
        for levels in args.levels:
            result = executar(n, levels, args.dt, args.t_max,
                              args.repetitions, args.alpha, args.c)
            print(
                f"{n:10d} {levels:6d} {result['passos']:6d} "
                f"{result['cpu_s']:7.4f} {result['gpu_s']:7.4f} "
                f"{result['speedup']:7.2f}x {result['erro_max_abs']:.3e} "
                f"{result['trajetoria_MiB']:12.2f}"
            )


if __name__ == "__main__":
    main()
