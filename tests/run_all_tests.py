"""Executa o benchmark CPU/GPU e, em seguida, o teste de estocasticidade."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend
import campos
from tests.benchmark_rk4_cpu_gpu import executar as executar_benchmark
from tests.test_estocasticidade import executar_teste_estocasticidade


def main():
    if not backend.is_cuda_available():
        _, motivo = backend.get_cuda_status()
        raise SystemExit(f"Erro: GPU NVIDIA/CUDA com CuPy indisponível. {motivo}")

    # O teste de estocasticidade sobrescreve arquivos em seu diretório de saída.
    # Verifique os caminhos antes de iniciar o benchmark ou qualquer simulação.
    output_dir = os.path.join("resultados", "estocasticidade_estatico")
    if os.path.exists(output_dir):
        raise SystemExit(
            f"Erro: o caminho de saída já existe ({output_dir}). "
            "Os resultados anteriores foram preservados; escolha ou arquive "
            "esse diretório antes de executar novamente."
        )

    print("\n=== 1/2: Benchmark RK4 CPU/GPU ===")
    for n_particulas in (100, 1000, 10000, 100000):
        for n_niveis in (3, 5):
            resultado = executar_benchmark(
                n_particulas=n_particulas,
                n_niveis=n_niveis,
                dt=0.01,
                t_max=0.1,
                repeticoes=3,
                alpha=campos.alpha_padrao,
                c=campos.c_padrao,
            )
            print(
                f"partículas={n_particulas} níveis={n_niveis} "
                f"passos={resultado['passos']} CPU={resultado['cpu_s']:.4f}s "
                f"GPU={resultado['gpu_s']:.4f}s "
                f"speedup={resultado['speedup']:.2f}x "
                f"erro_max={resultado['erro_max_abs']:.3e} "
                f"trajetória={resultado['trajetoria_MiB']:.2f} MiB"
            )

    print("\n=== 2/2: Estocasticidade para N = 1, 2, 3, 4 ===")
    executar_teste_estocasticidade(
        N_particulas=100,
        t_max=1 / 5,
        seed=42,
        lista_N_fields=(1, 2, 3, 4),
        backend_name="cuda",
        output_dir=output_dir,
    )

    print("\nExecução concluída. Resultados salvos em 'resultados/'.")


if __name__ == "__main__":
    main()
