"""Sanidade e benchmark mínimo do campo podado em CPU e CUDA."""

import statistics
import time
import unittest

import numpy as np

import backend
import campos


def benchmark_campo(n_particulas, n_max, repeticoes=5):
    """Compara CPU vetorizada e GPU; retorna tempos medianos em segundos."""
    rng = np.random.default_rng(2026)
    x = rng.uniform(-1.0, 1.0, n_particulas)
    y = rng.uniform(-1.0, 1.0, n_particulas)
    saida = {}
    for nome, func in (
        ("cpu", lambda: campos.campo_total(x, y, 0.25, n_max, backend_name="cpu")),
        ("gpu", lambda: campos.campo_total(x, y, 0.25, n_max, backend_name="cuda")),
    ):
        func()  # aquecimento, incluindo compilação JIT
        tempos = []
        for _ in range(repeticoes):
            inicio = time.perf_counter()
            func()
            backend.synchronize("cuda" if nome == "gpu" else "cpu")
            tempos.append(time.perf_counter() - inicio)
        saida[nome] = statistics.median(tempos)
    return saida


class TestCampoSanidade(unittest.TestCase):
    def test_cpu_vetorizada_igual_a_referencia_podada(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, 128)
        y = rng.uniform(-1.0, 1.0, 128)
        for n in (1, 2, 4):
            ref = campos.campo_total_podado_2(x, y, 0.31, n)
            cpu = campos.campo_total(x, y, 0.31, n, backend_name="cpu")
            np.testing.assert_allclose(cpu[0], ref[0], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(cpu[1], ref[1], rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(backend.is_cuda_available(), "CuPy/GPU CUDA indisponível")
    def test_gpu_igual_a_referencia_podada(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, 256)
        y = rng.uniform(-1.0, 1.0, 256)
        for n in (1, 2, 4):
            ref = campos.campo_total_podado_2(x, y, 0.31, n)
            gpu = campos.campo_total(x, y, 0.31, n, backend_name="cuda")
            np.testing.assert_allclose(backend.to_cpu(gpu[0]), ref[0], rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(backend.to_cpu(gpu[1]), ref[1], rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
