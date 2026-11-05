"""
Testes unitários para o módulo backend.py.
Valida seleção de backend, detecção de hardware, tratamento de erros e conversão de arrays.
"""

import unittest
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import backend


class TestBackend(unittest.TestCase):

    def test_cuda_status_types(self):
        avail, reason = backend.get_cuda_status()
        self.assertIsInstance(avail, bool)
        self.assertIsInstance(reason, str)
        self.assertEqual(backend.is_cuda_available(), avail)

    def test_cpu_backend_resolution(self):
        xp = backend.get_backend_module("cpu")
        self.assertIs(xp, np)
        self.assertEqual(backend.resolve_backend("cpu"), "cpu")

    def test_auto_backend_resolution(self):
        resolved = backend.resolve_backend("auto")
        if backend.is_cuda_available():
            self.assertEqual(resolved, "cuda")
        else:
            self.assertEqual(resolved, "cpu")
            self.assertIs(backend.get_backend_module("auto"), np)

    def test_cuda_backend_behavior(self):
        if not backend.is_cuda_available():
            with self.assertRaises(backend.CudaNotAvailableError) as ctx:
                backend.resolve_backend("cuda")
            self.assertIn("O backend CUDA foi solicitado, mas não está disponível", str(ctx.exception))

            with self.assertRaises(backend.CudaNotAvailableError):
                backend.get_backend_module("cuda")
        else:
            self.assertEqual(backend.resolve_backend("cuda"), "cuda")
            xp = backend.get_backend_module("cuda")
            import cupy as cp
            self.assertIs(xp, cp)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            backend.resolve_backend("invalid_backend")
        with self.assertRaises(ValueError):
            backend.get_backend_module("tpu")

    def test_to_cpu(self):
        # Array NumPy
        arr = np.array([1.0, 2.0, 3.0])
        res = backend.to_cpu(arr)
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, arr)

        # Lista
        res_list = backend.to_cpu([1, 2, 3])
        self.assertIsInstance(res_list, np.ndarray)
        np.testing.assert_array_equal(res_list, np.array([1, 2, 3]))

        # None
        self.assertIsNone(backend.to_cpu(None))

    def test_to_device_cpu(self):
        arr = np.array([4.0, 5.0, 6.0])
        res = backend.to_device(arr, backend="cpu")
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, arr)

    def test_synchronize_cpu(self):
        # Não deve levantar exceções no backend CPU
        backend.synchronize(backend="cpu")

    def test_device_timer_cpu(self):
        timer = backend.DeviceTimer(backend="cpu")
        with timer:
            time.sleep(0.01)
        self.assertGreaterEqual(timer.elapsed_time, 0.009)
        self.assertLess(timer.elapsed_time, 0.1)


if __name__ == "__main__":
    unittest.main()
