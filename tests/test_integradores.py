import unittest

import numpy as np

from integradores import solve_RK4


class TestRK4Armazenamento(unittest.TestCase):
    def test_estado_final_sem_historico_equivale_a_trajetoria_completa(self):
        def campo(t, r):
            return np.column_stack((-r[:, 1], r[:, 0]))

        r0 = np.array([[1.0, 0.0], [0.5, -0.25]])
        t_completo, trajetoria = solve_RK4(
            campo, r0, 0.0, 0.01, 0.1, backend_name="cpu"
        )
        t_final, estado_final = solve_RK4(
            campo, r0, 0.0, 0.01, 0.1,
            backend_name="cpu", guardar_trajetoria=False
        )

        np.testing.assert_array_equal(t_final, t_completo)
        np.testing.assert_allclose(estado_final, trajetoria[-1], rtol=0, atol=0)
        self.assertEqual(estado_final.shape, r0.shape)


if __name__ == "__main__":
    unittest.main()
