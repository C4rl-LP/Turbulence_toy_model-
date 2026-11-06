"""Executa o benchmark CPU/GPU e, em seguida, o teste de estocasticidade."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import backend
import campos
from tests.benchmark_rk4_cpu_gpu import executar as executar_benchmark
from tests.test_estocasticidade import executar_teste_estocasticidade
from tests.test_estocasticidade_parametros import executar_teste_estocasticidade_parametros

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
    for n in range(1,21):
    
        executar_teste_estocasticidade_parametros(N_fields=n, temporal=True, lista_alpha = [(2.5)**(1/2)],t_max = 0.2, lista_c = [0.2, 0.4, 0.6, 0.8, 1.0], N_particulas= 10000, backend_name= "cuda")

    print("\nExecução concluída. Resultados salvos em 'resultados/'.")


if __name__ == "__main__":
    main()
