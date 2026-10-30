# NumPy versus CuPy otimizado — 30/10/2026

Comparação do núcleo numérico usado em `tests/test_estocasticidade.py`: campo
podado e integração RK4 determinística. A aleatoriedade está apenas na condição
inicial. Nenhuma equação, parâmetro físico ou função de produção foi alterado.

## Método

- CPU Intel Core i9-13900K; GPU NVIDIA RTX A4500; NumPy 2.5.2; CuPy 14.2.0.
- NumPy executa as definições existentes em namespaces privados via
  `tests.diagnostico_gpu.load_backend(np)`, sem alterar imports de produção.
  O campo NumPy é o algoritmo podado de referência, com laços Python por partícula;
  CuPy usa `campo_total_podado` e seu kernel CUDA otimizado. Não é uma comparação
  com uma implementação NumPy vetorizada ou com CPU compilada.
- `float64`, semente 812, mesma nuvem pré-gerada em NumPy e copiada para a GPU.
  Primeira partícula em `r0=[0.3, 0.2333]`; demais uniformes em área no disco de
  raio `0.1*R(n)/2`, seguindo a implementação efetiva de `simular_particulas`.
- `dt=0.01*T(n)`; `t_max=passos*dt`. Mesmos valores para CPU/GPU em cada caso.
  Mantidos `R_1=1`, `T_1=1`, `lambda=2**(2/3)`, `c=0.6` e `alpha=sqrt(2.5)`.
  Ao variar níveis, dt e tempo final também variam; isto não é um teste de
  convergência com tempo final fixo.
- Caso pequeno executado primeiro. Medianas de três repetições após aquecimento.
  Tempo de parede sincronizado; CUDA Events também registrados no JSON.
  As primeiras chamadas ficam separadas dos tempos aquecidos.
- Integração GPU: entradas já no dispositivo e trajetória mantida na GPU.
  GPU com cópias: CPU→GPU, integração e trajetória completa GPU→CPU.
  CPU: mesma integração e trajetória completa na memória da CPU.
- Geração da nuvem, validação estatística, gráficos e gravação dos resultados
  científicos não entram nos tempos. Os ganhos abaixo não são do driver completo.

## Tempos observados

Tempos em ms; ganho = NumPy / (CuPy com cópias). Valores menores que 1
indicam vantagem para NumPy.

| Partículas | Níveis | Passos | Campo | NumPy | CuPy GPU | CuPy com cópias | Ganho |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | 4 | estático | 0.310 | 1.487 | 1.557 | 0.20× |
| 1 | 1 | 4 | temporal | 0.328 | 1.509 | 1.622 | 0.20× |
| 1 | 3 | 4 | estático | 1.069 | 1.625 | 1.644 | 0.65× |
| 1 | 3 | 4 | temporal | 1.078 | 1.566 | 1.609 | 0.67× |
| 8 | 3 | 4 | estático | 3.968 | 1.613 | 1.661 | 2.39× |
| 8 | 3 | 4 | temporal | 4.005 | 1.584 | 1.643 | 2.44× |
| 32 | 3 | 4 | estático | 13.732 | 1.613 | 1.681 | 8.17× |
| 32 | 3 | 4 | temporal | 13.847 | 1.593 | 1.691 | 8.19× |
| 128 | 3 | 4 | estático | 53.675 | 1.630 | 1.710 | 31.38× |
| 128 | 3 | 4 | temporal | 53.629 | 1.824 | 1.660 | 32.30× |
| 512 | 3 | 4 | estático | 209.491 | 1.646 | 1.727 | 121.30× |
| 512 | 3 | 4 | temporal | 208.894 | 1.642 | 1.678 | 124.49× |
| 2048 | 3 | 4 | estático | 829.543 | 1.620 | 1.743 | 475.99× |
| 2048 | 3 | 4 | temporal | 827.353 | 1.649 | 1.698 | 487.16× |
| 8192 | 3 | 4 | estático | 3322.513 | 1.870 | 1.853 | 1792.91× |
| 8192 | 3 | 4 | temporal | 3316.713 | 1.684 | 1.915 | 1732.35× |
| 128 | 5 | 4 | estático | 105.069 | 1.692 | 1.721 | 61.04× |
| 128 | 5 | 4 | temporal | 104.725 | 1.906 | 1.755 | 59.69× |
| 128 | 3 | 16 | estático | 213.876 | 6.327 | 6.471 | 33.05× |
| 128 | 3 | 16 | temporal | 212.459 | 6.322 | 6.389 | 33.25× |

## Validação e conclusão

Todos os 20 casos passaram com `rtol=1e-10`, `atol=1e-12` nos campos, nas
trajetórias completas e em `sigma²(t)`. A nuvem inicial foi verificada por
igualdade exata em ambos os backends. Maiores erros absolutos observados:

- Campo: 7.772e-16.
- Trajetória: 2.220e-16.
- Dispersão `sigma²(t)`: 1.220e-19.

NumPy foi mais rápido com uma partícula. Com três níveis, entre os tamanhos
medidos, CuPy passou a ser mais rápido a partir de oito partículas. Os grandes
ganhos em lotes maiores refletem também os laços Python do algoritmo de referência.
Atribuir todo o ganho ao hardware exigiria uma referência CPU igualmente otimizada.
Os tempos foram medidos em execuções independentes; pequenas diferenças entre as
medianas GPU com e sem cópias podem refletir variação de execução.

Três repetições e trajetórias curtas não caracterizam simulações longas nem
estabelecem estocasticidade espontânea. Nenhum resultado em `tests/resultados`
foi alterado por estas execuções.

## Arquivos e comandos executados

- `tests/benchmark_estocasticidade_gpu.py`: modo NumPy/CuPy com `--case`,
  `--repeats` e `--seed`, validação de sigma² e parâmetros físicos no JSON.
- `tests/benchmark_numpy_cupy_resultados.json`: medições completas desta execução.
- `README_TESTES.md`: instruções do benchmark.

O Python padrão desta sessão não possui CuPy. Foi usado o ambiente existente
`/tmp/turbulence-gpu-diag-venv`, sem instalar dependências.

```bash
# Compilação e interface: passaram.
/tmp/turbulence-gpu-diag-venv/bin/python -m py_compile tests/benchmark_estocasticidade_gpu.py
/tmp/turbulence-gpu-diag-venv/bin/python -m tests.benchmark_estocasticidade_gpu --help

# Caso pequeno: quatro combinações, todas passaram.
/tmp/turbulence-gpu-diag-venv/bin/python tests/benchmark_estocasticidade_gpu.py \
    --numpy-vs-cupy --case 1,1,2 --case 8,3,2 --repeats 1 \
    --output /tmp/turbulence-numpy-cupy-smoke-20261030.json

# Grade padrão: vinte combinações, todas passaram.
/tmp/turbulence-gpu-diag-venv/bin/python tests/benchmark_estocasticidade_gpu.py \
    --numpy-vs-cupy --output tests/benchmark_numpy_cupy_resultados.json
```

Também passaram quatro verificações da CLI por subprocesso: caso com zero
partículas, repetições zero, opções sem `--numpy-vs-cupy` e saída já existente
foram recusados com código 2. O benchmark não sobrescreve o JSON; para repetir
a medição, forneça outro caminho de saída.
