# NumPy CPU versus CuPy otimizado GPU — 30/10/2026

Comparação direta no campo e na integração RK4 do teste de estocasticidade espontânea.
O teste usa evolução determinística de uma nuvem inicial aleatória. Não foram
executadas simulações científicas longas nem alterados resultados existentes.

## Referência e método

- GPU NVIDIA RTX A4500; NumPy 2.5.2; CuPy 14.2.0; float64.
- CPU: mesmas funções de campos/integradores carregadas com backend NumPy pelo
  `load_backend` do diagnóstico existente. Preserva o algoritmo podado original,
  incluindo seus laços Python. Não é uma implementação CPU adicional otimizada.
- GPU: `campo_total_podado`, que seleciona o RawKernel já validado.
- Nuvem única pré-gerada, distribuição uniforme em área no disco em torno de
  r0=[0.3,0.2333], dimensão=0.1*R(n) e raio=0.05*R(n), conforme `simular_particulas`.
  Amostras NumPy default_rng(812) idênticas nos dois backends; cópia H2D sem mudar valores.
- dt=0.01*T(n), seguindo o padrão do teste existente. Mantido fixo antes/depois em
  cada linha. Ao variar níveis, dt acompanha T(n); não é um estudo de convergência.
  A maioria dos casos tem quatro passos; há também uma verificação com 16 passos.
- Tempos de parede em ms, medianas de três repetições após aquecimento. GPU também
  medida com CUDA Events; espera final explícita. Events constam no JSON.
- RK4 GPU mede entrada já residente; GPU total inclui H2D da nuvem, alocação e RK4,
  mais D2H da trajetória completa. CPU RK4 usa essa mesma entrada já preparada.
- Geração da nuvem, compilação, imports/contexto, estatísticas, gráficos e I/O
  ficam fora. Portanto, é comparação do núcleo da simulação, não do teste completo
  com gráficos. As primeiras chamadas são registradas separadamente no JSON e
  não correspondem a processos frios independentes.
- Campo temporal avaliado isoladamente em t=0.137; estático em t=0.

Estes speedups comparam a implementação NumPy original com o kernel GPU otimizado;
não isolam a diferença intrínseca entre hardware CPU e GPU nem entre bibliotecas.

## Resultados medidos

Ganho = CPU RK4 / GPU total. Acima de 1 favorece GPU; abaixo de 1 favorece CPU.

| Partículas | Níveis | Passos | Campo | CPU RK4 (ms) | GPU RK4 (ms) | GPU total (ms) | Ganho |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | 4 | estático | 0.306 | 1.513 | 1.560 | 0.20× |
| 1 | 1 | 4 | temporal | 0.313 | 1.520 | 1.662 | 0.19× |
| 1 | 3 | 4 | estático | 1.067 | 1.589 | 1.670 | 0.64× |
| 1 | 3 | 4 | temporal | 1.061 | 1.601 | 1.659 | 0.64× |
| 8 | 3 | 4 | estático | 3.948 | 1.679 | 1.740 | 2.27× |
| 8 | 3 | 4 | temporal | 3.983 | 1.713 | 1.800 | 2.21× |
| 32 | 3 | 4 | estático | 13.731 | 1.645 | 1.715 | 8.01× |
| 32 | 3 | 4 | temporal | 13.831 | 1.672 | 1.802 | 7.67× |
| 128 | 3 | 4 | estático | 53.204 | 1.671 | 1.766 | 30.12× |
| 128 | 3 | 4 | temporal | 53.435 | 1.669 | 1.771 | 30.17× |
| 512 | 3 | 4 | estático | 208.354 | 1.691 | 1.770 | 117.74× |
| 512 | 3 | 4 | temporal | 209.253 | 1.677 | 1.796 | 116.49× |
| 2048 | 3 | 4 | estático | 829.621 | 1.711 | 1.822 | 455.43× |
| 2048 | 3 | 4 | temporal | 827.596 | 1.710 | 1.842 | 449.35× |
| 8192 | 3 | 4 | estático | 3320.223 | 2.157 | 2.389 | 1389.86× |
| 8192 | 3 | 4 | temporal | 3305.699 | 1.681 | 1.867 | 1770.50× |
| 128 | 5 | 4 | estático | 104.898 | 1.716 | 1.766 | 59.41× |
| 128 | 5 | 4 | temporal | 104.562 | 1.713 | 1.843 | 56.72× |
| 128 | 3 | 16 | estático | 213.525 | 6.511 | 6.638 | 32.17× |
| 128 | 3 | 16 | temporal | 213.678 | 6.487 | 6.648 | 32.14× |

Para 3 níveis/4 passos, o lote de uma partícula favorece CPU, e o lote de oito já
favorece GPU nos dois tipos de campo. Não se mediu o limiar exato entre esses tamanhos.
Os resultados dependem da nuvem, quantidade de centros ativos e carga compartilhada.
Não há estimativa de incerteza estatística com somente três repetições.

## Avaliação isolada do campo temporal

| Partículas | Níveis | CPU campo (ms) | GPU campo (ms) | Ganho |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.020 | 0.071 | 0.29× |
| 1 | 3 | 0.067 | 0.060 | 1.11× |
| 8 | 3 | 0.246 | 0.067 | 3.68× |
| 32 | 3 | 0.864 | 0.066 | 13.07× |
| 128 | 3 | 3.398 | 0.080 | 42.34× |
| 512 | 3 | 13.077 | 0.079 | 166.26× |
| 2048 | 3 | 51.905 | 0.082 | 632.68× |
| 8192 | 3 | 207.577 | 0.096 | 2168.00× |
| 128 | 5 | 6.585 | 0.093 | 70.84× |

## Equivalência, memória e arquivos

Todas as 20 configurações passaram com rtol=1e-10 e atol=1e-12, incluindo campos
estáticos/temporais e trajetórias de quatro e 16 passos. A nuvem inicial foi
verificada exatamente igual após as transferências. Maior erro absoluto do campo:
7.78e-16; das trajetórias completas: 2.23e-16.

Maior pool reservado GPU amostrado: 4.019.712 bytes (3.83 MiB). Maior trajetória:
655.360 bytes (0.625 MiB). Não são estimativas de pico transitório, VRAM de contexto
ou uso de memória CPU. O campo preserva a poda; não há árvore completa materializada.

Foi ampliado somente `tests/benchmark_estocasticidade_gpu.py`, com a opção
`--numpy-vs-cupy`, reutilizando a referência e temporizadores do diagnóstico.
Produção e testes científicos existentes permaneceram inalterados nesta etapa.
Criados este relatório e `tests/benchmark_numpy_cupy_estocasticidade_resultados.json`.
Não houve commit, push ou merge.

Comandos executados, ambos concluídos sem falhas:

```bash
/tmp/turbulence-gpu-diag-venv/bin/python tests/benchmark_estocasticidade_gpu.py --numpy-vs-cupy --output /tmp/turbulence-estocasticidade-numpy-vs-cupy.json
python -m py_compile tests/benchmark_estocasticidade_gpu.py
```

Para repetir, use outro caminho de saída; o benchmark recusa sobrescrever JSON.
