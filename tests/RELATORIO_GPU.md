# Diagnóstico NumPy → CuPy — 29/10/2026

**Conclusão observada:** o código executa kernels CUDA na RTX A4500, mas a implementação podada aproveita muito pouco do paralelismo disponível. Nos casos com hierarquia, CuPy aquecido ficou aproximadamente **28 vezes mais lento** que a referência NumPy. O principal problema é a granularidade do algoritmo: Python coordena operações pequenas por partícula, com sincronizações e cópias escalares frequentes.

Nenhuma equação, parâmetro físico, algoritmo de produção ou resultado experimental foi alterado. As alterações e resultados que já existiam no Git antes da investigação foram preservados.

## Ambiente e método

- NVIDIA RTX A4500, 56 multiprocessadores, capacidade computacional 8.6, 21.018.640.384 bytes de memória total (~19,58 GiB).
- CPU Intel Core i9-13900K; Python 3.13.15; NumPy 2.5.2; CuPy 14.2.0; driver 595.99.02. CUDA runtime 12.9 (`runtime=12090` no JSON); a indicação CUDA 13.2 do `nvidia-smi` é a capacidade do driver.
- O Python ativo não tinha CuPy. Foi criado `/tmp/turbulence-gpu-diag-venv`, com acesso aos pacotes existentes e instalação isolada de `cupy-cuda12x==14.2.0` e `nvidia-cuda-runtime-cu12==12.4.127` (headers). Bibliotecas CUDA/NVRTC existentes na máquina foram utilizadas. Nenhuma dependência do projeto foi modificada.
- A GPU executa os módulos de produção importados normalmente. A referência CPU carrega as mesmas definições de `campos.py`, `integradores.py` e `funcoes_para_centros.py` em namespaces privados, substituindo o backend `cp` por NumPy. Imports e blocos principais são excluídos nessa referência. Portanto, compara o mesmo algoritmo atual, não uma versão histórica possivelmente diferente.
- Entradas idênticas, float64, gerador NumPy `default_rng(812)`, posições uniformes em `[0.2,0.4] × [0.15,0.3]`. Mesmo `c`, `alpha`, raios, períodos e lambda de produção. Campo avaliado em `t=0.137`; RK4 temporal com `dt=1/1024` fixo. Ao variar passos, varia-se o tempo final. Ao variar partículas ou níveis, mantêm-se `dt`, passos e distribuição inicial.
- Medianas de três repetições após primeira chamada separada. GPU medida com CUDA Events e relógio de parede com sincronização ao final. CUDA Events abrangem também intervalos sem kernels enquanto Python prepara operações; não representam soma do tempo ativo dos kernels. O trace fornece essa distinção. Método conforme a [documentação CuPy](https://docs.cupy.dev/en/stable/user_guide/performance.html).
- `GPU total` inclui H2D da nuvem predefinida, alocação de trajetórias, RK4 e D2H da trajetória completa. CPU total corresponde ao RK4 sobre essa mesma entrada CPU. Geração da entrada comum, imports, primeira compilação, estatísticas, gráficos e arquivos ficam fora desse total aquecido; preparação inicial é discriminada abaixo. Não é uma medição de ponta a ponta dos drivers que produzem gráficos.
- Casos pequenos primeiro, até 512 partículas/4 níveis/8 passos, sem produto cartesiano. Controle vetorizado de até 65.536 partículas restrito ao nível 1, sem explosão da árvore. Não foram executadas as suítes científicas longas que gravam resultados.

## Mapeamento de execução

| Componente | CPU | GPU e comunicação |
|---|---|---|
| `campo_2` / `campo_2_com_index` | Chamada e decisão do `if` | Diferenças, raios, máscara, exponencial e atribuições em CuPy; `if cp.any(mask)` traz resultado para decisão host; indexação booleana compacta arrays |
| `campo_total_podado` | Laços por nível e **por partícula**, listas e controle de poda | Cada partícula gera operações sobre poucos centros; cria dois `cp.array([0.0])` por partícula em cada avaliação; máscaras, compactações, reduções e atualizações escalares GPU |
| `campo_total_correto` | Laços sobre todos os centros e cálculo recursivo NumPy de `x_centros`; número de centros cresce como `4**(n-1)` | Campo de cada centro vetorizado sobre partículas; acumulação GPU; sincronizações por centro |
| Hierarquia | `R`, `T`, `Omega`, `phi`, `indices_nivel`, `x_centros`: escalares/NumPy e recursão CPU | `subcentro`, `centros_nivel_todos`, `atualiza_centro`, sondagem vetorizada: CuPy, porém laços externos continuam Python; `subcentro` opera sobre somente 4 deslocamentos |
| RK4 | Laço temporal sequencial e vetor de tempos NumPy; quatro chamadas ao campo por passo | `r0`, trajetória, estágios `k1`–`k4` e operações aritméticas ficam na GPU; não há cópia completa da trajetória para CPU a cada passo |
| Euler–Maruyama e simuladores | Controle temporal Python e cálculo de número de passos | Campos, ruído CuPy e trajetórias GPU; `solve_euler_maruyama` isolado trabalha com apenas uma trajetória 2D |
| Estatísticas e visualização | NumPy para distribuições, gráficos, compressão e I/O | Médias/variâncias/covariância temporais usam CuPy; `cp.asnumpy` e `float` transferem resultados; há cópia redundante do estado final depois de copiar toda a trajetória em `testar_estabilidade_temporal` |
| `plot_campos.py` | Grade NumPy, magnitude e gráficos | H2D da grade, avaliação GPU, D2H das velocidades |

`cp.asarray` sobre um array que já é CuPy e não requer mudança de dtype não implica nova transferência. O vetor de tempos NumPy é pequeno e `float(t[i])` é CPU; não se deve confundi-lo com a extração sincronizante de um escalar CuPy. A interface escalar de `campo_total_podado` termina em dois `float(...)`, exigindo resultados host. `comparar.py` compara dois algoritmos GPU, não NumPy versus CuPy, e usa uma única partícula.

## Tempos observados

Valores em **ms**, aquecidos; speedup = CPU RK4 / GPU total. Valores inferiores a 1 indicam GPU mais lenta.

| Partículas | Níveis | Passos | Campo CPU | Campo GPU | RK4 CPU | RK4 GPU | GPU total | Speedup total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 0.030 | 0.488 | 0.168 | 4.141 | 4.086 | 0.0412 |
| 8 | 2 | 2 | 0.149 | 3.678 | 1.105 | 30.060 | 30.581 | 0.0361 |
| 32 | 3 | 2 | 0.887 | 24.356 | 6.874 | 192.559 | 191.231 | 0.0359 |
| 128 | 3 | 2 | 3.340 | 94.288 | 26.569 | 750.245 | 751.382 | 0.0354 |
| 512 | 3 | 2 | 13.149 | 372.552 | 105.163 | 2967.972 | 2970.149 | 0.0354 |
| 32 | 1 | 2 | 0.025 | 0.468 | 0.160 | 4.072 | 4.113 | 0.0390 |
| 32 | 2 | 2 | 0.470 | 12.327 | 3.577 | 100.405 | 100.066 | 0.0357 |
| 32 | 4 | 2 | 1.269 | 34.956 | 10.152 | 279.808 | 279.797 | 0.0363 |
| 32 | 3 | 8 | 0.880 | 24.561 | 27.427 | 770.520 | 767.327 | 0.0357 |
| 8192 | 1 | 2 | 0.169 | 0.623 | 1.287 | 4.933 | 5.043 | 0.2553 |
| 65536 | 1 | 2 | 0.951 | 0.632 | 8.921 | 5.131 | 5.639 | 1.5821 |

O controle de nível 1, já vetorizado sobre partículas, chega a **1,58× de speedup total** com 65.536 partículas. Isso demonstra que a GPU pode ganhar quando recebe um lote maior; não autoriza extrapolar esse ganho para a hierarquia podada. Não foi medido o limiar exato de cruzamento.

Preparação inicial medida (carregamento dos módulos, contexto e operação mínima): **552.4 ms**. Cache de kernels dedicado e novo em `/tmp/turbulence-cupy-cache-final`. A primeira avaliação do campo N=1/L=1 levou **840.4 ms**, contra 0,488 ms aquecida. A primeira integração desse caso levou 56,7 ms, contra 4,141 ms aquecida. Novas formas/strides e caminhos também podem disparar compilação: primeira avaliação N=128/L=3 levou 2.135,8 ms e N=512/L=3, 2.337,9 ms. Esses valores incluem compilação e preparação; não isolam o compilador. As primeiras chamadas posteriores não são processos frios independentes.

Transferências explícitas medidas separadamente (medianas de parede; incluem overhead da chamada/alocação):

| N / níveis / passos | H2D entrada | D2H trajetória | RK4 CUDA Events |
|---|---:|---:|---:|
| 1/1/2 | 0.0523 | 0.0168 | 4.136 |
| 8/2/2 | 0.0292 | 0.0158 | 30.053 |
| 32/3/2 | 0.0288 | 0.0159 | 192.551 |
| 128/3/2 | 0.0371 | 0.0175 | 750.236 |
| 512/3/2 | 0.0392 | 0.0203 | 2967.962 |
| 32/1/2 | 0.0371 | 0.0171 | 4.066 |
| 32/2/2 | 0.0281 | 0.0167 | 100.397 |
| 32/4/2 | 0.0303 | 0.0168 | 279.800 |
| 32/3/8 | 0.0307 | 0.0179 | 770.513 |

As pequenas diferenças entre RK4 e total decorrem de execuções independentes e flutuação; não são tempos subtraíveis. Os JSON preservam os demais tempos de parede e CUDA Events. Não há estimativa de incerteza estatística com apenas três repetições.

## Evidência direta do paralelismo e dos gargalos

Trace Nsight Systems 2023.4.4, capturando **somente RK4 aquecido, N=32, níveis=3, passos=4**, sem temporizadores do benchmark nem transferência final da trajetória. Conforme os [relatórios oficiais do Nsight](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), foram analisados kernels, APIs e cópias CUDA:

- **32.820 kernels CUDA**, incluindo `cupy_exp`, `cupy_any`, operações aritméticas, máscaras e reduções CUB. Confirma execução de cálculos do modelo, não apenas reconhecimento da GPU.
- **Todos os kernels têm grid de um único bloco**; 23.993 deles (~73%) têm até 8 threads por bloco. Não há sobreposição entre kernels no trace. Cada kernel só pode utilizar um dos 56 SMs; não há distribuição simultânea entre os 56 SMs nesse caso. Isso não constitui medição de ocupação de warps: não foram coletados contadores Nsight Compute.
- Soma/união temporal dos kernels: **52,81 ms**, em um intervalo entre primeiro e último kernel de **489,08 ms**: **10,8%** desse intervalo com algum kernel ativo. Não é o percentual de capacidade aritmética aproveitada. O profiler adiciona overhead; os tempos comparativos vêm das execuções sem profiler.
- **6.289 chamadas `cudaStreamSynchronize`**. Além de `if cp.any(mask)`, as compactações booleanas precisam determinar tamanhos: **5.248 `cupy_scan_naive`** e **5.200 `cupy_getitem_mask`**.
- **6.288 cópias D2H, somando apenas 22.032 bytes**, e **1.024 H2D, somando 8.192 bytes**. O problema dessas cópias é a frequência/latência e dependência host, não largura de banda ou volume de trajetórias.
- As 1.024 H2D correspondem exatamente a `4 passos × 4 avaliações RK4 × 32 partículas × 2 arrays de centro raiz`; forte evidência estrutural da criação repetida de `cp.array([0.0])`.
- Tempo host em lançamentos: aproximadamente **69,46 ms** (`cuLaunchKernel` + `cudaLaunchKernel`); em `cudaMemcpyAsync`: **54,78 ms**. São parcelas instrumentadas de APIs, não uma decomposição completa do tempo nem valores somáveis ao tempo GPU.

Portanto: **A (sobrecarga), B (pouco paralelismo efetivo) e D (organização inadequada à GPU)** são dominantes; **C também existe em milhares de transferências escalares implícitas**. A cópia em bloco inicial/final não explica a lentidão. Não há evidência de esgotamento de VRAM nem necessidade de kernels personalizados neste momento.

Memória: o maior pool reservado observado nos casos hierárquicos foi **1.228.288 bytes (~1,17 MiB)**. No caso N=512/2 passos, a trajetória tem 24.576 bytes. A memória total ocupada do dispositivo ficou perto de 1,27 GiB e inclui contexto, desktop e outros processos; não equivale a memória de arrays desta simulação nem a pico de alocação. Uma amostra `nvidia-smi` durante a execução mostrou 25% GPU-Util, 1.305 MiB e 56,16 W, contra 14%, 1.111 MiB e 20 W antes. Como há desktop/outros processos, essa amostra global não atribui atividade exclusivamente ao modelo; a atribuição é feita pelo trace do processo. Consumo de VRAM não comprova eficiência.

A trajetória residente ocupa aproximadamente `(passos+1)*partículas*2*8` bytes em float64. Isso pode crescer em simulações longas, mas não foi o gargalo destes casos. Não foi feita medição de pico transitório de memória.

## Correção e reprodutibilidade

- Campo podado CPU/GPU: 66 pontos, níveis 1–4, tempos 0 e 0,137. Maior diferença absoluta: **2,45e-15** (arredondado para cima).
- Campo completo CPU/GPU e podado/completo: passaram, diferenças da ordem de 1e-15. Máscaras de sondagem CPU/GPU coincidiram exatamente nas amostras testadas.
- RK4 estático e temporal: oito partículas, três níveis, quatro passos; passaram. Todos os nove casos de benchmark também passaram; diferença máxima de trajetórias abaixo de 1e-15.
- Simulador estocástico oficial: oito partículas, três níveis, `dt_1=1/1024`, `t_max=4/1024`, `sigma0=0.001`, estático e temporal; resultados coincidiram quando se forneceram **as mesmas amostras de ruído e perturbação inicial**. Somente nessa validação, o RNG foi substituído temporariamente por um gerador NumPy local com conversão para o backend correspondente. Essas cópias não participam dos benchmarks nem do perfil.
- Tolerâncias float64: `rtol=1e-10`, `atol=1e-12`. São verificações locais e curtas; não provam equivalência estatística de longo prazo nem convergência física. Não foi investigada estocasticidade espontânea a partir de divergência numérica.
- `tests/test_estocasticidade.py` e `tests/test_comparacao_simuladores.py` chamam `np.random.seed`, mas os simuladores usam `cp.random`: essa semente **não controla** o ruído ou nuvem CuPy. Além disso, geradores NumPy/CuPy não precisam produzir a mesma sequência mesmo com a mesma semente. Por isso foi necessário compartilhar as amostras para a comparação numérica.

## Recomendações, em ordem de impacto esperado

1. **Processar conjuntamente partículas e centros ativos por nível**, mantendo o critério de poda, em vez do laço Python por partícula. A evidência para essa mudança é o grid de um bloco e o crescimento quase linear do custo com N. Estudar arrays compactos de pares partícula/centro ou blocos limitados; não materializar cegamente todas as combinações `N × 4**níveis`.
2. **Eliminar decisões host por partícula e compactações repetidas**, mantendo máscaras de tamanho conhecido onde viável. Validar a mesma poda antes/depois. Isso ataca as milhares de sincronizações e cópias escalares.
3. **Reutilizar os arrays da hierarquia e os deslocamentos comuns às partículas**; evitar as duas alocações/cópias de raiz por partícula por avaliação. No caso estático, posições dos centros não mudam no tempo; no temporal, reutilização deve respeitar cada tempo intermediário RK4.
4. **Reduzir temporários e lançamentos elementares** após agrupar o trabalho. Só então avaliar fusão CuPy ou kernel personalizado, se novo perfil ainda justificar. Não reduzir precisão ou alterar o campo para obter ganho.
5. **Usar NumPy para lotes pequenos enquanto o caminho GPU continuar assim**. Medir o ponto de cruzamento após qualquer otimização. Aquecer kernels e registrar preparação separadamente em medições de produção.
6. **Corrigir futuramente sementes e temporização dos drivers científicos**; sincronizar antes de encerrar o cronômetro de integração, separar estatística/gráficos/I/O e evitar cópia redundante do estado final. Não foram feitas essas alterações nesta investigação.

## Reprodução e arquivos

Comandos executados (as saídas experimentais existentes não são usadas):

```bash
python -m venv --system-site-packages /tmp/turbulence-gpu-diag-venv
/tmp/turbulence-gpu-diag-venv/bin/python -m pip install cupy-cuda12x==14.2.0 nvidia-cuda-runtime-cu12==12.4.127
CUPY_CACHE_DIR=/tmp/turbulence-cupy-cache-final /tmp/turbulence-gpu-diag-venv/bin/python tests/diagnostico_gpu.py --output tests/diagnostico_gpu_resultados.json
CUPY_CACHE_DIR=/tmp/turbulence-cupy-cache /tmp/turbulence-gpu-diag-venv/bin/python tests/diagnostico_gpu.py --batch-control --output /tmp/turbulence-gpu-controle.json
python tests/diagnostico_gpu.py --cpu-only --output /tmp/turbulence-cpu-verificacao.json
CUPY_CACHE_DIR=/tmp/turbulence-cupy-cache-final nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=/tmp/turbulence-gpu-trace-final /tmp/turbulence-gpu-diag-venv/bin/python tests/diagnostico_gpu.py --profile
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i /tmp/turbulence-gpu-trace-final.qdstrm -o /tmp/turbulence-gpu-trace-final.nsys-rep
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv /tmp/turbulence-gpu-trace-final.nsys-rep
python tests/analisar_perfil_gpu.py /tmp/turbulence-gpu-trace-final.sqlite
```

O importador foi chamado manualmente porque o empacotamento local do `nsys` não o encontrou automaticamente. Todos os comandos de medição/análise concluíram; a primeira tentativa CuPy sem headers falhou e foi resolvida instalando os headers somente no ambiente temporário. Uma tentativa de instalação do extra completo `[ctk]` foi interrompida antes de instalar esse conjunto, pois as bibliotecas necessárias já estavam no sistema.

Para repetir, escolha **outro caminho de saída**: o script recusa sobrescrever JSON existente. Use um novo diretório de cache para estudar primeira compilação; use o mesmo para cache aquecido. Os arquivos binários de trace e SQLite ficaram em `/tmp`, fora do Git. O script de diagnóstico funciona também sem CuPy com `--cpu-only`.

Arquivos criados: `diagnostico_gpu.py` (benchmark e validação), `analisar_perfil_gpu.py` (resumo do trace), `diagnostico_gpu_resultados.json`, `diagnostico_gpu_controle.json`, `diagnostico_gpu_perfil.json` e este relatório, todos em `tests/`. Nenhum arquivo de produção foi modificado, e não houve commit/push/merge.
