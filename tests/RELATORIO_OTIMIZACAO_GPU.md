# Otimização CUDA do campo hierárquico podado — 29/10/2026

A implementação float64 passou a processar partículas em paralelo, preservando
a poda, as constantes físicas e os integradores. A implementação anterior foi
mantida integralmente como `campo_total_podado_referencia`.

## Estratégia e causa

O gargalo confirmado era a coordenação Python por partícula: compactações,
`cp.any`, reduções pequenas e cópias escalares. Um protótipo CuPy com pares
ativos agrupados por nível reduziu o campo de 32 partículas/3 níveis de
24,398 ms para 1,484 ms, e de 512 partículas de 360,708 ms para 1,673 ms.
Ainda exigia compactações de tamanho variável sincronizantes por nível.
Por isso se adotou o RawKernel autorizado, com travessia em profundidade (DFS)
independente por partícula, sem compactações nem atomics.

Cada bloco tem 128 threads. Cada thread mantém as coordenadas e o próximo
filho de cada nível em uma pilha de tamanho conhecido. Apenas filhos de pais
aceitos são gerados; o teste original `r² < R_cut²` permanece estrito. A raiz
sempre gera filhos, inclusive fora de seu próprio suporte, como na referência.
Não há matriz partículas × árvore completa. A memória da pilha cresce como
O(níveis) por thread, e não como 4**níveis. A soma muda de ordem (DFS em vez de
soma por nível), portanto não se exige identidade bit a bit.

Uma tabela pequena de raios, períodos e frequências é reutilizada por dispositivo
e valores físicos, com cache limitado a 16 configurações. Um kernel prepara os
quatro deslocamentos de cada nível no tempo solicitado; outro avalia o campo.
As posições, velocidades e temporários permanecem na GPU. Strides, incluindo
as colunas intercaladas do RK4, são lidos diretamente, sem cópias de contiguidade.
FMA está desabilitado para manter operações separadas nos critérios de corte.
Não se alteraram precisão, equações, amplitudes, sinais, períodos, `dt` ou RK4.

## Método de medição

RTX A4500, 56 SMs, 20.470 MiB segundo `nvidia-smi`, CuPy 14.2.0,
NumPy 2.5.2, Python 3.13.15. Três repetições, mediana, float64,
semente NumPy 812, posições em [0,2;0,4] × [0,15;0,3].
Campo em t=0,137; RK4 temporal com dt=1/1024 fixo. Variar passos altera o tempo
final; variar níveis ou partículas conserva os demais parâmetros da linha.
CUDA Events e relógio de parede com espera final. Os tempos abaixo são de
parede, aquecidos; os Events constam no JSON. Events também incluem lacunas
entre lançamentos enquanto Python trabalha. Total GPU inclui H2D da entrada,
integração, alocação da trajetória e D2H da trajetória inteira. CPU total é RK4
com a mesma entrada já preparada. Imports, compilação, geração da entrada,
estatística, gráficos e I/O ficam fora. Não é tempo de um driver científico completo.

Referências CPU e CuPy usam exatamente o algoritmo original. A referência CPU
é carregada pelo diagnóstico existente com backend NumPy. Os novos caminhos
são comparados no mesmo processo e hardware; nenhum ganho abaixo é extrapolado.
O CuPy antigo foi medido até 512 partículas/4 níveis/8 passos; os casos maiores
omitem essa versão para evitar repetir seu custo elevado. Um traço indica ausência
de medição, não uma estimativa. A ordem de medição foi CPU, CuPy novo, CuPy antigo.
Não há estimativa de incerteza com três repetições e o desktop compartilha a GPU.

## Desempenho medido

Todos os tempos em ms; speedup = tempo de referência / total GPU novo.

| Partículas | Níveis | Passos | Campo CPU | Campo antigo | Campo novo | RK4 CPU | RK4 antigo | RK4 novo | Total antigo | Total novo | CPU/novo | Antigo/novo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 0.021 | 0.432 | 0.048 | 0.166 | 3.806 | 0.755 | 3.840 | 0.820 | 0.20× | 4.68× |
| 8 | 2 | 2 | 0.151 | 3.580 | 0.060 | 1.101 | 28.082 | 0.808 | 29.731 | 0.868 | 1.27× | 34.25× |
| 32 | 3 | 2 | 0.870 | 23.815 | 0.076 | 6.860 | 188.025 | 0.827 | 190.158 | 0.886 | 7.75× | 214.69× |
| 128 | 3 | 2 | 3.406 | 93.414 | 0.091 | 26.746 | 730.210 | 0.857 | 732.389 | 0.943 | 28.36× | 776.59× |
| 512 | 3 | 2 | 13.457 | 356.932 | 0.093 | 106.323 | 2951.421 | 0.837 | 2895.432 | 0.903 | 117.76× | 3206.84× |
| 32 | 1 | 2 | 0.030 | 0.440 | 0.054 | 0.176 | 3.784 | 0.814 | 3.991 | 0.861 | 0.20× | 4.64× |
| 32 | 2 | 2 | 0.476 | 12.070 | 0.059 | 3.692 | 97.883 | 0.808 | 98.156 | 0.863 | 4.28× | 113.80× |
| 32 | 4 | 2 | 1.284 | 34.964 | 0.089 | 10.200 | 274.689 | 0.836 | 274.802 | 0.922 | 11.06× | 297.94× |
| 32 | 3 | 8 | 0.904 | 23.512 | 0.076 | 27.662 | 753.330 | 3.195 | 760.771 | 3.310 | 8.36× | 229.87× |
| 2048 | 3 | 2 | 52.935 | — | 0.103 | 423.953 | — | 0.860 | — | 0.922 | 459.82× | — |
| 8192 | 3 | 2 | 211.813 | — | 0.152 | 1688.926 | — | 0.890 | — | 1.027 | 1644.64× | — |
| 65536 | 3 | 2 | 1698.501 | — | 0.440 | 13546.656 | — | 2.070 | — | 2.542 | 5328.14× | — |
| 512 | 6 | 2 | 32.984 | — | 0.188 | 262.841 | — | 1.129 | — | 1.195 | 219.92× | — |
| 512 | 3 | 16 | 13.351 | — | 0.092 | 847.590 | — | 6.329 | — | 6.466 | 131.09× | — |

Os casos de 2.048 e 8.192 partículas terminaram normalmente, passaram na
comparação das trajetórias e estão no JSON completo. O maior erro absoluto de
trajetória nesses dois casos foi 5,56e-17. O lote de 65.536 também terminou:
erro máximo 1,12e-16, sem avaliar a versão CuPy antiga nesse tamanho.

## Perfil Nsight Systems e paralelismo

Captura somente RK4 aquecido, níveis=3, passos=4, sem temporizadores nem cópia
final host. O comparativo antigo de 32 partículas vem do diagnóstico preservado;
os dois traces novos foram coletados nesta etapa. O importador Nsight foi chamado
manualmente devido ao empacotamento local, e os relatórios foram exportados para SQLite.

| Medida | CuPy antigo, N=32 | Novo, N=32 | Novo, N=8.192 |
|---|---:|---:|---:|
| Kernels CUDA | 32.820 | 116 | 116 |
| `cudaStreamSynchronize` | 6.289 | 1 | 1 |
| Cópias H2D | 1.024 | 0 | 0 |
| Cópias D2H | 6.288 | 0 | 0 |
| Cópias D2D | 5 | 5 | 5 |
| Grid do kernel do campo | 1 bloco | 1 bloco | 64 blocos |
| Maior grid entre todos os kernels | 1 | 1 | 128 |

A sincronização restante é a espera explícita no fim da captura, e não uma decisão
de poda no host. Há 16 kernels `pruned_field`, 16 de preparação de deslocamentos e
84 kernels de aritmética/empilhamento do RK4: dois kernels por avaliação do campo,
independentemente do número de partículas. O campo usa 128 threads/bloco. Em N=8.192,
64 blocos permitem distribuição entre os 56 SMs; o trace confirma o grid, não mede
a ocupação por SM nem garante que todos os SMs estejam simultaneamente ocupados.
Só os 16 kernels pequenos de preparação usam um bloco nesse lote (13,8% do total).
O campo de 3 níveis usou 47 registradores/thread e 0 bytes de local memory/thread
segundo o trace; isso não se extrapola a níveis mais altos. As cinco cópias D2D
correspondem à colocação da condição inicial e dos quatro estados na trajetória.
Não houve H2D/D2H no campo/RK4 aquecido. O total dos benchmarks inclui essas cópias
nas fronteiras da integração, conforme descrito no método.

A redução de kernels no caso comparável foi 99,65%. Não se usa a duração do trace
como benchmark: a instrumentação altera tempos.

## Ponto de cruzamento e compilação

Varredura adicional com níveis=3 e passos=2 fixos, mesma distribuição e dt:

| Partículas | CPU RK4 (ms) | GPU total novo (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 0.567 | 0.930 | 0.61× |
| 2 | 0.771 | 0.954 | 0.81× |
| 4 | 1.209 | 0.958 | 1.26× |
| 8 | 2.082 | 0.960 | 2.17× |
| 16 | 3.664 | 0.967 | 3.79× |

Nessa carga, a GPU passa a ganhar a partir de 4 partículas entre os tamanhos
amostrados; com 1–2, a CPU ainda ganha. Isso depende do nível, distribuição,
quantidade de centros ativos, passos e ambiente. Não é um limiar universal.
O controle de nível 1 continua favorecendo CPU nos lotes pequenos.

Em processo separado, cache CuPy novo, contexto e entrada já preparados,
N=8.192/L=3: primeira avaliação 147,627 ms; aquecida 0,141 ms de parede
(Events: 147,620 e 0,136 ms). A primeira inclui compilação e preparação/cache;
não é uma medição isolada do compilador. As colunas `*_first_field` da bateria
principal são a primeira chamada daquela linha, após validações, e não representam
processos frios independentes. Novas especializações ainda podem compilar ali.

## Correção numérica e poda

Tolerâncias rtol=1e-10, atol=1e-12. O diagnóstico original foi executado antes e
depois da otimização. A bateria ampliada contém 30 combinações de tempo
(0 e 0,137), alpha (0,8; padrão; 2,1) e níveis (1,2,3,5,8), cada uma com 98 pontos.
Inclui centros descendentes, suporte, fronteira exata, floats adjacentes com
`nextafter` e perturbações de 1e-13 dos raios. Comparação direta com CuPy original:
erro absoluto máximo 5,33e-15. As contagens de centros ativos por partícula e
nível coincidiram exatamente com uma travessia independente CuPy baseada em
máscaras. Isso complementa a comparação do campo: contribuição praticamente
nula na fronteira, isoladamente, não demonstraria igualdade das máscaras.
Também passaram entradas vazias, escalares, strides negativos e fallback float32.

RK4 estático/temporal: 12 partículas, níveis 3,6,8, 16 passos, dt fixo 1/1024.
Maior erro absoluto em toda a trajetória e no estado final: 1,67e-16;
maior relativo: 6,03e-16. Não houve crescimento relevante nesses intervalos curtos.
Foram comparados explicitamente k1,k2,k3,k4 para NumPy, CuPy original e novo,
com 16 partículas, níveis 1,3,6,8, quatro passos, estático e temporal:
256 comparações de estágio, maior erro absoluto 4,27e-14 e relativo 4,46e-12.
Todas passaram. Aritmética entre estágios permanece no integrador original.

O simulador `simular_particulas_ruidosa` e o Euler–Maruyama isolado passaram
em campo estático e temporal com as mesmas amostras de ruído e perturbação inicial,
fornecidas pelo mecanismo de validação do diagnóstico. Não se presume que sementes
iguais em NumPy/CuPy produzam sequências iguais. Essas cópias de ruído são exclusivas
da validação e não entram nas medições de desempenho.

## Segurança da pilha

Invariante: no início de cada iteração, quadros 0..depth estão inicializados,
e 0 <= depth < L. O teste de folha `depth == L-1` ocorre antes de ler um filho.
Assim, ao descer, `level = depth+1 <= L-1`; `next[depth]` enumera apenas 0..3.
Cada novo quadro recebe cx,cy,next antes de ser utilizado. Ao voltar da raiz,
depth passa a -1 e o laço termina antes de novo acesso. A thread verifica p<N
antes de ler posições. Não há recursão CUDA nem pilha de capacidade fixa
independente do nível: a especialização aloca exatamente L quadros.

Testes construíram caminhos que efetivamente atingem níveis 1,2,8,16,32,48,64,
com 129 partículas (incluindo sobra no segundo bloco), ponto fora do domínio,
contagens independentes e comparação direta do campo original no caminho profundo.
Todos passaram. Isso testa a descida real até o último quadro, além da compilação
com níveis altos. Compute Sanitizer memcheck direcionado ao código novo terminou
com exit code 0 e `ERROR SUMMARY: 0 errors`, incluindo os sete tamanhos de pilha.
A tentativa inicial instrumentando também toda a referência CuPy foi interrompida
por custo de instrumentação (mais de 2 minutos e aproximadamente 7,6 GiB de VRAM
global); esses valores não descrevem a memória do algoritmo otimizado. O teste
completo sem instrumentação já havia passado; a versão direcionada terminou em
16 segundos e não instrumentou a referência antiga. Não equivale a validar precisão física em escalas sub-ulp.

## Memória e limites

O maior pool reservado amostrado durante a bateria foi 22.931.968 bytes
(21,87 MiB), incluindo buffers reutilizados de casos anteriores e da referência
antiga; não é pico transitório nem VRAM total do processo. No lote 65.536/3 níveis/
2 passos, arrays ainda vivos no pool somaram 4.221.952 bytes (4,03 MiB) e a
trajetória ocupou 3.145.728 bytes (3 MiB). Em 2.048 e 8.192 partículas, pools
reservados de 1.763.840 e 4.057.600 bytes; trajetórias de 98.304 e 393.216 bytes.
A pilha lógica por thread é 20*L bytes (duas coordenadas double e um inteiro);
a alocação física depende de registradores/local memory e do compilador.

O trabalho ainda depende da quantidade de centros realmente aceitos. Alpha grande
pode aumentar muito o custo mesmo sem materializar a árvore. Partículas com percursos
diferentes causam divergência de warps. A especialização compila um kernel por
número de níveis; primeira compilação deve ser separada de execução aquecida.
O cache de configuração é limitado; deslocamentos e saídas usam alocações do pool.
Precisões/formas diferentes do caminho float64 1D conservam a implementação antiga.
Chamadas com escalares Python preservam retorno float e exigem sincronização.
Concorrência entre múltiplas streams não foi validada; os testes usaram a stream padrão.
A interface interna `campo_gpu.evaluate` assume arrays 1D float64 de mesmo tamanho;
a função pública faz a seleção desse caminho.

## Arquivos da otimização

- `campos.py`: referência original preservada e despacho para float64 GPU.
- `campo_gpu.py`: preparação, cache e RawKernel DFS independente por partícula.
- `tests/otimizacao_gpu.py`: validações, fronteiras, estágios, pilha, benchmarks e captura.
- `tests/otimizacao_gpu_resultados.json`: medições e validações desta otimização.
- `tests/RELATORIO_OTIMIZACAO_GPU.md`: este relatório.

`integradores.py` e `funcoes_para_centros.py` permaneceram inalterados. Os arquivos
de diagnóstico e resultados científicos já presentes/modificados foram preservados.
Nenhum commit, push ou merge foi realizado. Não foram executados drivers científicos
longos nem escritos arquivos em `tests/resultados`.

## Próximas otimizações

O ganho principal está concluído. Se cargas de produção exigirem mais, medir antes
de alterar: fusão das operações aritméticas/empilhamento dos estágios RK4, reutilização
dos deslocamentos em tempos repetidos e estudo de ocupação/register spills com Nsight
Compute. Não se recomenda alterar o modelo, precisão ou passo temporal para ganhar tempo.
Validação longa/convergência física e equivalência estatística de longo prazo continuam
fora desta bateria curta. Não se interpreta divergência numérica como estocasticidade espontânea.

## Comandos de reprodução e resultados

Use outro caminho de saída: os scripts recusam sobrescrever JSON existente.
O ambiente temporário do diagnóstico permanece em `/tmp/turbulence-gpu-diag-venv`.

```bash
/tmp/turbulence-gpu-diag-venv/bin/python tests/diagnostico_gpu.py --validate-only --output /tmp/novo-diagnostico.json
/tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --validate-only --output /tmp/nova-validacao.json
/tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --safety-only --output /tmp/nova-pilha-estagios.json
compute-sanitizer --tool memcheck --error-exitcode 99 /tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --memcheck-only --output /tmp/novo-memcheck.json
/tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --output /tmp/novo-benchmark.json
/tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --crossover --output /tmp/novo-cruzamento.json
CUPY_CACHE_DIR=/tmp/novo-cache-frio /tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --cold --output /tmp/novo-frio.json
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --output=/tmp/novo-trace /tmp/turbulence-gpu-diag-venv/bin/python tests/otimizacao_gpu.py --profile 8192
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i /tmp/novo-trace.qdstrm -o /tmp/novo-trace.nsys-rep
nsys export --type sqlite --output /tmp/novo-trace.sqlite /tmp/novo-trace.nsys-rep
python tests/analisar_perfil_gpu.py /tmp/novo-trace.sqlite
python -m py_compile campos.py campo_gpu.py tests/otimizacao_gpu.py
git diff --check -- campos.py
```

Todos os testes finais acima concluíram, com os caminhos efetivos `/tmp/turbulence-*`
registrados no JSON consolidado. Os comandos de validação e compilação passaram.
Para o trace de 32 partículas também foi usado `--profile 32`. Os arquivos grandes
de trace/SQLite ficaram em `/tmp`, fora do Git. Uma comparação AST confirmou que
o corpo da referência original permaneceu idêntico, salvo o nome da função.
