# Desempenho no teste de estocasticidade espontânea — 30/10/2026

Houve ganho tanto na simulação utilizada pelo teste quanto no driver completo.
O ganho total é menor porque estatísticas, gráficos e arquivos permanecem com
seu custo original. Nenhum código de produção foi alterado nesta medição.

## O que foi executado

O teste existente `tests/test_estocasticidade.py` chama `it.simular_particulas`,
com integração determinística RK4 e aleatoriedade apenas na nuvem inicial.
Não é o simulador Euler–Maruyama com ruído ao longo da trajetória.

Comparação: `campo_total_podado_referencia` versus `campo_total_podado`,
na mesma RTX A4500, float64, mesmas nuvens iniciais verificadas exatamente.
A semente CuPy 812 foi redefinida antes de cada chamada; a semente NumPy isolada
usada no teste original não controla a geração da nuvem CuPy.
Mantidos os parâmetros originais: r0=[0.3,0.2333], dimensão inicial=0.1*R(n),
dt=0.01*T(n), quatro passos RK4. Ao variar nível, dt e tempo final seguem a regra
do teste: t_max=4*dt. Antes/depois usam rigorosamente os mesmos valores em cada linha.
Casos pequenos primeiro, sem simulações científicas longas.

Medianas de três repetições após aquecimento. A simulação inclui redefinição da
semente, geração da nuvem e integração; exclui estatísticas, gráficos, gravação
e cópia final da trajetória para CPU. CUDA Events e relógio de parede com
sincronização ao fim; a tabela mostra tempo de parede. Os Events estão no JSON.
Não se usa diretamente o `tempo_execucao` impresso pelo driver, pois ele termina
a medição antes de sincronizar explicitamente a GPU e exclui gráficos/arquivos.

## Simulação usada pelo teste

Tempos em ms; ganho = CuPy original / CuPy otimizado.

| Partículas | Níveis | Campo | Original | Otimizado | Ganho |
|---:|---:|---|---:|---:|---:|
| 8 | 2 | estático | 58.055 | 2.005 | 29.0× |
| 8 | 2 | temporal | 61.185 | 2.001 | 30.6× |
| 32 | 3 | estático | 374.438 | 1.998 | 187.4× |
| 32 | 3 | temporal | 379.120 | 2.002 | 189.4× |
| 128 | 3 | estático | 1451.381 | 2.029 | 715.4× |
| 128 | 3 | temporal | 1451.740 | 2.032 | 714.5× |
| 128 | 5 | estático | 2856.989 | 2.132 | 1340.2× |
| 128 | 5 | temporal | 2845.006 | 2.078 | 1369.3× |
| 512 | 3 | estático | 5776.926 | 1.960 | 2948.2× |
| 512 | 3 | temporal | 5779.692 | 2.053 | 2815.7× |

## Teste completo, incluindo gráficos e arquivos

O próprio `executar_teste_estocasticidade` foi executado sem modificar suas funções,
com 32 partículas, nível 3, quatro passos e ambos os campos estático/temporal.
Cada repetição escreveu em uma pasta nova em /tmp. Uma rodada inicial foi excluída
para aquecimento de kernels, estatísticas e Matplotlib. Nas três rodadas medidas,
a ordem original/otimizado foi alternada. Medição externa de parede, sincronizada,
incluindo nuvem, RK4, estatísticas, transferências, compressão NPZ, todos os gráficos
e arquivos do driver. A leitura posterior dos NPZ para comparar resultados não
entra no tempo medido.

| Campo | Original (s) | Otimizado (s) | Ganho | Redução de tempo |
|---|---:|---:|---:|---:|
| estático | 1.627 | 1.245 | 1.31× | 23.5% |
| temporal | 1.703 | 1.282 | 1.33× | 24.7% |

A diferença entre os ganhos das duas tabelas é consistente com o custo remanescente
de gráficos e arquivos nesse caso curto. Não foi feito perfil individual desses
componentes, portanto não se atribuem porcentagens exatas a cada um. O ganho no
teste completo com mais partículas/passos não foi medido nem extrapolado.

## Correção, memória e limites

Todas as trajetórias passaram em rtol=1e-10, atol=1e-12. Na simulação, maior erro
absoluto 5,56e-17. No teste completo de 32 partículas, trajetórias, sigma²(t) e média
final coincidiram; erro máximo de trajetórias e sigma² foi zero para ambas as formas
do campo. Isso valida esta execução curta, sem conclusões sobre convergência física
ou o fenômeno de estocasticidade espontânea.

O maior pool reservado amostrado após as simulações foi 1.277.440 bytes (~1,22 MiB).
Não é medição de pico transitório nem VRAM do contexto. A medição do teste completo
não incluiu amostragem específica de pico de memória. Todos os gráficos e dados
foram gravados em diretório temporário isolado, identificado no JSON; nenhum arquivo
em `tests/resultados` foi alterado pelo benchmark.

Aquecimento/compilação foram excluídos; imports e inicialização do processo também.
Três repetições não estimam incerteza estatística e o desktop compartilha a GPU.
O caso completo foi limitado a 32 partículas/4 passos para manter o teste curto.

## Arquivos e reprodução

Criados somente o benchmark, este relatório e seu JSON de resultados:

- `tests/benchmark_estocasticidade_gpu.py`
- `tests/benchmark_estocasticidade_resultados.json`
- `tests/RELATORIO_DESEMPENHO_ESTOCASTICIDADE.md`

```bash
/tmp/turbulence-gpu-diag-venv/bin/python tests/benchmark_estocasticidade_gpu.py --output /tmp/novo-benchmark-estocasticidade.json
python -m py_compile tests/benchmark_estocasticidade_gpu.py
```

Ambos os comandos passaram. O benchmark recusa sobrescrever o JSON solicitado e
cria um diretório temporário novo para suas figuras e dados a cada execução.
Não houve commit, push ou merge.
