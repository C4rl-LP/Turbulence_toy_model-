# Testes numéricos e visualização dos resultados

As simulações salvam **somente dados e metadados**, sem gerar figuras ou abrir
janelas. Os gráficos são reconstruídos depois, em outro processo, usando
apenas **NumPy e Matplotlib**, sem CuPy, CUDA ou importação dos simuladores.
As assinaturas e os valores retornados pelos drivers foram mantidos.

## Executar uma simulação

A geração numérica continua exigindo o ambiente CuPy/CUDA do projeto.
O runner aceita `--output` para escolher uma pasta nova e `--niveis` para
limitar a resolução. O caminho é relativo ao diretório de trabalho; não há
mudança automática para `tests/`. Evite reutilizar uma pasta científica:
os drivers mantêm o comportamento anterior de gravar seus nomes fixos.

```bash
# Caso pequeno: duas escalas, oito partículas, campo temporal.
python tests/run_all_tests.py --teste estocasticidade \
    --niveis 1 2 --N_particulas 8 --t_max 0.03 --campo temporal \
    --output resultados/nova_execucao

# Varredura alpha_cut x c somente no nível 2.
python tests/run_all_tests.py --teste estocasticidade_parametros \
    --niveis 2 --N_particulas 8 --t_max 0.03 --campo temporal \
    --output resultados/nova_varredura

# Outros testes disponíveis, agora encaminhados pelo runner.
python tests/run_all_tests.py --teste convergencia_dt \
    --niveis 1 --N_particulas 8 --t_max 0.03 --output resultados/nova_convergencia
python tests/run_all_tests.py --teste difusao \
    --niveis 1 2 --N_particulas 8 --t_max 0.03 --output resultados/nova_difusao
```

`--teste` aceita `estocasticidade`, `estocasticidade_parametros`,
`convergencia_dt`, `convergencia_N`, `estatisticas_temporais`, `difusao`,
`comparacao`, `transicao_temporal`, `distribuicoes`, `robustez` e `todos`.
`todos` executa todos os drivers e pode ser custoso. Sem `--niveis`, são
mantidas as grades padrão das funções; nas varreduras paramétricas do runner,
N=6,7,8 temporais e N=0,...,7 estáticos. `--campo` aceita `estatico`,
`temporal` ou `ambos`; aplica-se às funções que oferecem esse argumento.
`transicao_temporal` usa sempre o campo temporal; `distribuicoes` usa o campo
estático, como antes. No teste de robustez, a quantidade variada é
`lista_N_particulas`, configurável na chamada Python, e não `--N_particulas`.

Para controlar a grade de parâmetros sem mudar os padrões do driver:

```python
from tests.test_estocasticidade_parametros import executar_teste_estocasticidade_parametros

executar_teste_estocasticidade_parametros(
    N_fields=2, N_particulas=8, t_max=0.03,
    lista_alpha=[1.8, 2.0], lista_c=[0.2, 0.6], temporal=True,
    output_dir="resultados/varredura_pequena", seed=42,
)
```

Essa chamada não desenha imagens. As funções `testar_estabilidade_estatico`
e `testar_estabilidade_temporal`, em `integradores.py`, também salvam somente
dados e continuam com suas assinaturas anteriores. Apenas essas rotinas de
teste foram alteradas nesse arquivo; RK4, Euler–Maruyama e os simuladores
mantêm o mesmo código de cálculo.

## Organização dos arquivos

A hierarquia e os nomes existentes foram preservados. Cada novo NPZ recebe
um JSON adjacente de mesmo nome, sem uma migração dos dados antigos:

```text
resultados/nova_varredura/estocasticidade_parametros_temporal/em_nivel_2/
├── resumo_varredura.npz
├── resumo_varredura.json
├── informacoes/Informações.txt
└── alpha_1.800_c_0.200/
    ├── dados_ponto.npz
    └── dados_ponto.json

figuras/nova_varredura/em_nivel_2/
├── mapa_calor_sigma2_final.png
└── alpha_1.800_c_0.200/
    └── distribuicao_final_hist2d.png
```

Os NPZ são comprimidos. Arrays numéricos simples mantêm suas chaves.
Dicionários antes armazenados como objetos passam a usar chaves hierárquicas,
por exemplo `stats_det__sigma2` e `relatorio__N_fields_1__rs_final`.
Assim, todos os arrays novos podem ser abertos com `allow_pickle=False`.
`tests.plot_resultados.Arquivo` fornece a leitura uniforme de ambos os formatos.
Os retornos Python das funções não mudam.

Os JSON registram o driver, os parâmetros físicos padrão, o callback de campo,
a configuração numérica, o tipo de campo, as sementes solicitadas e, quando
há vetor temporal no arquivo, o passo e tempo final efetivos. Uma função de
campo personalizada é identificada pelo nome; seus parâmetros internos não
são presumidos a partir das constantes padrão.

**Sementes:** os drivers existentes inicializam principalmente o RNG NumPy,
mas os simuladores usam RNG CuPy. Essa separação foi preservada para não
mudar as simulações. O JSON distingue a semente NumPy solicitada de uma
semente CuPy efetiva não registrada. Ele não promete reprodutibilidade
bit a bit a partir da semente NumPy. As trajetórias salvas continuam sendo
suficientes para redesenhar as figuras sem novos sorteios.

## Gerar as figuras posteriormente

```bash
# Reconstruir todas as figuras dos resultados antigos, em uma pasta separada.
python tests/plot_resultados.py \
    --input tests/resultados/estocasticidade_parametros_temporal \
    --output figuras/estocasticidade_parametros_temporal --all

# Apenas mapas: não carrega as trajetórias dos arquivos individuais.
python tests/plot_resultados.py \
    --input tests/resultados/estocasticidade_parametros_temporal \
    --output figuras/mapas_parametros --tipo mapas

# Dispersão temporal, cortes e mapas; sem trajetórias e histogramas.
python tests/plot_resultados.py \
    --input tests/resultados/estocasticidade_parametros_temporal \
    --output figuras/dispersao_parametros --tipo dispersao

# Histogramas e distribuição inicial/final de apenas um experimento antigo.
python tests/plot_resultados.py \
    --input tests/resultados/estocasticidade_parametros_temporal/em_nivel_8/alpha_1.800_c_0.200/dados_ponto.npz \
    --output figuras/ponto_n8 --tipo distribuicoes

# Escolher mais de uma categoria.
python tests/plot_resultados.py --input resultados/nova_execucao \
    --output figuras/selecao --tipo dispersao --tipo estatisticas
```

`--input` aceita um arquivo NPZ ou diretório, pesquisado recursivamente.
`--all` é o padrão. `--tipo` pode ser repetido e aceita:

| Categoria | Conteúdo |
|---|---|
| `dispersao` | sigma² temporal/final, dispersão média, cortes e mapas paramétricos |
| `mapas` | somente os mapas paramétricos, incluindo log10(sigma²/L²) |
| `trajetorias` | caminhos individuais e comparações de trajetórias |
| `distribuicoes` | histogramas, scatters inicial/final e distribuições comparadas |
| `estatisticas` | médias, variâncias, covariâncias e painéis estatísticos |
| `difusao` | MSD e expoente local de difusão |
| `convergencia` | passo, número de passos e custo computacional em função do nível |

Painéis que agrupam várias estatísticas permanecem inteiros na categoria
`estatisticas`. Uma categoria não cria gráficos para os quais aquele arquivo
não contém a informação correspondente. Por exemplo, a série sigma² de um
ponto paramétrico é desenhada no gráfico comparativo do resumo da varredura;
para esse gráfico mantenha o resumo e as pastas individuais juntos.

Sem `--output`, a saída vai para `imagens/` dentro da entrada (ou ao lado do
arquivo). A ferramenta rejeita a pasta original como saída, seus ancestrais
ou uma saída que contenha NPZ. Não grava NPZ/JSON, não inicializa geradores
aleatórios e não importa simuladores. As figuras são geradas com backend
`Agg`, sem janelas. Arquivos são processados um a um; selecionar somente
estatísticas evita carregar as trajetórias grandes. `--all` pode redesenhar
muitas trajetórias e imagens, portanto selecione categorias ou um arquivo
quando esse for o objetivo.

## Dados necessários e compatibilidade

| Experimento | Dados numéricos usados nas figuras | Trajetórias armazenadas |
|---|---|---|
| Estocasticidade por nível | `t`, `rs`, momentos, sigma² e resumo global | completas: o teste desenha todas |
| Estocasticidade paramétrica | `rs_ref`, `t`, `sigma2_t`, médias finais e matrizes do resumo | completas da primeira amostra, como antes |
| Convergência dt | momentos finais e posições finais por dt | somente posições finais |
| Convergência N | dt_N, passos, custo e momentos finais | nenhuma |
| Estatísticas temporais | `t`, médias, variâncias, covariância, sigma² e MSD | nenhuma |
| Difusão | `t`, MSD e alpha_t | nenhuma |
| Comparação de simuladores | dois tempos, trajetórias e estatísticas | mantidas para os gráficos e análises existentes |
| Distribuições | posições finais e estatísticas de cada caso | somente posições finais |
| Robustez | momentos finais versus número de partículas | nenhuma |
| Transição temporal | matrizes de observáveis finais versus parâmetro nominal | nenhuma |
| Estabilidade | `t`, `rs`, r0, nível e identificação do simulador | novas trajetórias, necessárias aos gráficos antigos |

Os formatos antigos numéricos de estocasticidade e varreduras são usados
diretamente, sem conversão nem alteração dos originais. Os gráficos preservam
estilos, eixos, escalas, binagens e nomes dos arquivos anteriores. A organização
relativa é reproduzida dentro da saída. A comparação de distribuição inicial/final
continua usando os valores salvos; definições antigas ambíguas não são reinterpretadas.

Para NPZ antigos com dicionários/arrays de objetos, de origem confiável:

```bash
python tests/plot_resultados.py --input CAMINHO_ANTIGO \
    --output figuras/legado --all --permitir-pickle
```

Essa opção autoriza a desserialização pickle apenas durante a leitura;
nenhum arquivo é migrado. Os pickles usuais de arrays CuPy são reconstruídos
como NumPy, sem importar CuPy. Não habilite essa opção para arquivos de
origem desconhecida. Arquivos novos não precisam dela.

Limitações identificadas:

- **Estabilidade antiga em `integradores.py`:** eram gravados somente PNG e,
  em um dos testes, momentos no TXT. Trajetórias e posições finais não foram
  salvas; seus gráficos não podem ser reconstruídos apenas desses momentos.
  As novas execuções salvam `estabilidade_nivel_N_dados.npz` para resolver isso.
- **Suíte monolítica legada:** o NPZ `estocasticidade_espontanea_N_*.npz`
  não registrava r0. Histogramas/curvas de momentos continuam disponíveis,
  mas o scatter com a marca de r0 exige esse parâmetro; sem JSON ele é
  identificado como incompleto e não é inventado. `comparacao_ruidos.npz`
  também é reconhecido. A geração de dados dessa suíte foi separada dos plots;
  seus outros comportamentos legados não foram refatorados.
- **Comparação antiga de simuladores:** N_fields não estava no NPZ. Sem
  metadados novos, o título indica `não registrado`; não se presume o padrão 3.
- **Séries antigas de difusão:** a ordem da lista de níveis não era salva.
  A ferramenta ordena numericamente na ausência do JSON; a curva de referência
  MSD~t usa o último nível dessa ordem. Os novos JSON preservam a lista original.
- **Varredura temporal paramétrica antiga:** é necessário manter os arquivos
  individuais para reconstruir o gráfico temporal selecionado; a matriz final
  sozinha só reconstrói mapas e cortes. Dados incompletos são sinalizados.
- Os nomes antigos `delta_r` têm significados diferentes entre resumos e
  arquivos individuais. Eles são mantidos para compatibilidade. O normalizador
  de `norm_sigma2` é o diâmetro efetivo L ao quadrado, e não o raio ao quadrado.
- A inconsistência já identificada em N=5 entre resumo e um arquivo individual
  permanece: mapas/cortes leem o resumo salvo e séries leem os pontos salvos.
  A ferramenta não corrige ou combina silenciosamente resultados científicos.
- `test_transicao_temporal.py` já não aplica o valor nominal varrido ao callback
  de campo. Essa limitação científica preexistente foi preservada; seus gráficos
  representam os valores armazenados e não comprovam resposta física ao parâmetro.

## Verificação e armazenamento

```bash
python -m unittest tests.test_fluxo_resultados -v
python -m py_compile tests/*.py integradores.py suite_testes_numericos.py
```

Os testes CPU verificam leitura antiga/nova, metadados, matrizes representadas,
seleção de séries, preservação dos arquivos e importação sem simuladores/GPU.
A validação da reorganização também executou dez drivers com casos pequenos,
comparando os valores aos drivers anteriores com a mesma semente CuPy.
Os arrays científicos coincidiram exatamente; somente tempos de execução
não foram comparados. Nenhum driver numérico criou figura. Os gráficos dos
casos pequenos foram então reconstruídos separadamente.

A documentação técnica com inventário, comandos e verificações detalhadas
está em [SEPARACAO_DADOS_VISUALIZACAO.md](tests/SEPARACAO_DADOS_VISUALIZACAO.md).

No acervo `estocasticidade_parametros_temporal`, as 945 imagens ocupam
**126,48 MiB**; os NPZ ocupam aproximadamente **5,62 GiB**. Uma nova execução
equivalente sem imagens evita cerca de **2,15%** do volume total, antes do
pequeno acréscimo dos JSON. O principal volume desse experimento continua
sendo o das trajetórias necessárias aos gráficos existentes. A economia de
Git depende dos arquivos já rastreados e da compressão; não foi extrapolada.

O `.gitignore` passa a ignorar imagens novas em `resultados/`,
`tests/resultados/` e `figuras/`. Isso não retira do Git nenhum arquivo já
rastreado. Nenhuma imagem antiga, resultado, histórico ou commit foi removido.

## Desempenho: NumPy versus CuPy otimizado

O benchmark compara o campo podado de referência em NumPy com o kernel CuPy
otimizado usado pelo teste de estocasticidade, usando o mesmo RK4, `float64`
e uma nuvem inicial idêntica nos dois backends. Requer um ambiente com NumPy,
CuPy, Matplotlib e GPU CUDA funcional.

```bash
# Caso pequeno primeiro; cada caso roda com campo estático e temporal.
python tests/benchmark_estocasticidade_gpu.py --numpy-vs-cupy \
    --case 8,3,4 --case 128,3,4 --repeats 3 --seed 812 \
    --output /tmp/numpy-cupy-curto.json

# Grade padrão: 1 a 8192 partículas, incluindo variações de níveis e passos.
python tests/benchmark_estocasticidade_gpu.py --numpy-vs-cupy \
    --output /tmp/numpy-cupy-completo.json
```

`--case` recebe `partículas,níveis,passos` e pode ser repetido. O JSON registra
medianas após aquecimento, tempo de parede sincronizado, CUDA Events, ganho
CPU/GPU e erros dos campos, trajetórias e dispersão `sigma²(t)`. Há tempos
separados para integração com dados na GPU e para integração incluindo cópias
CPU→GPU→CPU. A geração da nuvem, as estatísticas, os gráficos e a gravação dos
resultados científicos ficam fora da medição. O ganho compara implementações
distintas: o algoritmo NumPy de referência percorre partículas em Python.

O benchmark recusa sobrescrever o JSON e não grava em `tests/resultados`.
Resultados medidos e limites estão em
[RELATORIO_NUMPY_VS_CUPY.md](tests/RELATORIO_NUMPY_VS_CUPY.md).
