# Separação da geração numérica e da visualização

## Organização implementada

Os drivers numéricos mantêm suas assinaturas, parâmetros, chamadas de simulação,
fórmulas estatísticas e retornos. Foram retirados os blocos de desenho e as
chamadas automáticas ao gerador de histogramas. A gravação usa NPZ comprimido e
JSON adjacente. Não há uma reorganização ou conversão dos arquivos científicos
antigos.

As responsabilidades novas são:

- `tests/dados_resultados.py`: conversão necessária de arrays CuPy para CPU,
  gravação comprimida sem objetos pickle e metadados pequenos. Não altera RNG.
- `tests/graficos_resultados.py`: funções de desenho transferidas dos drivers,
  preservando estilos, binagens, legendas, escalas e nomes das imagens.
- `tests/plot_resultados.py`: descoberta recursiva dos NPZ, leitura sob demanda,
  interpretação dos esquemas antigo/novo e seleção dos gráficos. Não importa
  campos, integradores, utilitários estatísticos de simulação ou CuPy.
- `tests/test_fluxo_resultados.py`: testes CPU do fluxo separado.

O JSON registra `formato=2`. Para dicionários, `pai__filho` representa a estrutura
de chaves do NPZ; arrays simples mantêm seus nomes. Isso elimina os novos
arquivos de objetos antes usados em comparação, distribuições e transição.
O leitor aceita ambos os esquemas. Os formatos antigos com pickle exigem opção
explícita, sem migração; os construtores usuais de arrays CuPy são mapeados
para NumPy, sem carregar o backend GPU.

## Arquivos modificados

Drivers numéricos:

- `tests/test_estocasticidade_parametros.py`
- `tests/test_estocasticidade.py`
- `tests/test_convergencia_dt.py`
- `tests/test_convergencia_N.py`
- `tests/test_estatisticas_temporais.py`
- `tests/test_difusao.py`
- `tests/test_comparacao_simuladores.py`
- `tests/test_distribuicoes.py`
- `tests/test_transicao_temporal.py`
- `tests/test_robustez_estatistica.py`

Outros arquivos:

- `integradores.py`: somente as rotinas `testar_estabilidade_estatico` e
  `testar_estabilidade_temporal` passaram a salvar trajetórias, sem desenhar.
  Foi retirado o import Matplotlib que deixou de ser necessário. Os quatro
  integradores/simuladores mantêm AST idêntica ao estado anterior.
- `suite_testes_numericos.py`: a suíte monolítica legada também deixa de
  desenhar, usando a mesma persistência. Seus cálculos não foram refatorados.
- `tests/utils_estatisticas.py`: o helper público de histogramas continua
  disponível como encaminhamento explícito ao módulo de visualização.
- `tests/run_all_tests.py`: novos `--output`, `--niveis` e `--campo`; todas as
  opções de teste passam a ser encaminhadas aos respectivos drivers. Antes,
  a versão examinada despachava somente os dois testes de estocasticidade.
- `tests/benchmark_estocasticidade_gpu.py`: comentários atualizados para a
  medição do driver numérico completo, que agora não inclui figuras. Relatórios
  históricos de desempenho não foram reescritos.
- `.gitignore`: ignora figuras novas nas raízes de resultados e em `figuras/`,
  sem retirar qualquer arquivo já rastreado.
- `README_TESTES.md`: comandos, dados por gráfico, organização e compatibilidade.

Arquivos criados: os quatro módulos listados na organização e este documento.
Não foram alterados `campos.py`, `campo_gpu.py` ou `funcoes_para_centros.py`.

## Dados de cada família de gráficos

| Família | Informações existentes usadas | Informação adicional necessária nas novas execuções |
|---|---|---|
| Varredura alpha/c | posições da primeira amostra, t, sigma²(t), médias, matrizes finais e normalização | configuração, precisão de identificação do campo e sementes solicitadas no JSON |
| Estocasticidade por nível | trajetórias, momentos, sigma²(t), distribuição final e resumo global | metadados; r0 pode ser recuperado da partícula central antiga |
| Convergência dt | momentos e posições finais para cada dt | JSON; arrays de objetos numéricos agora são arrays float |
| Convergência N | dt_N, número de passos, tempos medidos e momentos finais | JSON; não exige trajetórias |
| Séries temporais | t e sete séries de momentos | ordem dos níveis e demais parâmetros no JSON |
| Difusão | t, MSD e expoente local alpha_t | ordem original dos níveis e t_max no JSON |
| Comparação | dois conjuntos de trajetórias e estatísticas | N_fields no JSON, antes ausente do arquivo |
| Distribuições | posições finais e estatísticas dos casos | JSON; não se adicionam trajetórias completas |
| Robustez | momentos finais por tamanho de amostra | JSON; não exige trajetórias |
| Transição | observáveis finais por nível e parâmetro nominal | JSON com identificação temporal explícita |
| Estabilidade | os arquivos antigos continham somente imagens e momentos em TXT | t, trajetórias, r0, nível e identificação do simulador em novo NPZ |

Não foram acrescentadas trajetórias aos testes que usavam somente estatísticas
agregadas ou posições finais. Os testes que já desenhavam trajetórias completas
mantêm essa informação. As novas rotinas de estabilidade precisam gravá-la;
nesse caso, o volume de dados aumenta em relação à antiga saída apenas gráfica.

## Validação realizada

Todos os experimentos de validação foram gravados em pastas novas em `/tmp`.
Nenhuma execução escreveu nos resultados históricos do repositório.

1. **Antes/depois dos dez drivers:** comparação com cópias do código anterior,
   preservadas antes das edições. Casos com oito partículas (robustez: 8 e 12),
   tempo 0,03, níveis 1–2 e grades reduzidas. O RNG CuPy foi reinicializado com
   a mesma semente 917 antes de cada versão. Todos os arrays científicos
   coincidiram exatamente, inclusive os dicionários antigos lidos como dados.
   O tempo computacional medido foi excluído da comparação. A versão nova
   foi executada com `plt.figure`/`plt.subplots` configurados para falhar se
   chamados; nenhum driver tentou desenhar. Todos os novos NPZ foram abertos
   com `allow_pickle=False`.
2. **Visualização dos casos pequenos:** foram reconstruídas 67 imagens ao todo
   nas dez famílias, somente a partir dos dados novos. As rotinas não receberam
   funções de simulação.
3. **Estabilidade:** três casos com quatro partículas e N=1, incluindo os
   caminhos determinístico estático, determinístico temporal e Euler–Maruyama.
   A etapa numérica gerou zero PNG; os dados reconstruíram cinco imagens.
   As AST das funções `solve_RK4`, `solve_euler_maruyama`, `simular_particulas`
   e `simular_particulas_ruidosa` foram comparadas e permaneceram idênticas.
4. **Dados científicos antigos:** os 18 mapas das nove varreduras temporais,
   três imagens de distribuição do ponto N=8, alpha=1,8, c=0,2, e 11 imagens
   de dispersão do teste de estocasticidade temporal foram reconstruídos.
   Os **32 arquivos PNG coincidiram pixel a pixel**, incluindo eixos, escalas,
   legendas e tamanhos das imagens. A comparação foi feita nos ambientes
   disponíveis; outras versões de Matplotlib/fontes podem produzir diferenças
   de rasterização sem alterar os valores representados.
5. **Legado com objetos CuPy:** os arquivos antigos de distribuições e de
   comparação produzidos pelos casos pequenos foram lidos pelo Python padrão,
   sem CuPy instalado, usando `--permitir-pickle`; os gráficos foram gerados.
6. **Runner:** convergência dt, robustez e varredura temporal N=1 foram
   executadas por CLI em uma pasta nova. Não foi criado PNG. A opção de robustez
   usa sua lista própria de tamanhos de amostra, conforme sua interface.
7. **Integridade:** os tamanhos e datas de modificação dos 1925 arquivos
   científicos preexistentes foram conferidos contra um manifesto anterior
   à reorganização. Nenhum foi alterado ou excluído.

Os sete testes CPU permanentes são executados por:

```bash
python -m unittest tests.test_fluxo_resultados -v
python -m py_compile tests/*.py integradores.py suite_testes_numericos.py
```

Comandos de reconstrução efetivamente usados, entre outros:

```bash
python tests/plot_resultados.py \
  --input tests/resultados/estocasticidade_parametros_temporal \
  --output /tmp/turbulence-plot-separado/todos_mapas --tipo mapas
python tests/plot_resultados.py \
  --input tests/resultados/estocasticidade_temporal \
  --output /tmp/turbulence-plot-separado/estocasticidade --tipo dispersao
```

O ambiente GPU existente `/tmp/turbulence-gpu-diag-venv/bin/python` foi usado
nas simulações pequenas, com `PYTHONUTF8=1`. O Python padrão sem CuPy foi usado
nas reconstruções e nos testes CPU. Não foram instaladas dependências.

## Limitações de compatibilidade e interpretação

- Trajetórias e histogramas de estabilidade antigos não são recuperáveis de
  momentos/TXT ou de PNG. Os novos NPZ resolvem a preservação futura, sem
  executar retroativamente esses experimentos.
- O scatter do teste 3 da suíte monolítica precisa de r0 para reconstruir a
  marca central. Sem JSON antigo, essa informação está ausente; a ferramenta
  avisa em vez de presumir o argumento padrão.
- N_fields ausente na comparação antiga é indicado como não registrado no
  título; a parte numérica da visualização permanece disponível.
- O gráfico temporal paramétrico precisa dos arquivos individuais, e não
  apenas do resumo final. A ordem original de curvas de difusão exige os
  novos metadados para reprodução exata em varreduras não ordenadas.
- A semente NumPy não é a semente CuPy. O estado externo CuPy não foi
  inventado ou modificado; esse limite é indicado nos novos JSON.
- As definições antigas divergentes de delta_r e a inconsistência existente
  em N=5 foram preservadas. Os gráficos leem os mesmos campos que antes.
- O teste de transição já ignorava o valor nominal no campo, e `lista_dt` do
  teste de distribuições já não era percorrida. Essas limitações científicas
  não foram corrigidas nesta reorganização para não mudar as simulações.

## Armazenamento

Nas varreduras temporais paramétricas atuais, as 945 imagens ocupam
132624847 bytes, ou 126,48 MiB. Os 234 NPZ ocupam aproximadamente 5,62 GiB.
Uma execução de mesmo tamanho sem essas imagens evita cerca de 2,15% do
volume total, descontado o pequeno acréscimo dos JSON. Isso não representa
uma estimativa de redução do histórico Git, que não foi modificado.

Nenhuma imagem antiga ou arquivo rastreado foi removido. Não houve commit,
push, merge ou alteração do histórico.
