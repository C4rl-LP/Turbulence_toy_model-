# Documentação da Suíte de Testes Numéricos

Esta suíte de testes foi desenvolvida para investigar **estocasticidade espontânea**, **convergência temporal**, **efeitos de difusão**, **transições de regime** e **propriedades estatísticas** do modelo de turbulência.

A arquitetura preserva integralmente as interfaces e simuladores originais (`integradores.py` e `campos.py`), organizando os testes de forma independente e reprodutível.

---

## Estrutura de Diretórios

```
Turbulence_toy_model-/
├── integradores.py                  # Simuladores base (RK4, Euler-Maruyama)
├── campos.py                        # Campos de velocidade (podado, correto)
├── funcoes_para_centros.py          # Geometria e centros hierárquicos
├── tests/                           # Suíte de testes numéricos
│   ├── __init__.py
│   ├── utils_estatisticas.py        # Módulo de métricas e histogramas
│   ├── test_convergencia_dt.py      # Teste 1: Convergência temporal dt
│   ├── test_convergencia_N.py       # Teste 2: Escala temporal com N_fields
│   ├── test_estocasticidade.py      # Teste 3: Estocasticidade espontânea (delta_r -> 0)
│   ├── test_estatisticas_temporais.py # Teste 5: Séries temporais (mean, var, sigma², MSD)
│   ├── test_difusao.py              # Teste 6: Análise de MSD(t) e expoente local alpha(t)
│   ├── test_comparacao_simuladores.py # Teste 7: Comparação Determinístico vs Estocástico
│   ├── test_transicao_temporal.py   # Teste 8: Dependência temporal explícita do campo
│   ├── test_distribuicoes.py        # Teste 9: Histogramas 1D/2D, assimetria, curtose e quantis
│   ├── test_robustez_estatistica.py # Teste 10: Robustez e convergência com N_particulas
│   └── run_all_tests.py             # Runner mestre da suíte
└── resultados/                      # Diretório de dados e gráficos gerados
    ├── convergencia_dt/
    ├── convergencia_N/
    ├── estocasticidade/
    ├── estatisticas_temporais/
    ├── difusao/
    ├── comparacao/
    ├── transicao_temporal/
    ├── distribuicoes/
    └── robustez/
```

---

## Descrição Detalhada dos Testes

### 1. Teste de Convergência Temporal (`test_convergencia_dt.py`)
* **O que mede:** A estabilidade e convergência dos observáveis estatísticos do simulador estocástico no limite $dt \to 0$.
* **Parâmetros variados:** Passo de tempo $dt \in [dt_{\text{base}}, dt_{\text{base}}/2, dt_{\text{base}}/4, dt_{\text{base}}/8]$. Mantém fixos $N_{\text{fields}}$, $r_0$, $t_{\text{max}}$, $N_{\text{particulas}}$.
* **Dados salvos:** `resultados/convergencia_dt/convergencia_dt_dados.npz` (médias, variâncias, covariâncias, $\sigma^2$, distribuições).
* **Gráficos produzidos:** `estatisticas_vs_dt.png` (painel $2 \times 2$) e `distribuicao_2d_dt_*.png` (histogramas 2D para cada $dt$).

---

### 2. Teste da Escala Temporal com $N$ (`test_convergencia_N.py`)
* **O que mede:** A escala de refinamento temporal $dt_N = dt_1 / \lambda^{N-1}$ e seu impacto no tempo de computação e convergência espacial conforme o número de níveis cresce.
* **Parâmetros variados:** Nível $N_{\text{fields}}$ de $N_{\text{min}}$ a $N_{\text{max}}$.
* **Dados salvos:** `convergencia_N_dados.npz` e tabela `convergencia_N_resumo.csv`.
* **Gráficos produzidos:** `dt_N_vs_N.png`, `num_passos_vs_N.png`, `sigma2_vs_N.png`, `variancia_x_y_vs_N.png` e `custo_computacional_vs_N.png`.

---

### 3. Teste de Estocasticidade Espontânea (`test_estocasticidade.py`)
* **O que mede:** A sensibilidade no limite de perturbação nula ($\delta r \to 0$) para investigar a existência de dispersão estatística não trivial induzida pela dinâmica.
* **Parâmetros variados:** Amplitude de perturbação inicial $\delta r \in [10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}]$ para vários $N_{\text{fields}}$.
* **Dados salvos:** `estocasticidade_N_*_dados.npz` para cada nível.
* **Gráficos produzidos:** `dispersao_vs_delta_r.png`, `variancia_vs_delta_r.png`, `sigma2_vs_delta_r.png` e espalhamentos `distribuicao_delta_r_*.png`.

---

### 4. Separação de Ruído Inicial vs Ruído Dinâmico (`test_comparacao_simuladores.py`)
* **O que mede:** A diferença de comportamento entre um sistema determinístico pertubado (A), estocástico puro (B) e estocástico perturbado (C).
* **Parâmetros variados:** Tipo de dinâmica (determinística vs estocástica) e presença de perturbação inicial.
* **Dados salvos:** `comparacao_ruidos.npz` ou `comparacao_simuladores_dados.npz`.
* **Gráficos produzidos:** `trajetorias_comparadas.png`, `distribuicao_final_comparada.png`, `medias_comparadas.png`, `variancias_comparadas.png`, `sigma2_comparado.png` e `msd_comparado.png`.

---

### 5. Estatísticas ao Longo do Tempo (`test_estatisticas_temporais.py`)
* **O que mede:** A evolução dinâmica completa dos momentos estatísticos $\langle x(t) \rangle$, $\langle y(t) \rangle$, $\text{Var}[x](t)$, $\text{Var}[y](t)$, $\text{Cov}[x,y](t)$, $\sigma^2(t)$ e $\text{MSD}(t)$.
* **Parâmetros variados:** Instantes de tempo $t_k \in [0, t_{\text{max}}]$ para diferentes $N_{\text{fields}}$.
* **Dados salvos:** `series_temporais_N_*.npz`.
* **Gráficos produzidos:** Curvas temporais individuais e comparativas para cada estatística (`media_x_vs_t.png`, `sigma2_vs_t.png`, `msd_vs_t.png`, etc.).

---

### 6. Teste de Difusão (`test_difusao.py`)
* **O que mede:** O crescimento do Deslocamento Quadrático Médio $\text{MSD}(t) = \langle |r(t)-r(0)|^2 \rangle$ e a estimativa local do expoente difusivo:
  $$\alpha(t) = \frac{d \log(\text{MSD})}{d \log(t)}$$
* **Parâmetros variados:** Tempo $t$ em escala logarítmica e nível $N_{\text{fields}}$.
* **Dados salvos:** `difusao_N_*_dados.npz`.
* **Gráficos produzidos:** `msd_loglog.png` (gráfico Log-Log de MSD vs $t$ com reta de referência difusiva $\text{MSD}\sim t$) e `expoente_difusao_alpha.png` ($\alpha(t)$ vs $t$).

---

### 7. Teste de Dependência Temporal do Campo (`test_transicao_temporal.py`)
* **O que mede:** A resposta do sistema sob campo vetorial dependente explicitamente do tempo (`temporal=True`) ao varrer parâmetros de controle.
* **Parâmetros variados:** Parâmetro de controle (ex: $c$, $\alpha$ ou frequência) e nível $N_{\text{fields}}$.
* **Dados salvos:** `transicao_temporal_dados.npz`.
* **Gráficos produzidos:** `sigma2_vs_parametro.png`, `msd_vs_parametro.png` e `distancia_vs_parametro.png`.

---

### 8. Distribuições Completas (`test_distribuicoes.py`)
* **O que mede:** A forma completa da densidade de probabilidade $P(x,y)$, assimetria (skewness), curtose (excess kurtosis) e quantis (5%, 25%, 50%, 75%, 95%).
* **Parâmetros variados:** Nível $N_{\text{fields}}$, passo de tempo $dt$, condição inicial e tipo de simulador.
* **Dados salvos:** `distribuicoes_dados.npz`.
* **Gráficos produzidos:** Painéis de histogramas 1D (`*_hist1d.png`) e histogramas 2D (`*_hist2d.png`).

---

### 9. Teste de Robustez Estatística (`test_robustez_estatistica.py`)
* **O que mede:** A taxa de convergência das estimativas dos momentos estatísticos em função do tamanho da amostra (número de realizações).
* **Parâmetros variados:** Número de partículas / realizações $N_{\text{particulas}} \in [100, 300, 1000, 3000, \dots]$.
* **Dados salvos:** `robustez_estatistica_dados.npz`.
* **Gráficos produzidos:** `convergencia_robustez.png` (média, variância, $\sigma^2$ e MSD vs $N_{\text{particulas}}$).

---

## Evidências Relevantes para Investigar Estocasticidade Espontânea

Para avaliar se o modelo apresenta indícios de **estocasticidade espontânea** (isto é, persistência de incerteza estatística no limite de perturbação/ruído infinitesimal e número infinito de modos):

1. **Limite $\delta r \to 0$ no Teste 3:** Se a variância final $\sigma^2(t_{\text{max}})$ tender a um valor estritamente positivo não nulo quando a perturbação inicial $\delta r \to 0$, isso constitui evidência empírica favorável à estocasticidade espontânea.
2. **Dependência com $N_{\text{fields}}$ no Teste 2 e 5:** Acompanhar se a taxa de crescimento da dispersão $\sigma^2(t)$ satura ou diverge à medida que $N_{\text{fields}} \to \infty$.
3. **Expoente Difusivo $\alpha(t)$ no Teste 6:** Valores de $\alpha > 1$ (superdifusão / regime balístico ou hiperdifusivo) indicam advecção caótica/turbulenta acelerada pelos centros hierárquicos.
4. **Forma não-gaussiana da Distribuição no Teste 9:** Curtose elevada (caudas pesadas) e assimetria na distribuição indicam eventos raros e intermitência característicos de estocasticidade dinâmica não trivial.

---

## Como Executar a Suíte

Para executar todos os testes da suíte (ou algum teste específico):

```bash
# Executar todos os testes com 300 realizações padrão
python3 tests/run_all_tests.py --teste todos --N_particulas 300

# Executar apenas o teste de difusão
python3 tests/run_all_tests.py --teste difusao --N_particulas 500

# Executar apenas o teste de estocasticidade
python3 tests/run_all_tests.py --teste estocasticidade --N_particulas 1000
```
