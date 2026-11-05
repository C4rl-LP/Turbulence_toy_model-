# Toy model de fluido turbulento 

## Motivação 

Esse projeto foi desenvolvido durante um projeto de IC supervisionado pelo professor Alexei A. Mailybaev.

A motivação é estudar um modelo simples hierárquico de vórtices com simulações numéricas. 

Além disso, o modelo, incialmente estático, foi estendido para um modelo temporal. 

## Objetivos

- Simular partículas no campo
- Desenvolver otimizações numéricas para o cálculo do campo

## Modelo 

Os centros obedecem a uma estrutura hierárquica:

$$
R_n = R_1 2^{1-n}
$$

e 

$$
T_n = T_1 \lambda ^{1 -n }
$$


onde  $1<\lambda < 2$ e $R_n$ e $T_n$ são parametros que definem alcance e intensidade do campo, respectivamente. 

Campo local em torno de um centro:

$$
V_{n,i}(x,y) = \frac{2\pi}{T_n} B_C( \frac{|r-r_{n,i}|}{\sqrt{\alpha_{cut}} R_n})( - (y - y_{n,i}),  
x - x_{n,i})
$$

Função suave tipo bump:

$$
B(x) = 
\begin{cases} 
e^{-\frac{C}{1-x^2}} & \text{se } |x| < 1 \\ 
0 & \text{se } |x| \ge 1 
\end{cases}
$$

$$C > 0$$ 

## Distribuições dos centros.

São colocados $4^n$ centros de nível n da seguinte forma.

$$r_1 = 
\begin{pmatrix}
0 \\
0
\end{pmatrix}
$$
e segue a recursão
$$
r_{n}^{(i_1, i_2, ..., i_{n-1})} = 
r_{n-1}^{(i_1,\dots,i_{n-2})}(t)
+
\sqrt{2} R_n
\begin{pmatrix}
\cos(\Omega_n t + \phi(i_{n-1})) \\
\sin(\Omega_n t + \phi(i_{n-1}))
\end{pmatrix}
$$

onde:

$$
\phi(i)= \frac{\pi}{4} + \frac{\pi}{2} i,
\qquad
i \in \{0,1,2,3\}
$$

O modelo parado segue esta recursão porém com centros fixos e $t=0$

---

## Arquitetura de Otimização e Backends (CPU e GPU CUDA)

O projeto conta com uma arquitetura modular de execução que permite simular milhões de partículas mantendo equivalência matemática exata com o modelo analítico original:

1. **Módulo de Backends (`backend.py`):**
   - Fornece seleção dinâmica e segura de backends: `'auto'`, `'cpu'` e `'cuda'`.
   - **Modo `cpu`**: Execução vetorizada pura em NumPy, sem qualquer dependência de GPU NVIDIA ou CuPy.
   - **Modo `cuda`**: Execução na GPU NVIDIA via CuPy e RawKernel. Lança `CudaNotAvailableError` com diagnóstico detalhado se requisitado em ambiente incompatível.
   - **Modo `auto`**: Seleção automática com política de fallback transparente: utiliza CUDA se GPU NVIDIA e CuPy estiverem funcionais; caso contrário, executa em CPU.
   - Fornece cronômetros de alta precisão (`DeviceTimer`) compatíveis com `cuda.Event` na GPU e `time.perf_counter` na CPU.
   - Monitoramento de memória VRAM via `get_gpu_memory_info()`.

2. **Avaliação Vetorizada em CPU (`campos.py`):**
   - A função `campo_total_podado_vetorizado` elimina o laço sequencial em Python sobre partículas e substitui micro-alocações por indexação e operações agregadas vetorizadas (`np.add.at`), mantendo rigorosa equivalência numérica com a formulação de poda analítica (`campo_total_podado_2`).

3. **Aceleração em GPU com CUDA RawKernel (`cuda_kernels.py`):**
   - Implementado diretamente em C++ CUDA com compilação JIT via `cupy.RawKernel` (`-O3`, `--use_fast_math`).
   - Cada thread CUDA avalia uma partícula de forma independente em paralelismo massivo.
   - A árvore de centros ativos por nível é mantida em registradores e memória local de cada thread (capacidade até 32 centros por nível), eliminando alocações intermediárias de VRAM e tráfego desnecessário no barramento PCIe.

4. **Interface Unificada (`campos.campo_total`):**
   - Despacha automaticamente para o backend selecionado via argumento `backend_name='auto'|'cpu'|'cuda'`.

---

## Requisitos e Instalação na Máquina NVIDIA

### Pré-requisitos de Hardware e Driver:
- GPU NVIDIA com suporte a Compute Capability >= 6.0 (Pascal ou superior; ex: GTX 10xx, RTX 20xx/30xx/40xx, A100, H100).
- Driver NVIDIA recente instalado (ex: versão >= 525.xx).
- CUDA Toolkit (11.8 ou 12.x).

### Instalação no Ambiente Virtual Python:

```bash
# Ativação do venv existente no projeto:
source venv/bin/activate

# Para CUDA 12.x:
pip install cupy-cuda12x

# Ou para CUDA 11.x:
pip install cupy-cuda11x
```

---

## Suíte de Benchmarks (`benchmark_suite.py`)

O script `benchmark_suite.py` permite medir o speedup relativo, tempo de execução, latência de transferências e tempo de compilação JIT.

### Principais Opções de Linha de Comando:
- `--modo`: Seleciona o benchmark: `'campo'`, `'integradores'`, `'transferencias'`, `'compilacao'` ou `'todos'` (padrão: `'todos'`).
- `--backend`: Força o backend: `'auto'`, `'cpu'` ou `'cuda'` (padrão: `'auto'`).
- `--Np`: Lista de número de partículas para o campo (ex: `--Np 10 100 500 1000 5000`).
- `--n_max`: Lista de níveis hierárquicos para o campo (ex: `--n_max 3 6 8`).
- `--Np_int`: Lista de partículas para integradores (ex: `--Np_int 50 200 1000`).
- `--dt`: Passo de tempo para integração (padrão: `0.01`).
- `--t_max`: Tempo final de integração (padrão: `0.1`).
- `--repeticoes`: Número de repetições por medição (padrão: `5`).
- `--rapido`: Executa uma grade reduzida de testes para verificação rápida.
- `--dry-run`: Exibe o plano completo de medição sem executar nenhum cálculo.

### Exemplos de Uso:

```bash
# Inspeção rápida de parâmetros sem executar cálculos:
python3 benchmark_suite.py --dry-run

# Benchmark da avaliação do campo em CPU:
python3 benchmark_suite.py --modo campo --backend cpu --rapido

# Benchmark completo de aceleração na GPU NVIDIA:
python3 benchmark_suite.py --modo todos --backend cuda --n_max 3 6 8 --Np 100 1000 5000 10000

# Benchmark isolado de transferências PCIe (CPU <-> GPU):
python3 benchmark_suite.py --modo transferencias --backend cuda
```

---

## Suíte de Testes e Validação Numérica

1. **Testes Unitários e Equivalência Matemática:**
   - `tests/test_backend.py`: Valida a resolução de backends, detecção de hardware, fallback gracioso e mensagens de erro.
   - `tests/test_campos_equivalencia.py`: Compara as implementações CPU (`campo_total_correto`, `campo_total_podado_2`, `campo_total_podado_vetorizado`) e testa a equivalência com CUDA GPU (`test_cuda_vs_cpu_equivalencia_campo`, `test_cuda_casos_fronteira`). Em ambientes sem GPU NVIDIA, os testes de CUDA são pulados automaticamente (`skipped`).

2. **Runner de Dinâmica Física (`tests/run_all_tests.py`):**
   - Suporta seleção granular de testes e controle de backend (`--backend {auto, cpu, cuda}`).
   - Possui proteção contra execuções acidentais pesadas e suporte à flag `--dry-run`.
   - Exemplos de execução segura:
     ```bash
     # Inspeção rápida sem rodar simulações:
     python3 tests/run_all_tests.py --teste convergencia_dt --dry-run

     # Validação na GPU com poucas partículas:
     python3 tests/run_all_tests.py --teste estocasticidade --backend cuda --N_particulas 500 --t_max 0.1
     ```

---

## Limitações Conhecidas e Tolerâncias Numéricas

1. **Precisão Numérica (CPU IEEE 754 vs GPU Fast-Math):**
   - Na CPU, a biblioteca padrão C/glibc segue estritamente a especificação IEEE 754 de 64 bits para funções transcendentais (`exp`, `sin`, `cos`).
   - No kernel CUDA, a flag `--use_fast_math` utiliza aproximações de hardware nativas de altíssima velocidade. Diferenças na ordem de $10^{-7}$ a $10^{-8}$ em float64 são esperadas e fisicamente desprezíveis, não representando erro conceitual.

2. **Geradores Pseudoaleatórios (PRNGs) em Dinâmica Estocástica:**
   - O gerador de números aleatórios do NumPy (PCG64 na CPU) e do CuPy (Curand na GPU) utilizam algoritmos diferentes. Portanto, mesmo com a mesma semente (`seed`), trajetórias ruidosas individuais não serão idênticas bit a bit. A equivalência científica é estatística (convergência das distribuições, médias e dispersão $\sigma^2$).

3. **Capacidade de VRAM para Trajetórias Densas:**
   - Ao simular trajetórias densas (muitas partículas e muitos passos de tempo), a matriz $r(t) \in \mathbb{R}^{N_t \times N_p \times 2}$ é mantida na VRAM antes de ser retornada ao host. Para $N_p > 500.000$ e $N_t > 5.000$, monitore a memória disponível da GPU.


