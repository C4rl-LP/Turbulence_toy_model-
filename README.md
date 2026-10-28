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



## Instalação

Instale NumPy e a variante CuPy compatível com o CUDA do computador (por exemplo, `cupy-cuda12x` para CUDA 12.x; use `cupy-cuda11x` para CUDA 11.x):

```bash
python -m pip install numpy cupy-cuda12x matplotlib
```

Execute a partir da raiz do repositório:

```bash
python campos.py
```

O código de simulação usa CuPy e requer uma GPU NVIDIA com driver/CUDA compatíveis.
