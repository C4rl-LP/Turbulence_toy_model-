"""Campo podado float64: uma thread por partícula, DFS com memória O(níveis).

Não enumera a árvore completa: só desce aos filhos de centros aceitos.
A raiz sempre gera filhos (mesmo fora do seu suporte), como a referência.
Sem atomics: cada thread possui sua soma e sua pilha. FMA desabilitado para
preservar as operações separadas usadas na máscara da implementação CuPy.
"""
from functools import lru_cache
import numpy as np
import cupy as cp
import funcoes_para_centros as fc


_prepare = cp.ElementwiseKernel(
    'raw float64 cfg, float64 t', 'float64 shifts',
    '''int level = i / 8; int child = (i % 8) / 2;
       double angle = cfg[level*4+3]*t + (3.141592653589793238462643383279502884/4 + (3.141592653589793238462643383279502884/2)*child);
       shifts = cfg[level*4+2] * ((i%2) ? sin(angle) : cos(angle));''',
    'prepare_pruned_shifts', options=('--fmad=false',))


@lru_cache(maxsize=16)
def _config(device, values):
    # Cache por dispositivo e valores físicos; não guarda posições de partículas.
    return cp.asarray(np.asarray(values, dtype=np.float64).reshape(-1, 4))


@lru_cache(maxsize=16)
def _kernel(levels):
    source = r'''
    extern "C" __global__ void pruned_field(
        const double* x, const double* y, const long long N, const long long xs, const long long ys,
        const double* cfg, const double* shifts, const double c,
        double* vx, double* vy, unsigned long long* counts, const int record) {
        long long p = (long long)blockDim.x*blockIdx.x + threadIdx.x;
        if (p >= N) return;
        const int L = LEVELS;
        double cx[LEVELS], cy[LEVELS]; int next[LEVELS];
        double X=x[p*xs], Y=y[p*ys], ax=0., ay=0.;
        double r2=X*X+Y*Y;
        if (r2 < cfg[0]) {
            // Root: same division and operation order as campo_2.
            double pot=2.*3.141592653589793238462643383279502884*exp(c/(r2/cfg[0]-1.))/cfg[1];
            ax=-Y*pot; ay=X*pot;
        }
        if (record) {
            for(int j=0;j<L;++j) counts[p*L+j]=0;
            counts[p*L]=1;
        }
        cx[0]=0.; cy[0]=0.; next[0]=0;
        int depth=0;
        // Invariant: live frames 0..depth are initialized and depth < L.
        // The leaf guard prevents writing frame L; each child is in 0..3.
        while(depth>=0) {
            if(depth==L-1 || next[depth]==4) { --depth; continue; }
            int child=next[depth]++, level=depth+1;
            double sx=cx[depth]+shifts[level*8+child*2];
            double sy=cy[depth]+shifts[level*8+child*2+1];
            double dx=X-sx, dy=Y-sy, rr=dx*dx+dy*dy;
            double cut=cfg[level*4];
            if(rr < cut) {
                double pot=(2.*3.141592653589793238462643383279502884/cfg[level*4+1])*exp(c/(rr*(1./cut)-1.));
                ax += -dy*pot; ay += dx*pot;
                if(record) ++counts[p*L+level];
                ++depth; cx[depth]=sx; cy[depth]=sy; next[depth]=0;
            }
        }
        vx[p]=ax; vy[p]=ay;
    }'''.replace('LEVELS', str(levels))
    return cp.RawKernel(source, 'pruned_field', options=('--fmad=false',))


def evaluate(x, y, t, n_max, c, alpha, *, record=False):
    """Entradas 1D float64; saída na GPU. record é só para testes da poda."""
    levels = max(1, int(n_max))  # referência avalia a raiz até quando n_max <= 0
    values = tuple(value for n in range(1, levels+1) for value in
                   (fc.R_cufoff(n, alpha)**2, fc.T(n),
                    fc.R(n)*np.sqrt(2), fc.Omega(n)))
    cfg = _config(cp.cuda.runtime.getDevice(), values)
    shifts = cp.empty(levels*8, dtype=cp.float64)
    _prepare(cfg, np.float64(t), shifts)
    vx = cp.empty_like(x)
    vy = cp.empty_like(y)
    counts = cp.empty((x.size, levels), dtype=cp.uint64) if record else vx
    if x.size:
        _kernel(levels)(((x.size+127)//128,), (128,),
                        (x, y, np.int64(x.size), np.int64(x.strides[0]//8),
                         np.int64(y.strides[0]//8), cfg, shifts, np.float64(c),
                         vx, vy, counts, np.int32(record)))
    return (vx, vy, counts) if record else (vx, vy)
