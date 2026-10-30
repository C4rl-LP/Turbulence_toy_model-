"""Benchmark curto do simulador e do driver real de estocasticidade.

Não modifica produção. Dados de validação ficam em diretório temporário novo.
python tests/benchmark_estocasticidade_gpu.py --output /tmp/novo.json
Acrescente --numpy-vs-cupy para medir CPU NumPy versus GPU otimizada.
python tests/benchmark_estocasticidade_gpu.py --numpy-vs-cupy \
    --case 8,3,4 --case 128,3,4 --repeats 3 --output /tmp/numpy-cupy.json
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cupy as cp
import matplotlib
matplotlib.use('Agg')
import campos
import integradores as it
import funcoes_para_centros as fc
from tests.test_estocasticidade import executar_teste_estocasticidade
from tests.diagnostico_gpu import timed, error, load_backend



def benchmark_numpy_vs_cupy(output, *, cases=None, repeats=3, seed=812):
    """Mesma nuvem pré-gerada; RK4 CPU vs GPU, com e sem transferências."""
    cpu=load_backend(np)
    rows=[]
    props=cp.cuda.runtime.getDeviceProperties(0)
    cp.zeros(1).sum().item()  # inicialização fora do timing aquecido
    if cases is None:
        cases=[(1,1,4),(1,3,4),(8,3,4),(32,3,4),(128,3,4),(512,3,4),
               (2048,3,4),(8192,3,4),(128,5,4),(128,3,16)]
    for N,L,steps in cases:
        # A mesma construção do disco usada em simular_particulas.
        rng=np.random.default_rng(seed)
        theta=rng.uniform(0,2*np.pi,N-1)
        u=rng.uniform(0,1,N-1)
        radius=(.1*fc.R(L)/2)*np.sqrt(u)
        r0=np.array([.3,.2333])
        r=np.vstack((r0,r0+np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))))
        g=cp.asarray(r)
        dt=.01*fc.T(L)
        for temporal in [False,True]:
            def integrate(xp,mod,initial):
                field=mod['campos'].campo_total_podado
                def rhs(t,state):
                    return xp.column_stack(field(state[:,0],state[:,1],t if temporal else 0.,L))
                return mod['it'].solve_RK4(rhs,initial,0.,dt,steps*dt)[1]
            gpu=dict(campos=campos,it=it)
            row=dict(particles=N,levels=L,steps=steps,temporal=temporal,dt=dt,t_max=steps*dt)
            def cpu_field():
                return cpu['campos'].campo_total_podado(r[:,0],r[:,1],.137 if temporal else 0.,L)
            def gpu_field():
                return campos.campo_total_podado(g[:,0],g[:,1],.137 if temporal else 0.,L)
            cpu_field()
            row['gpu_first_field'],_=timed(gpu_field,cp,1)
            row['cpu_field'],a=timed(cpu_field,repeats=repeats)
            row['gpu_field'],b=timed(gpu_field,cp,repeats)
            row['field_error']=error(np.stack(a),cp.asnumpy(cp.stack(b)))
            assert row['field_error']['allclose'],row
            integrate(np,cpu,r)
            row['gpu_first_rk4'],_=timed(lambda:integrate(cp,gpu,g),cp,1)
            row['cpu_rk4'],a=timed(lambda:integrate(np,cpu,r),repeats=repeats)
            row['gpu_rk4'],b=timed(lambda:integrate(cp,gpu,g),cp,repeats)
            assert a.shape==(steps+1,N,2),a.shape
            row['trajectory_error']=error(a,cp.asnumpy(b))
            assert row['trajectory_error']['allclose'],row
            def total_gpu():
                return cp.asnumpy(integrate(cp,gpu,cp.asarray(r)))
            row['gpu_total'],host=timed(total_gpu,cp,repeats)
            np.testing.assert_array_equal(r,a[0])
            np.testing.assert_array_equal(r,host[0])
            np.testing.assert_allclose(a,host,rtol=1e-10,atol=1e-12)
            # Observável usado no teste; validação fora do tempo medido.
            def sigma2(trajectory):
                mean=trajectory.mean(axis=1,keepdims=True)
                return np.mean(np.sum((trajectory-mean)**2,axis=2),axis=1)
            row['sigma2_error']=error(sigma2(a),sigma2(host))
            assert row['sigma2_error']['allclose'],row
            row['speedup_field']=row['cpu_field']['wall_ms']/row['gpu_field']['wall_ms']
            row['speedup_rk4']=row['cpu_rk4']['wall_ms']/row['gpu_rk4']['wall_ms']
            row['speedup_total']=row['cpu_rk4']['wall_ms']/row['gpu_total']['wall_ms']
            pool=cp.get_default_memory_pool()
            row['memory']=dict(pool_used=pool.used_bytes(),pool_reserved=pool.total_bytes(),trajectory=b.nbytes)
            rows.append(row)
            print(json.dumps(row),flush=True)
    report=dict(mode='numpy_vs_cupy_optimized',seed=seed,dtype='float64',repeats=repeats,
                numpy=np.__version__,cupy=cp.__version__,device=props['name'].decode(),
                physics=dict(r0=[.3,.2333],delta_r0=.1,R_1=fc.R_1,T_1=fc.T_1,
                             lamb=fc.lamb,c=fc.c_padrao,alpha=fc.alpha_padrao),
                rows=rows,note='Nuvem idêntica pré-gerada; RK4 e campo do teste, sem gráficos ou arquivos científicos. CPU usa algoritmo podado original em NumPy.')
    with output.open('x') as f:
        json.dump(report,f,indent=2)
    print('Saída:',output,flush=True)


def parse_case(value):
    """Caso explícito: partículas,níveis,passos (inteiros positivos)."""
    try:
        case=tuple(int(part) for part in value.split(','))
    except ValueError:
        case=()
    if len(case)!=3 or any(part<1 for part in case):
        raise argparse.ArgumentTypeError('use partículas,níveis,passos positivos, ex.: 128,3,4')
    return case

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--numpy-vs-cupy',action='store_true')
    parser.add_argument('--case',type=parse_case,action='append',
                        help='caso NumPy/CuPy: partículas,níveis,passos; pode repetir')
    parser.add_argument('--repeats',type=int,default=3,help='repetições NumPy/CuPy após aquecimento')
    parser.add_argument('--seed',type=int,default=812,help='semente da nuvem NumPy/CuPy compartilhada')
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('--repeats deve ser positivo')
    if not args.numpy_vs_cupy and (args.case or args.repeats!=3 or args.seed!=812):
        parser.error('--case, --repeats e --seed exigem --numpy-vs-cupy')
    if args.output.exists():
        parser.error('saída já existe')
    if args.numpy_vs_cupy:
        benchmark_numpy_vs_cupy(args.output,cases=args.case,repeats=args.repeats,seed=args.seed)
        return
    work=Path(tempfile.mkdtemp(prefix='turbulence-estocasticidade-bench-'))
    rows=[]
    steps=4
    # Caso mínimo primeiro; tempo/escala física permanecem iguais antes/depois.
    for N,L in [(8,2),(32,3),(128,3),(128,5),(512,3)]:
        dt=.01*fc.T(L)
        for temporal in [False,True]:
            def simulate(field):
                # Mesmas amostras CuPy da nuvem, não apenas np.random.seed.
                cp.random.seed(812)
                return it.simular_particulas(N,L,.1*fc.R(L),np.array([.3,.2333]),
                    dt=dt,t_max=steps*dt,func=field,temporal=temporal)
            row=dict(particles=N,levels=L,steps=steps,dt=dt,t_max=steps*dt,temporal=temporal)
            results={}
            for label,field in [('original',campos.campo_total_podado_referencia),
                                ('optimized',campos.campo_total_podado)]:
                # Aquecimento separado; o RNG é redefinido em cada chamada.
                t,r=simulate(field); cp.cuda.get_current_stream().synchronize()
                row[label],(t,r)=timed(lambda:simulate(field),cp)
                assert len(t)==steps+1
                results[label]=cp.asnumpy(r)
            np.testing.assert_array_equal(results['original'][0],results['optimized'][0])
            row['trajectory_error']=error(results['original'],results['optimized'])
            assert row['trajectory_error']['allclose']
            row['speedup']=row['original']['wall_ms']/row['optimized']['wall_ms']
            row['pool_reserved_bytes']=cp.get_default_memory_pool().total_bytes()
            rows.append(row)
            print(json.dumps(row),flush=True)
    # Driver numérico completo: dados preservados em diretório novo por execução.
    # Lote pequeno evita transformar o benchmark em uma simulação científica longa.
    whole=[]
    N,L=32,3
    dt=.01*fc.T(L)
    for temporal in [False,True]:
        samples={'original':[],'optimized':[]}
        outputs={}
        for rep in range(4):  # primeira rodada excluída (aquecimento de I/O)
            # Alterna ordem para reduzir viés de execução.
            labels=['original','optimized'] if rep%2==0 else ['optimized','original']
            for label in labels:
                field=(campos.campo_total_podado_referencia if label=='original'
                       else campos.campo_total_podado)
                out=work/f'{temporal}-{rep}-{label}'
                cp.random.seed(812)
                cp.cuda.get_current_stream().synchronize()
                tic=time.perf_counter()
                with redirect_stdout(io.StringIO()):
                    executar_teste_estocasticidade(
                        lista_N_fields=(L,),N_particulas=N,t_max=steps*dt,
                        dt_ref=None,func=field,temporal=temporal,output_dir=str(out),
                        delta_r0=.1,seed=812)
                cp.cuda.get_current_stream().synchronize()
                elapsed=1000*(time.perf_counter()-tic)
                if rep:
                    samples[label].append(elapsed)
                data=np.load(out/f'N_fields_{L}'/f'estocasticidade_nivel_{L}_dados.npz')
                outputs[label]={key:data[key].copy() for key in ['rs','sigma2','mean_final']}
                data.close()
        for key in outputs['original']:
            np.testing.assert_allclose(outputs['original'][key],outputs['optimized'][key],rtol=1e-10,atol=1e-12)
        np.testing.assert_array_equal(outputs['original']['rs'][0],outputs['optimized']['rs'][0])
        a,b=[statistics.median(samples[key]) for key in ['original','optimized']]
        row=dict(particles=N,levels=L,steps=steps,temporal=temporal,dt=dt,
                 original_total_ms=a,optimized_total_ms=b,speedup=a/b,
                 samples_ms=samples,trajectory_error=error(outputs['original']['rs'],outputs['optimized']['rs']),
                 sigma2_error=error(outputs['original']['sigma2'],outputs['optimized']['sigma2']))
        whole.append(row)
        print('COMPLETE '+json.dumps(row),flush=True)
    report=dict(seed=812,dtype='float64',repeats=3,simulation=rows,complete_test=whole,
                artifact_directory=str(work),
                note='Teste determinístico: aleatoriedade apenas na nuvem inicial; mesmos valores antes/depois.')
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
    print('Saída:',args.output,'Artefatos:',work,flush=True)


if __name__=='__main__':
    main()
