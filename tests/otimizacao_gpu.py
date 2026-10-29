"""Validação/benchmark incremental, sem gravar resultados científicos.

python tests/otimizacao_gpu.py --output /tmp/novo.json [--profile N]
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cupy as cp
import campos
import campo_gpu
from diagnostico_gpu import load_backend, timed, integrate, correctness, error, identical_noise


def pruning_counts(x, y, t, levels, alpha):
    """Contagem independente com operações CuPy e a mesma poda da referência."""
    fc = campos.fc
    p = cp.arange(x.size)
    cx = cp.zeros(x.size); cy = cp.zeros(x.size)
    counts = [cp.ones(x.size, dtype=cp.uint64)]
    for n in range(1, levels):
        sx, sy = fc.subcentro(n, 0., 0., t)
        sx = (cx[:, None]+sx).ravel(); sy = (cy[:, None]+sy).ravel()
        p = cp.repeat(p, 4)
        dx=x[p]-sx; dy=y[p]-sy
        mask=dx*dx+dy*dy < fc.R_cufoff(n+1, alpha)**2
        p=p[mask]; cx=sx[mask]; cy=sy[mask]
        counts.append(cp.bincount(p, minlength=x.size).astype(cp.uint64))
    return cp.stack(counts, axis=1)


def validate(cpu, gpu):
    checks = correctness(cpu, gpu, cp)
    for row in checks:
        for value in row.values():
            if isinstance(value, dict) and 'allclose' in value:
                assert value['allclose'], row
    trajectory_checks=[]
    r=np.random.default_rng(812).uniform([.2,.15],[.4,.3],(12,2))
    for temporal in [False, True]:
        for levels in [3, 6, 8]:
            a=integrate(cpu,r,levels,16,temporal)
            b=cp.asnumpy(integrate(gpu,cp.asarray(r),levels,16,temporal))
            e=error(a,b); assert e['allclose'],e
            diff=np.abs(a-b)
            significant=np.abs(a)>1e-12
            trajectory_checks.append(dict(temporal=temporal,levels=levels,steps=16,
                max_rel=float(np.max(diff[significant]/np.abs(a[significant]))),
                final_abs=float(np.max(diff[-1])),**e))
        results=[]
        for mod,xp in [(cpu,np),(gpu,cp)]:
            def rhs(t,r):
                return xp.stack(mod['campos'].campo_total_podado(
                    r[:1],r[1:],t if temporal else 0.,3)).reshape(2)
            with identical_noise(mod,xp,94):
                _,rs=mod['it'].solve_euler_maruyama(rhs,np.array([.3,.2333]),4/1024,1/1024)
            results.append(cp.asnumpy(rs) if xp is cp else rs)
        e=error(*results); assert e['allclose'],e
        trajectory_checks.append(dict(temporal=temporal,euler_same_noise=e))
    rng=np.random.default_rng(921)
    boundary=[]
    for t in [0., .137]:
        for alpha in [.8, campos.alpha_padrao, 2.1]:
            points=[rng.uniform(-1.7,1.7,(48,2))]
            for n in range(1,6):
                center=campos.fc.x_centros(n,[0]*(n-1),t)
                radius=campos.fc.R_cufoff(n,alpha)
                # Exact, adjacent floats, and small offsets around the cutoff.
                for r in [np.nextafter(radius,0),radius,np.nextafter(radius,np.inf),radius*(1-1e-13),radius*(1+1e-13)]:
                    points.append(np.array([center+[r,0],center+[0,r]]))
            points=np.vstack(points); g=cp.asarray(points)
            for levels in [1,2,3,5,8]:
                a=campos.campo_total_podado_referencia(g[:,0],g[:,1],t,levels,c=.43,alpha=alpha)
                b=campos.campo_total_podado(g[:,0],g[:,1],t,levels,c=.43,alpha=alpha)
                e=error(cp.asnumpy(cp.stack(a)),cp.asnumpy(cp.stack(b)))
                assert e['allclose'], (t,alpha,levels,e)
                _,_,count=campo_gpu.evaluate(g[:,0],g[:,1],t,levels,.43,alpha,record=True)
                expected=pruning_counts(g[:,0],g[:,1],t,levels,alpha)
                cp.testing.assert_array_equal(count,expected)
                boundary.append(dict(t=t,alpha=float(alpha),levels=levels,**e))
    # Empty, scalar, reverse stride; float32 retains original path.
    for x,y in [(cp.array([],dtype=float),cp.array([],dtype=float)),
                (.3,.2333), (g[::-1,0],g[::-1,1]),
                (g[:,0].astype(cp.float32),g[:,1].astype(cp.float32))]:
        a=campos.campo_total_podado_referencia(x,y,.137,3)
        b=campos.campo_total_podado(x,y,.137,3)
        for aa,bb in zip(a,b):
            cp.testing.assert_allclose(aa,bb,rtol=1e-10,atol=1e-12)
    return dict(standard=checks,boundary=boundary,trajectories=trajectory_checks)



def safety_and_stages(cpu, gpu, memcheck_only=False):
    """Stress limitado de pilha e comparação explícita dos quatro estágios RK4."""
    stack=[]
    for levels in [1, 2, 8, 16, 32, 48, 64]:
        # Constrói um caminho profundo usando os próprios deslocamentos GPU.
        values=tuple(v for n in range(1,levels+1) for v in
            (campos.fc.R_cufoff(n,campos.alpha_padrao)**2,campos.fc.T(n),
             campos.fc.R(n)*np.sqrt(2),campos.fc.Omega(n)))
        cfg=campo_gpu._config(cp.cuda.runtime.getDevice(),values)
        shifts=cp.empty(levels*8,dtype=cp.float64)
        campo_gpu._prepare(cfg,np.float64(0.),shifts)
        shifts=cp.asnumpy(shifts).reshape(levels,4,2)
        point=np.zeros(2)
        for n in range(1,levels):
            point=point+shifts[n,0]
        points=np.tile(point,(129,1))
        points[0]=[10.,10.]
        g=cp.asarray(points)
        vx,vy,count=campo_gpu.evaluate(g[:,0],g[:,1],0.,levels,
            campos.c_padrao,campos.alpha_padrao,record=True)
        max_depth=int(cp.asnumpy(cp.nonzero(count[1])[0])[-1])+1
        assert max_depth==levels,(levels,max_depth)
        if memcheck_only:
            host=cp.asnumpy(count)
            assert np.all(host[:,0]==1)
            assert np.all(host[0,1:]==0)
            assert np.all(host[1:,-1]>0)
            assert bool(cp.asnumpy(cp.all(cp.isfinite(vx) & cp.isfinite(vy))))
        else:
            expected=pruning_counts(g[:,0],g[:,1],0.,levels,campos.alpha_padrao)
            cp.testing.assert_array_equal(count,expected)
            a=campos.campo_total_podado_referencia(g[1:2,0],g[1:2,1],0.,levels)
            cp.testing.assert_allclose(cp.stack(a),cp.stack((vx[1:2],vy[1:2])),rtol=1e-10,atol=1e-12)
        stack.append(dict(levels=levels,reached=max_depth,particles=129))
    if memcheck_only:
        return dict(stack=stack)
    stages=[]
    for temporal in [False,True]:
        for levels in [1,3,6,8]:
            r=np.random.default_rng(812).uniform([.2,.15],[.4,.3],(16,2))
            states=[r.copy(),cp.asarray(r),cp.asarray(r)]
            methods=[cpu['campos'].campo_total_podado,
                     campos.campo_total_podado_referencia,campos.campo_total_podado]
            dt=1/1024
            for step in range(4):
                all_k=[]
                for j,(state,method) in enumerate(zip(states,methods)):
                    xp=np if j==0 else cp
                    def rhs(t,r):
                        return xp.column_stack(method(r[:,0],r[:,1],t if temporal else 0.,levels))
                    t=step*dt
                    k1=rhs(t,state)
                    k2=rhs(t+dt/2,state+dt/2*k1)
                    k3=rhs(t+dt/2,state+dt/2*k2)
                    k4=rhs(t+dt,state+dt*k3)
                    all_k.append([k1,k2,k3,k4])
                    states[j]=state+dt/6*(k1+2*k2+2*k3+k4)
                for k in range(4):
                    a=all_k[0][k]
                    for j in [1,2]:
                        b=cp.asnumpy(all_k[j][k]); e=error(a,b)
                        assert e['allclose'],(temporal,levels,step,k,j,e)
                        significant=np.abs(a)>1e-12
                        rel=float(np.max(np.abs(a-b)[significant]/np.abs(a[significant]))) if significant.any() else 0.
                        stages.append(dict(temporal=temporal,levels=levels,step=step,
                            stage=k+1,backend=j,max_rel=rel,**e))
    return dict(stack=stack,stages=stages)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--profile',type=int)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--safety-only',action='store_true')
    p.add_argument('--crossover',action='store_true')
    p.add_argument('--cold',action='store_true')
    p.add_argument('--memcheck-only',action='store_true')
    args=p.parse_args()
    if args.output and args.output.exists():
        p.error('saída já existe')
    cpu=load_backend(np)
    gpu=dict(campos=campos,it=campos.it,fc=campos.fc)
    optimized=campos.campo_total_podado
    if args.cold:
        g=cp.asarray(np.random.default_rng(812).uniform(.2,.4,(8192,2)))
        cp.cuda.get_current_stream().synchronize()
        first,_=timed(lambda:optimized(g[:,0],g[:,1],.137,3),cp,1)
        warm,_=timed(lambda:optimized(g[:,0],g[:,1],.137,3),cp)
        report=dict(first=first,warm=warm)
        print(json.dumps(report),flush=True)
        if args.output:
            with args.output.open('x') as f:
                json.dump(report,f,indent=2)
        return
    if args.profile:
        g=cp.asarray(np.random.default_rng(812).uniform(.2,.4,(args.profile,2)))
        integrate(gpu,g,3,4); cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.profilerStart()
        integrate(gpu,g,3,4)
        cp.cuda.get_current_stream().synchronize(); cp.cuda.runtime.profilerStop()
        return
    tic=time.perf_counter()
    validation=safety_and_stages(cpu,gpu,args.memcheck_only) if (args.safety_only or args.memcheck_only) else validate(cpu,gpu)
    print('Validação passou',time.perf_counter()-tic,flush=True)
    rows=[]
    cases=[(1,1,2),(8,2,2),(32,3,2),(128,3,2),(512,3,2),
           (32,1,2),(32,2,2),(32,4,2),(32,3,8),
           (2048,3,2),(8192,3,2),(65536,3,2),(512,6,2),(512,3,16)]
    if args.crossover:
        cases=[(N,3,2) for N in [1,2,4,8,16]]
    if args.validate_only or args.safety_only or args.memcheck_only:
        cases=[]
    for N,L,S in cases:
        r=np.random.default_rng(812).uniform([.2,.15],[.4,.3],(N,2)); g=cp.asarray(r)
        row=dict(particles=N,levels=L,steps=S)
        row['cpu_field'],_=timed(lambda:cpu['campos'].campo_total_podado(r[:,0],r[:,1],.137,L))
        row['cpu_rk4'],a=timed(lambda:integrate(cpu,r,L,S))
        for label,field in [('new',optimized),('old',campos.campo_total_podado_referencia)]:
            # O antigo cresce linearmente com chamadas host; evitar minutos inúteis.
            if label=='old' and (N>512 or S>8 or L>4):
                continue
            campos.campo_total_podado=field
            row[label+'_first_field'],_=timed(lambda:field(g[:,0],g[:,1],.137,L),cp,1)
            row[label+'_field'],_=timed(lambda:field(g[:,0],g[:,1],.137,L),cp)
            integrate(gpu,g,L,S)
            row[label+'_rk4'],b=timed(lambda:integrate(gpu,g,L,S),cp)
            e=error(a,cp.asnumpy(b)); assert e['allclose'],e
            row[label+'_error']=e
            row[label+'_total'],_=timed(lambda:cp.asnumpy(integrate(gpu,cp.asarray(r),L,S)),cp)
        campos.campo_total_podado=optimized
        pool=cp.get_default_memory_pool()
        row['memory']=dict(used=pool.used_bytes(),reserved=pool.total_bytes(),trajectory=(S+1)*N*16)
        rows.append(row); print(json.dumps(row),flush=True)
    if args.output:
        with args.output.open('x') as f:
            json.dump(dict(validation=validation,rows=rows,dt=1/1024,repeats=3),f,indent=2)

if __name__=='__main__':
    main()
