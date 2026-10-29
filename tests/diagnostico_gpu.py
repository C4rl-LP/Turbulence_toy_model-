"""Diagnóstico sem alterar produção: python tests/diagnostico_gpu.py --help.

Referência CPU: mesmas funções carregadas em namespaces privados, com cp=NumPy.
Não importa nem executa os drivers de testes que salvam resultados experimentais.
"""
import argparse
import ast
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_backend(xp):
    """Executa definições originais, sem imports de backend nem blocos __main__."""
    modules = {}
    for filename, alias in [('funcoes_para_centros.py', 'fc'),
                            ('integradores.py', 'it'), ('campos.py', 'campos')]:
        tree = ast.parse((ROOT / filename).read_text())
        tree.body = [node for node in tree.body
                     if not isinstance(node, (ast.Import, ast.ImportFrom, ast.If))]
        module = types.ModuleType('diagnostico_' + alias)
        module.__dict__.update(np=np, cp=xp, os=os, time=time,
                               product=itertools.product, **modules)
        if 'fc' in modules:
            module.__dict__.update({k: v for k, v in vars(modules['fc']).items()
                                    if not k.startswith('__') and k not in ('cp', 'np')})
        exec(compile(tree, str(ROOT / filename), 'exec'), module.__dict__)
        modules[alias] = module
    return modules


def timed(fn, cp=None, repeats=3):
    wall, events = [], []
    result = None
    for _ in range(repeats):
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
        tic = time.perf_counter()
        result = fn()
        if cp is not None:
            end.record()
            end.synchronize()
            events.append(cp.cuda.get_elapsed_time(start, end))
        wall.append(1000 * (time.perf_counter() - tic))
    return {'wall_ms': statistics.median(wall),
            'event_ms': statistics.median(events) if events else None}, result


def integrate(mod, r0, levels, steps, temporal=True, full=False):
    field = mod['campos'].campo_total_correto if full else mod['campos'].campo_total_podado
    xp = mod['it'].cp

    def rhs(t, r):
        return xp.column_stack(field(r[:, 0], r[:, 1], t if temporal else 0., levels))

    # dt exato binário; mantém dt e mesmos tempos ao variar N e níveis.
    return mod['it'].solve_RK4(rhs, r0, 0., 1 / 1024, steps / 1024)[1]


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {'max_abs': float(np.max(np.abs(a-b))),
            'allclose': bool(np.allclose(a, b, rtol=1e-10, atol=1e-12))}


@contextmanager
def identical_noise(mod, xp, seed):
    """Somente validação: mesmas amostras host; não usado nos timings."""
    original = mod['it'].cp
    rng = np.random.RandomState(seed)
    proxy = types.SimpleNamespace(**{name: getattr(xp, name) for name in
                                   ['asarray', 'tile', 'empty', 'column_stack', 'sqrt']})
    proxy.random = types.SimpleNamespace(normal=lambda **kw: xp.asarray(rng.normal(**kw)))
    mod['it'].cp = proxy
    try:
        yield
    finally:
        mod['it'].cp = original


def correctness(cpu, gpu, cp):
    rng = np.random.default_rng(812)
    r = np.vstack((rng.uniform(-1.7, 1.7, (64, 2)), [[0., 0.], [.3, .2333]]))
    g = cp.asarray(r)
    checks = []
    for levels in [1, 2, 3, 4]:
        for t in [0., .137]:
            a = cpu['campos'].campo_total_podado(r[:, 0], r[:, 1], t, levels)
            b = gpu['campos'].campo_total_podado(g[:, 0], g[:, 1], t, levels)
            full = cpu['campos'].campo_total_correto(r[:, 0], r[:, 1], t, levels)
            full_gpu = gpu['campos'].campo_total_correto(g[:, 0], g[:, 1], t, levels)
            masks_equal = True
            for parent_level in range(1, levels):
                cm = cpu['fc'].sondar_vet(parent_level, r[:, 0], r[:, 1], 0., 0., t)[2]
                gm = gpu['fc'].sondar_vet(parent_level, g[:, 0], g[:, 1], 0., 0., t)[2]
                masks_equal &= np.array_equal(cm, cp.asnumpy(gm))
            checks.append({'levels': levels, 't': t,
                           'cpu_gpu_field': error(a, cp.asnumpy(cp.stack(b))),
                           'cpu_gpu_full': error(full, cp.asnumpy(cp.stack(full_gpu))),
                           'pruning_masks_equal': bool(masks_equal),
                           'pruned_full': error(a, full)})
    for temporal in [False, True]:
        a = integrate(cpu, r[:8], 3, 4, temporal)
        b = integrate(gpu, g[:8], 3, 4, temporal)
        checks.append({'temporal': temporal, 'rk4': error(a, cp.asnumpy(b))})
        results = []
        for mod, xp in [(cpu, np), (gpu, cp)]:
            with identical_noise(mod, xp, 73):
                _, rs = mod['it'].simular_particulas_ruidosa(
                    8, 3, np.array([.3, .2333]), dt_1=1/1024, t_max=4/1024,
                    func=mod['campos'].campo_total_podado, temporal=temporal, sigma0=.001)
            results.append(cp.asnumpy(rs) if xp is cp else rs)
        checks.append({'temporal': temporal, 'sde_same_noise': error(*results)})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, help='JSON novo; recusa sobrescrever')
    p.add_argument('--profile', action='store_true', help='captura CUDA só de RK4 aquecido N=32,L=3,S=4')
    p.add_argument('--cpu-only', action='store_true')
    p.add_argument('--batch-control', action='store_true', help='controle vetorizado: nível 1, N=8192 e 65536')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats deve ser positivo')
    if args.output and args.output.exists():
        p.error('arquivo de saída já existe')
    tic = time.perf_counter()
    cpu = load_backend(np)
    report = {'python': sys.version, 'numpy': np.__version__, 'seed': 812,
              'dtype': 'float64', 'dt': 1/1024, 'rows': []}
    cp = None
    if not args.cpu_only:
        import cupy as cp
        # GPU usa os módulos de produção por seus imports normais.
        sys.path.insert(0, str(ROOT))
        import campos
        import integradores
        import funcoes_para_centros
        gpu = dict(campos=campos, it=integradores, fc=funcoes_para_centros)
        cp.cuda.Device(0).use()
        cp.zeros(1).sum().item()
        props = cp.cuda.runtime.getDeviceProperties(0)
        report.update(device={k: str(props[k]) for k in
                              ['name', 'totalGlobalMem', 'multiProcessorCount', 'major', 'minor']},
                      cupy=cp.__version__, runtime=cp.cuda.runtime.runtimeGetVersion(),
                      driver=cp.cuda.runtime.driverGetVersion())
    report['setup_ms'] = 1000*(time.perf_counter()-tic)
    print(json.dumps({'environment': report}), flush=True)
    if args.profile:
        r = cp.asarray(np.random.default_rng(812).uniform(.2, .4, (32, 2)))
        integrate(gpu, r, 3, 4)
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.profilerStart()
        integrate(gpu, r, 3, 4)
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.profilerStop()
        return
    # Pequeno primeiro; variações isoladas em N, níveis e passos; sem explosão cartesiana.
    cases = [(1, 1, 2), (8, 2, 2), (32, 3, 2), (128, 3, 2),
             (512, 3, 2), (32, 1, 2), (32, 2, 2), (32, 4, 2), (32, 3, 8)]
    if args.batch_control:
        cases = [(8192, 1, 2), (65536, 1, 2)]
    if args.validate_only:
        cases = []
    for n, levels, steps in cases:
        r = np.random.default_rng(812).uniform([.2, .15], [.4, .3], (n, 2))
        row = dict(particles=n, levels=levels, steps=steps)
        row['cpu_field'], _ = timed(lambda: cpu['campos'].campo_total_podado(r[:, 0], r[:, 1], .137, levels), repeats=args.repeats)
        row['cpu_rk4'], a = timed(lambda: integrate(cpu, r, levels, steps), repeats=args.repeats)
        if cp is not None:
            row['h2d'], g = timed(lambda: cp.asarray(r), cp, args.repeats)
            fn = lambda: gpu['campos'].campo_total_podado(g[:, 0], g[:, 1], .137, levels)
            row['gpu_first_field'], _ = timed(fn, cp, 1)
            row['gpu_field'], field = timed(fn, cp, args.repeats)
            row['gpu_first_rk4'], _ = timed(lambda: integrate(gpu, g, levels, steps), cp, 1)
            row['gpu_rk4'], b = timed(lambda: integrate(gpu, g, levels, steps), cp, args.repeats)
            assert isinstance(b, cp.ndarray) and all(isinstance(v, cp.ndarray) for v in field)
            row['d2h'], host = timed(lambda: cp.asnumpy(b), cp, args.repeats)
            row['rk4_error'] = error(a, host)
            def total_gpu():
                return cp.asnumpy(integrate(gpu, cp.asarray(r), levels, steps))
            row['gpu_total'], _ = timed(total_gpu, cp, args.repeats)
            row['speedup_rk4'] = row['cpu_rk4']['wall_ms']/row['gpu_rk4']['wall_ms']
            row['speedup_total'] = row['cpu_rk4']['wall_ms']/row['gpu_total']['wall_ms']
            free, total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            row['memory'] = dict(pool_used=pool.used_bytes(), pool_reserved=pool.total_bytes(),
                                 device_used=total-free, trajectory_bytes=b.nbytes)
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    if cp is not None:
        report['correctness'] = correctness(cpu, gpu, cp)
        print(json.dumps(report['correctness']), flush=True)
    if args.output:
        with args.output.open('x') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
