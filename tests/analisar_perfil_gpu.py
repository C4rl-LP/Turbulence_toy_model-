"""Resume o SQLite exportado pelo Nsight; não estima ocupação de warps.

python tests/analisar_perfil_gpu.py /tmp/turbulence-gpu-trace.sqlite
"""
import argparse
import json
from pathlib import Path
import sqlite3


def summarize(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        kernels = db.execute('SELECT start,end,gridX*gridY*gridZ,blockX*blockY*blockZ '
                             'FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
        if not kernels:
            raise ValueError('Sem kernels CUDA no trace')
        busy, end = 0, kernels[0][0]
        for start, stop, _, _ in kernels:
            busy += max(0, stop-max(start, end))
            end = max(end, stop)
        span = max(k[1] for k in kernels)-kernels[0][0]
        api = db.execute('SELECT s.value,COUNT(*),SUM(r.end-r.start) '
                         'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id '
                         'GROUP BY s.value').fetchall()
        copies = db.execute('SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start) '
                            'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
        return dict(kernels=len(kernels), kernel_sum_ms=sum(k[1]-k[0] for k in kernels)/1e6,
                    kernel_active_union_ms=busy/1e6, kernel_span_ms=span/1e6,
                    kernel_active_fraction=busy/span,
                    single_block_fraction=sum(k[2] == 1 for k in kernels)/len(kernels),
                    max_grid_blocks=max(k[2] for k in kernels),
                    small_block_le_8_threads=sum(k[3] <= 8 for k in kernels),
                    api={name: dict(calls=count, host_ms=ns/1e6) for name,count,ns in api},
                    copies={str(kind): dict(count=count, bytes=size, gpu_ms=ns/1e6)
                            for kind,count,size,ns in copies},
                    note='copyKind CUDA: 1=H2D, 2=D2H, 8=D2D; fração temporal não é ocupação SM')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    print(json.dumps(summarize(parser.parse_args().sqlite), indent=2))
