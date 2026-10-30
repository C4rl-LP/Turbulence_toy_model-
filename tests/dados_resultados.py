"""Persistência de experimentos; não importa CuPy, Matplotlib ou simuladores."""
import inspect
import json
from pathlib import Path
import sys

import numpy as np


def para_numpy(value):
    if type(value).__module__.startswith('cupy') and hasattr(value, 'get'):
        return value.get()
    return np.asarray(value)


def _flatten(values, prefix=''):
    for key, value in values.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            yield from _flatten(value, name + '__')
        else:
            array = para_numpy(value)
            if array.dtype.hasobject:
                # Arquivos novos não precisam de pickle para arrays numéricos.
                array = np.asarray(array.tolist(), dtype=float)
            yield name, array


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    return str(value)


def salvar_npz(filename, **values):
    """Preserva chaves numéricas; dicionários viram chaves ``pai__filho``.

    O JSON adjacente registra apenas configurações pequenas do chamador.
    Não altera o RNG nem o cálculo das estatísticas.
    """
    filename = Path(filename)
    np.savez_compressed(filename, **dict(_flatten(values)))
    caller = inspect.currentframe().f_back
    try:
        context = caller.f_locals
        names = (
            'r0', 'N_fields', 'n', 'N_min', 'N_max', 'nivel_min', 'nivel_max',
            'N_particulas', 'n_part', 'lista_N_particulas', 'lista_N_fields',
            'lista_alpha', 'lista_c', 'alpha_val', 'c_val', 'param_nome',
            'lista_param_valores', 'dt', 'dt_n', 'dt_1', 'dt_ref', 'lista_dt',
            't_max', 'temporal', 'seed', 'seed_amostra', 'num_amostras',
            'delta_r0', 'delta_r', 'dimensao_quadrado', 'L', 'R_n', 'R_N',
            'func', 'funcao', 'simulador', 'sigma0')
        config = {k: _json_value(context[k]) for k in names if k in context}
        fc = sys.modules.get('funcoes_para_centros')
        physics = {k: _json_value(getattr(fc, k)) for k in
                   ['R_1', 'T_1', 'lamb', 'alpha_padrao', 'c_padrao'] if hasattr(fc, k)}
        physics['lambda'] = physics.get('lamb')
        times = next((values[k] for k in ['t', 'ts_ref', 'ts_det', 't_A'] if k in values), None)
        temporal = config.get('temporal', False)
        metadata = dict(formato=2, experimento=caller.f_code.co_name,
                        arquivo=filename.name, parametros=config, fisica=physics,
                        campo='temporal' if temporal else 'estatico',
                        semente_numpy_solicitada=config.get('seed'),
                        semente_cupy_registrada=None,
                        nota_sementes='O RNG existente foi preservado. A semente NumPy não controla o RNG CuPy; estado externo não registrado.',
                        dicionarios='chaves separadas por __; arrays sem pickle')
        metadata['N_fields'] = _json_value(values.get('N_fields', config.get('N_fields', config.get('n'))))
        metadata['alpha_cut'] = _json_value(values.get('alpha_cut', values.get('lista_alpha', config.get('alpha_val', physics.get('alpha_padrao')))))
        metadata['c'] = _json_value(values.get('c', values.get('lista_c', config.get('c_val', physics.get('c_padrao')))))
        if 'seed_amostra' in context and 'rs_ref' in values:
            first = context['seed_amostra'] - context['num_amostras'] + 1
            metadata['sementes_numpy_amostras'] = list(range(first, context['seed_amostra']+1))
        if config.get('func') and 'campo_total_podado' not in config['func']:
            metadata['nota_campo'] = 'Função personalizada: constantes padrão não certificam os parâmetros internos do callback.'
        if times is not None:
            times = para_numpy(times)
            metadata['tempo_efetivo'] = dict(inicial=float(times[0]), final=float(times[-1]),
                                            dt=float(times[1]-times[0]) if len(times)>1 else None)
        filename.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    finally:
        del caller
