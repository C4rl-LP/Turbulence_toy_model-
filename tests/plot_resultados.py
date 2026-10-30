"""Reconstrói figuras de arquivos existentes, sem GPU ou simulações.

python tests/plot_resultados.py --input CAMINHO --output figuras/experimento --all
"""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import pickle
import re
import sys
import warnings
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import graficos_resultados as g

TIPOS = ('dispersao', 'trajetorias', 'distribuicoes', 'mapas',
         'estatisticas', 'difusao', 'convergencia')


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Pickles antigos de CuPy guardam dados NumPy e o construtor array.
        if module.startswith('cupy'):
            if name == 'array':
                return np.array
            raise pickle.UnpicklingError(f'Construtor CuPy não suportado: {module}.{name}')
        return super().find_class(module, name)


class Arquivo:
    """Acesso sob demanda: gráficos agregados não carregam trajetórias."""
    def __init__(self, path, permitir_pickle=False):
        self.path = Path(path)
        self.npz = np.load(self.path, allow_pickle=False)
        self.permitir_pickle = permitir_pickle

    def close(self):
        self.npz.close()

    def __contains__(self, key):
        return key in self.npz or any(k.startswith(key+'__') for k in self.npz.files)

    def __getitem__(self, key):
        if key not in self.npz:
            prefix = key+'__'
            children = {k[len(prefix):].split('__')[0] for k in self.npz.files if k.startswith(prefix)}
            if not children:
                raise KeyError(key)
            return {child: self[prefix+child] for child in sorted(children)}
        try:
            return self.npz[key]
        except ValueError as exc:
            if not self.permitir_pickle:
                raise ValueError(f'{self.path}:{key}: formato antigo com objetos; use '
                                 '--permitir-pickle apenas para arquivos de confiança.') from exc
            with zipfile.ZipFile(self.path) as z, z.open(key+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                reader(stream)
                value = _LegacyUnpickler(stream).load()
            return value.item() if isinstance(value, np.ndarray) and value.shape == () else value

    def get(self, key, default=None):
        return self[key] if key in self else default


class Contexto(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value):
            value = value()
            self[key] = value
        return value


def _selecionado(categorias, tipos):
    return tipos is None or bool(set(categorias) & set(tipos))


def _desenhar(familia, contexto, tipos, indices=None):
    for family, fn, categorias in g.REGISTRO:
        if family != familia or not _selecionado(categorias, tipos):
            continue
        if indices is not None and int(fn.__name__.rsplit('_', 1)[1]) not in indices:
            continue
        names = list(inspect.signature(fn).parameters)
        if all(name in contexto for name in names):
            for key in ['output_dir', 'sub_dir', 'pasta_ponto', 'pasta_info', 'pasta_saida', 'pasta_saida_info']:
                if key in names:
                    Path(contexto[key]).mkdir(parents=True, exist_ok=True)
            fn(**{key: contexto[key] for key in names})


def _metadados(path):
    sidecar = path.with_suffix('.json')
    return json.loads(sidecar.read_text(encoding='utf-8')).get('parametros', {}) if sidecar.exists() else {}


def _historico(path, alphas, cs, permitir_pickle):
    result = {}
    for c in [cs[0], cs[len(cs)//2], cs[-1]]:
        for alpha in [alphas[0], alphas[len(alphas)//2], alphas[-1]]:
            point = path.parent/f'alpha_{alpha:.3f}_c_{c:.3f}/dados_ponto.npz'
            if not point.exists():
                warnings.warn(f'Série temporal ausente: {point}')
                continue
            data = Arquivo(point, permitir_pickle)
            try:
                result[f'alpha_{alpha:.4f}_c_{c:.4f}'] = dict(t=data['t'], sigma2=data['sigma2_t'])
            finally:
                data.close()
    return result


def _histogramas(rs, title, output, prefix, tipos):
    if _selecionado(['distribuicoes'], tipos):
        g.gerar_histogramas_completos(rs, title, str(output), prefix)


def _render_arquivo(path, output, tipos, permitir_pickle):
    data = Arquivo(path, permitir_pickle)
    meta = _metadados(path)
    ctx = Contexto(output_dir=str(output), sub_dir=str(output), pasta_ponto=str(output),
                   pasta_saida=str(output), pasta_info=str(output/'informacoes'),
                   pasta_saida_info=str(output/'informacoes'))
    try:
        if path.name == 'resumo_varredura.npz':
            for key in ['N_fields', 'lista_alpha', 'lista_c', 'matriz_sigma2_final', 'matriz_norm_sigma2']:
                ctx[key] = data[key]
            ctx['historico_temporal'] = lambda: _historico(path, ctx['lista_alpha'], ctx['lista_c'], permitir_pickle)
            _desenhar('estocasticidade_parametros', ctx, tipos)
        elif path.name == 'dados_ponto.npz':
            ctx.update(alpha_val=float(data['alpha_cut']), c_val=float(data['c']),
                       delta_r=float(data['delta_r']), mean_final_ref=data['mean_final'])
            ctx['rs_ref'] = lambda: data['rs_ref']
            ctx['N_particulas'] = lambda: ctx['rs_ref'].shape[1]
            ctx['r0'] = lambda: ctx['rs_ref'][0, 0]
            _desenhar('estocasticidade_parametros', ctx, tipos)
            if _selecionado(['distribuicoes'], tipos):
                _histogramas(ctx['rs_ref'][-1], f"(alpha={ctx['alpha_val']:.2f}, c={ctx['c_val']:.2f})",
                             output, 'distribuicao_final', tipos)
        elif re.fullmatch(r'estocasticidade_nivel_\d+_dados.npz', path.name):
            ctx.update(n=int(data['N_fields']), dimensao_quadrado=float(data['dimensao_quadrado']),
                       mean_final=data['mean_final'], ts=data['t'])
            ctx['stats'] = lambda: {k:data[k] for k in ['sigma2']}
            ctx['rs'] = lambda: data['rs']
            ctx['rs_final'] = lambda: ctx['rs'][-1]
            ctx['r0'] = lambda: ctx['rs'][0, 0]
            ctx['N_particulas'] = lambda: ctx['rs'].shape[1]
            _desenhar('estocasticidade', ctx, tipos)
            if _selecionado(['distribuicoes'], tipos):
                _histogramas(ctx['rs_final'], f"Distribuição final (n={ctx['n']})", output, 'distribuicao_final', tipos)
        elif path.name == 'resumo_global.npz':
            ctx.update(niveis=data['niveis'], deltas=data['delta_r_n'],
                       sigma2_finais=data['sigma2_final'], dispersoes_finais=data['dispersao_media_final'])
            _desenhar('estocasticidade', ctx, tipos)
        elif path.name == 'convergencia_N_dados.npz':
            ctx.update(N_arr=data['N_fields'], dados=data)
            _desenhar('convergencia_N', ctx, tipos)
        elif path.name == 'convergencia_dt_dados.npz':
            ctx.update(dts=data['dt'], resultados=data)
            _desenhar('convergencia_dt', ctx, tipos)
            if _selecionado(['distribuicoes'], tipos):
                final = np.asarray(data['distribuicoes_finais'], dtype=float)
                for i, dt in enumerate(data['dt']):
                    ctx.update(idx=i, dt=dt, rs_final=final[i])
                    _desenhar('convergencia_dt', ctx, tipos, {1})
        elif path.name == 'robustez_estatistica_dados.npz':
            ctx.update(n_pts=data['N_particulas'], dados_robustez=data)
            _desenhar('robustez_estatistica', ctx, tipos)
        elif path.name == 'transicao_temporal_dados.npz':
            ctx.update(lista_N_fields=data['N_fields'], lista_param_valores=data['param_valores'],
                       param_nome=str(data['param_nome']),
                       dados_transicao={'resultados_por_N':{int(k):v for k,v in data['dados_por_N'].items()}})
            _desenhar('transicao_temporal', ctx, tipos)
        elif path.name == 'comparacao_simuladores_dados.npz':
            ctx.update(ts_det=data['ts_det'], ts_estoc=data['ts_estoc'],
                       N_fields=meta.get('N_fields', 'não registrado'))
            for mode in ['det', 'estoc']:
                ctx['stats_'+mode] = lambda mode=mode: data['stats_'+mode]
                ctx['rs_'+mode+'_cpu'] = lambda mode=mode: data['rs_'+mode]
            ctx['r0'] = lambda: ctx['rs_det_cpu'][0, 0]
            ctx['N_particulas'] = lambda: ctx['rs_det_cpu'].shape[1]
            _desenhar('comparacao_simuladores', ctx, tipos)
        elif path.name == 'distribuicoes_dados.npz':
            if _selecionado(['distribuicoes'], tipos):
                for name, result in data['relatorio'].items():
                    n = name.removeprefix('N_fields_')
                    prefix = f'distribuicao_N_{n}' if name.startswith('N_fields_') else 'distribuicao_deterministic'
                    title = f'N_fields = {n}' if name.startswith('N_fields_') else 'Simulador Determinístico (N=3)'
                    _histogramas(np.asarray(result['rs_final'], dtype=float), title, output, prefix, tipos)
        elif re.fullmatch(r'estabilidade_nivel_\d+_dados.npz', path.name):
            ctx.update(n=int(data['N_fields']), estocastico=bool(data['estocastico']), r0_array=data['r0'])
            ctx['rs_cpu'] = lambda: data['rs']
            ctx['rs_final_cpu'] = lambda: ctx['rs_cpu'][-1]
            ctx['N_particulas'] = lambda: ctx['rs_cpu'].shape[1]
            _desenhar('estabilidade', ctx, tipos, {2,3} if bool(data['estabilidade_temporal']) else {1})
        elif path.name.startswith('estocasticidade_espontanea_N_'):
            ctx.update(n=int(re.search(r'_N_(\d+)',path.name)[1]), deltas=data['delta_r'], res_n=data)
            ctx['r0'] = np.asarray(meta['r0']) if 'r0' in meta else None
            _desenhar('legado', ctx, tipos, {2,3,4})
            if _selecionado(['distribuicoes'], tipos):
                if ctx['r0'] is None:
                    warnings.warn(f'{path}: r0 não salvo; os scatters antigos com marca r0 não podem ser reproduzidos integralmente.')
                else:
                    for i, delta in enumerate(data['delta_r']):
                        ctx.update(d_idx=i, delta_r=delta, rs_final=np.asarray(data['distribuicoes_finais'][i], dtype=float))
                        _desenhar('legado', ctx, tipos, {1})
        elif path.name == 'comparacao_ruidos.npz':
            for mode in ['A','B','C']:
                ctx['ts_'+mode] = data['t_'+mode]
                ctx['stats_'+mode] = {key:data[prefix+'_'+mode] for key,prefix in
                                      [('var_x','var_x'),('sigma2','sigma2'),('dist_media_trajetorias','dist')]}
                ctx['rs_'+mode] = lambda mode=mode: data['rs_'+mode]
            _desenhar('legado', ctx, tipos, {5,6,7,8})
        else:
            return False
        return True
    finally:
        data.close()


def _series(paths, output, tipos, permitir_pickle, difusao=False):
    series = {}
    meta = {}
    for path in paths:
        n = int(re.search(r'_N_(\d+)',path.name)[1])
        data = Arquivo(path, permitir_pickle)
        try:
            series[n] = {k:data[k] for k in data.npz.files}
        finally:
            data.close()
        meta = _metadados(path)
    order = meta.get('lista_N_fields', sorted(series))
    series = {n:series[n] for n in order if n in series}
    if not difusao:
        plots = [('mean_x','Média <x>(t)','media_x_vs_t.png'),('mean_y','Média <y>(t)','media_y_vs_t.png'),
                 ('var_x','Variância Var[x](t)','variancia_x_vs_t.png'),('var_y','Variância Var[y](t)','variancia_y_vs_t.png'),
                 ('cov_xy','Covariância Cov[x,y](t)','covariancia_vs_t.png'),('sigma2',r'$\sigma^2(t)$ Total','sigma2_vs_t.png'),
                 ('msd','MSD(t)','msd_vs_t.png')]
        for key,title,name in plots:
            category = 'dispersao' if key=='sigma2' else 'difusao' if key=='msd' else 'estatisticas'
            if _selecionado([category], tipos):
                ctx=Contexto(chave=key,titulo=title,filename=name,lista_N_fields=list(series),
                             series_temporais_n=series,output_dir=str(output))
                _desenhar('estatisticas_temporais',ctx,None)
        return
    if not _selecionado(['difusao'],tipos):
        return
    fig, ax = g.plt.subplots(figsize=(7,5))
    fig_alpha, ax_alpha = g.plt.subplots(figsize=(7,5))
    for n,res in series.items():
        valid=(res['t']>0)&(res['msd']>0)
        ax.loglog(res['t'][valid],res['msd'][valid],label=f'N_fields = {n}')
        ax_alpha.plot(res['t'][valid],res['alpha_t'][valid],label=f'N_fields = {n}')
    t_max=meta.get('t_max',float(res['t'][-1]))
    t_ref=np.linspace(t_max/10,t_max,50)
    ax.loglog(t_ref,t_ref*(res['msd'][-1]/t_max),'k--',label='Referência difusiva (MSD ~ t)')
    ax.set(xlabel='Tempo t (escala log)',ylabel='MSD(t) (escala log)',title='Gráfico Log-Log do Deslocamento Quadrático Médio MSD(t)')
    ax.legend();ax.grid(True,which='both',ls='--',alpha=.3)
    fig.tight_layout();fig.savefig(output/'msd_loglog.png',dpi=200);g.plt.close(fig)
    ax_alpha.axhline(1.,color='black',linestyle='--',label='Difusão normal (alpha=1)')
    ax_alpha.set(xlabel='Tempo t',ylabel=r'Expoente local $\alpha(t) = d \log(MSD) / d \log(t)$',
                 title=r'Evolução Temporal do Expoente Difusivo Local $\alpha(t)$')
    ax_alpha.legend();ax_alpha.grid(True,alpha=.3)
    fig_alpha.tight_layout();fig_alpha.savefig(output/'expoente_difusao_alpha.png',dpi=200);g.plt.close(fig_alpha)


def gerar_graficos(entrada, saida=None, tipos=None, permitir_pickle=False):
    if isinstance(tipos, str):
        tipos = [tipos]
    if tipos is not None and set(tipos)-set(TIPOS):
        raise ValueError(f'Tipos desconhecidos: {set(tipos)-set(TIPOS)}')
    entrada=Path(entrada).resolve()
    if not entrada.exists():
        raise ValueError(f'Entrada inexistente: {entrada}')
    root=entrada if entrada.is_dir() else entrada.parent
    saida=Path(saida).resolve() if saida else root/'imagens'
    if saida==root or root.is_relative_to(saida):
        raise ValueError('A saída deve ser uma pasta separada dos arquivos originais.')
    if saida.exists() and any(saida.rglob('*.npz')):
        raise ValueError('A saída contém dados numéricos; escolha uma pasta de imagens separada.')
    paths=sorted(entrada.rglob('*.npz')) if entrada.is_dir() else [entrada]
    if not paths:
        raise ValueError('Nenhum arquivo NPZ encontrado.')
    groups=defaultdict(list)
    recognized=0
    saida.mkdir(parents=True,exist_ok=True)
    for path in paths:
        output=saida/path.parent.relative_to(root)
        output.mkdir(parents=True,exist_ok=True)
        if path.name.startswith('series_temporais_N_') or path.name.startswith('difusao_N_'):
            groups[(path.parent,path.name.startswith('difusao_'))].append(path)
        elif _render_arquivo(path,output,tipos,permitir_pickle):
            recognized+=1
    for (parent,difusao),group in groups.items():
        _series(group,saida/parent.relative_to(root),tipos,permitir_pickle,difusao)
        recognized+=len(group)
    if not recognized:
        raise ValueError('Os arquivos encontrados não pertencem aos formatos de experimentos suportados.')
    return recognized


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--all',action='store_true',help='todos os gráficos disponíveis (padrão)')
    selection.add_argument('--tipo',action='append',choices=TIPOS,help='categoria; pode repetir')
    parser.add_argument('--permitir-pickle',action='store_true',help='ler objetos de NPZ antigos de confiança; sem migração')
    args=parser.parse_args()
    try:
        count=gerar_graficos(args.input,args.output,args.tipo,args.permitir_pickle)
    except (ValueError,KeyError,pickle.UnpicklingError) as exc:
        parser.exit(2,f'Erro: {exc}\n')
    print(f'{count} arquivos numéricos reconhecidos. Gráficos salvos separadamente.')


if __name__=='__main__':
    main()
