"""Verificações rápidas de persistência e visualização, sem CuPy/CUDA.

python -m unittest tests.test_fluxo_resultados -v
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.dados_resultados import salvar_npz
from tests.plot_resultados import Arquivo, gerar_graficos, g


class FluxoResultados(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'dados'
        self.data.mkdir()

    def test_dicionarios_numericos_sem_pickle_e_metadados(self):
        path = self.data/'experimento.npz'
        values = np.array([1., 2., 3.])
        salvar_npz(path, t=values, stats={'sigma2':values}, distribuicoes=np.array([[1.,2.]], dtype=object))
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                self.assertFalse(data[key].dtype.hasobject)
        data = Arquivo(path)
        self.addCleanup(data.close)
        np.testing.assert_array_equal(data['stats']['sigma2'], values)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        self.assertIsNone(metadata['semente_cupy_registrada'])
        self.assertEqual(metadata['tempo_efetivo']['final'], 3.)

    def test_formato_antigo_exige_opcao_e_preserva_valores(self):
        path = self.data/'antigo.npz'
        np.savez_compressed(path, stats={'sigma2':np.array([0., .2])})
        data = Arquivo(path)
        self.addCleanup(data.close)
        with self.assertRaisesRegex(ValueError, 'permitir-pickle'):
            data['stats']
        legacy = Arquivo(path, permitir_pickle=True)
        self.addCleanup(legacy.close)
        np.testing.assert_array_equal(legacy['stats']['sigma2'], [0., .2])

    def test_mapas_representam_a_matriz_original_sem_carregar_trajetorias(self):
        path = self.data/'resumo_varredura.npz'
        matrix = np.array([[.01, .02], [.03, .04]])
        np.savez_compressed(path, N_fields=2, lista_alpha=[1.8,2.], lista_c=[.2,.6],
                            matriz_sigma2_final=matrix, matriz_norm_sigma2=matrix*100)
        before = path.read_bytes()
        captured = []
        original = g.plt.imshow
        def imshow(array, **kw):
            captured.append((np.asarray(array).copy(), kw))
            return original(array, **kw)
        with patch.object(g.plt, 'imshow', side_effect=imshow):
            gerar_graficos(path, self.root/'imagens', ['mapas'])
        self.assertEqual(len(captured), 2)
        np.testing.assert_array_equal(captured[0][0], matrix)
        np.testing.assert_array_equal(captured[1][0], np.log10(matrix*100))
        self.assertEqual(captured[0][1]['origin'], 'lower')
        self.assertEqual(captured[0][1]['extent'], [1.8,2.,.2,.6])
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(len(list((self.root/'imagens').glob('*.png'))), 2)

    def test_selecao_de_series_temporais(self):
        path = self.data/'series_temporais_N_2.npz'
        t = np.array([0., .01, .02])
        stats = {k:np.array([0., .2, .4]) for k in
                 ['mean_x','mean_y','var_x','var_y','cov_xy','sigma2','msd']}
        np.savez_compressed(path, t=t, **stats)
        captured = []
        original = g.plt.plot
        def plot(x,y,**kw):
            captured.append((x.copy(), y.copy()))
            return original(x,y,**kw)
        with patch.object(g.plt,'plot',side_effect=plot):
            gerar_graficos(self.data, self.root/'imagens', ['dispersao'])
        self.assertEqual([p.name for p in (self.root/'imagens').glob('*.png')], ['sigma2_vs_t.png'])
        np.testing.assert_array_equal(captured[0][0],t)
        np.testing.assert_array_equal(captured[0][1],stats['sigma2'])

    def test_rejeita_saida_sobre_dados(self):
        path = self.data/'x.npz'
        np.savez_compressed(path, a=[1.])
        with self.assertRaisesRegex(ValueError,'separada'):
            gerar_graficos(path,self.data)
        with self.assertRaisesRegex(ValueError,'separada'):
            gerar_graficos(path,self.root)
        other = self.root/'outra_pasta_com_dados'
        other.mkdir()
        np.savez_compressed(other/'dados.npz', a=[2.])
        with self.assertRaisesRegex(ValueError,'dados numéricos'):
            gerar_graficos(path,other)

    def test_importacao_da_visualizacao_nao_carrega_simuladores(self):
        code = '''
import sys
from tests import plot_resultados
assert not any(k in sys.modules for k in ['cupy','integradores','campos','campo_gpu','funcoes_para_centros'])
'''
        subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(__file__).resolve().parents[1])

    def test_entrada_inexistente_e_formato_desconhecido(self):
        with self.assertRaisesRegex(ValueError,'inexistente'):
            gerar_graficos(self.root/'ausente')
        np.savez_compressed(self.data/'desconhecido.npz', a=[1.])
        with self.assertRaisesRegex(ValueError,'suportados'):
            gerar_graficos(self.data,self.root/'imagens')


if __name__=='__main__':
    unittest.main()
