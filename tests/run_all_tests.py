import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_convergencia_dt import executar_teste_convergencia_dt
from tests.test_convergencia_N import executar_teste_convergencia_N
from tests.test_estocasticidade import executar_teste_estocasticidade
from tests.test_estatisticas_temporais import executar_teste_estatisticas_temporais
from tests.test_difusao import executar_teste_difusao
from tests.test_comparacao_simuladores import executar_teste_comparacao_simuladores
from tests.test_transicao_temporal import executar_teste_transicao_temporal
from tests.test_distribuicoes import executar_teste_distribuicoes
from tests.test_robustez_estatistica import executar_teste_robustez
from tests.test_estocasticidade_parametros import executar_teste_estocasticidade_parametros


def main():
    parser = argparse.ArgumentParser(
        description="Runner mestre para a suíte de testes numéricos de estocasticidade espontânea e convergência."
    )
    parser.add_argument(
        "--teste",
        type=str,
        default="todos",
        choices=[
            "todos",
            "convergencia_dt",
            "convergencia_N",
            "estocasticidade",
            "estocasticidade_parametros",
            "estatisticas_temporais",
            "difusao",
            "comparacao",
            "transicao_temporal",
            "distribuicoes",
            "robustez"
        ],
        help="Nome do teste a ser executado."
    )
    parser.add_argument("--N_particulas", type=int, default=10000, help="Número de partículas / realizações.")
    parser.add_argument("--t_max", type=float, default=1/5, help="Tempo máximo de simulação.")
    parser.add_argument("--seed", type=int, default=10, help="Semente de números aleatórios.")
    parser.add_argument("--output", type=Path, default=Path('resultados'), help="Raiz dos dados numéricos.")
    parser.add_argument("--niveis", type=int, nargs='+', help="Níveis explícitos para casos pequenos; mantém os padrões se omitido.")
    parser.add_argument("--campo", choices=['estatico', 'temporal', 'ambos'], default=None,
                        help="Tipo de campo; mantém o padrão anterior de cada teste se omitido.")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" SUÍTE COMPLETA DE TESTES NUMÉRICOS DA DINÂMICA TURBULENTA")
    print("=" * 70)
    common = dict(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)
    fields = [True, False] if args.campo in (None, 'ambos') else [args.campo=='temporal']
    if args.teste in ['todos', 'estocasticidade_parametros']:
        for temporal in fields:
            levels = args.niveis if args.niveis is not None else (range(6,9) if temporal else range(8))
            name = 'estocasticidade_parametros_'+('temporal' if temporal else 'estatico')
            for n in levels:
                executar_teste_estocasticidade_parametros(**common, N_fields=n, temporal=temporal,
                    output_dir=str(args.output/name/f'em_nivel_{n}'))
    if args.teste in ['todos', 'estocasticidade']:
        for temporal in fields:
            name = 'estocasticidade_'+('temporal' if temporal else 'estatico')
            executar_teste_estocasticidade(**common, lista_N_fields=args.niveis or tuple(range(1,9)),
                temporal=temporal, output_dir=str(args.output/name))
    others = {
        'convergencia_dt': executar_teste_convergencia_dt,
        'convergencia_N': executar_teste_convergencia_N,
        'estatisticas_temporais': executar_teste_estatisticas_temporais,
        'difusao': executar_teste_difusao,
        'comparacao': executar_teste_comparacao_simuladores,
        'transicao_temporal': executar_teste_transicao_temporal,
        'distribuicoes': executar_teste_distribuicoes,
        'robustez': executar_teste_robustez,
    }
    import inspect
    for name, fn in others.items():
        if args.teste not in ('todos', name):
            continue
        params = inspect.signature(fn).parameters
        kw = {k:v for k,v in common.items() if k in params}
        if args.niveis:
            if 'lista_N_fields' in params:
                kw['lista_N_fields'] = args.niveis
            elif 'N_fields' in params:
                kw['N_fields'] = args.niveis[0]
            elif 'N_min' in params:
                kw.update(N_min=min(args.niveis), N_max=max(args.niveis))
        modes = fields if args.campo is not None and 'temporal' in params else [None]
        for mode in modes:
            suffix = '' if mode is None else ('_temporal' if mode else '_estatico')
            if mode is not None:
                kw['temporal'] = mode
            fn(**kw, output_dir=str(args.output/(name+suffix)))
    print(f"\nExecução concluída. Somente dados numéricos salvos em {args.output}.\n")


if __name__ == "__main__":
    main()
