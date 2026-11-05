import os
import sys
import argparse

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
        default=None,
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
        help="Nome do teste a ser executado. Especifique um teste ou 'todos'."
    )
    parser.add_argument("--N_particulas", type=int, default=10000, help="Número de partículas / realizações.")
    parser.add_argument("--t_max", type=float, default=1/5, help="Tempo máximo de simulação.")
    parser.add_argument("--seed", type=int, default=42, help="Semente de números aleatórios.")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Backend computacional de execução ('auto', 'cpu', 'cuda')."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exibe o plano de teste e parâmetros sem executar nenhuma simulação."
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" SUÍTE COMPLETA DE TESTES NUMÉRICOS DA DINÂMICA TURBULENTA")
    print(f" Backend selecionado: {args.backend}")
    print("=" * 70)

    if args.teste is None:
        print("\n[AVISO DE SEGURANÇA] Nenhum teste foi selecionado.")
        print("Para evitar execuções pesadas acidentais na CPU/RAM, especifique o teste desejado:")
        print("  python tests/run_all_tests.py --teste <nome_do_teste> [opções]")
        print("  python tests/run_all_tests.py --teste todos [opções]")
        print("\nTestes disponíveis:")
        print("  - convergencia_dt")
        print("  - convergencia_N")
        print("  - estocasticidade")
        print("  - estocasticidade_parametros")
        print("  - estatisticas_temporais")
        print("  - difusao")
        print("  - comparacao")
        print("  - transicao_temporal")
        print("  - distribuicoes")
        print("  - robustez")
        print("  - todos")
        print("\nDica: utilize --dry-run para inspecionar os parâmetros sem rodar simulações.\n")
        return

    if args.dry_run:
        print("\n[DRY RUN - MODO DE INSPEÇÃO SEM EXECUÇÃO]")
        print(f"  Teste selecionado : {args.teste}")
        print(f"  N_particulas      : {args.N_particulas}")
        print(f"  t_max             : {args.t_max}")
        print(f"  seed              : {args.seed}")
        print(f"  backend           : {args.backend}")
        print("\nNenhuma simulação foi iniciada.\n")
        return

    if args.teste in ["todos", "estocasticidade"]:
        executar_teste_estocasticidade(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            lista_N_fields=(1, 2, 3, 4, 5, 6, 7, 8),
            backend_name=args.backend
        )

    if args.teste in ["todos", "estocasticidade_parametros"]:
        for i in range(8):
            executar_teste_estocasticidade_parametros(
                N_particulas=args.N_particulas,
                t_max=args.t_max,
                seed=args.seed,
                N_fields=i,
                backend_name=args.backend
            )

    if args.teste in ["todos", "convergencia_dt"]:
        executar_teste_convergencia_dt(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "convergencia_N"]:
        executar_teste_convergencia_N(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "estatisticas_temporais"]:
        executar_teste_estatisticas_temporais(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "difusao"]:
        executar_teste_difusao(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "comparacao"]:
        executar_teste_comparacao_simuladores(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "transicao_temporal"]:
        executar_teste_transicao_temporal(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "distribuicoes"]:
        executar_teste_distribuicoes(
            N_particulas=args.N_particulas,
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    if args.teste in ["todos", "robustez"]:
        executar_teste_robustez(
            t_max=args.t_max,
            seed=args.seed,
            backend_name=args.backend
        )

    print("\nExecução concluída! Todos os resultados e gráficos foram salvos na pasta 'resultados/'.\n")


if __name__ == "__main__":
    main()
