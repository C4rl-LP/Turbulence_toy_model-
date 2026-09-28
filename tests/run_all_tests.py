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
            "estatisticas_temporais",
            "difusao",
            "comparacao",
            "transicao_temporal",
            "distribuicoes",
            "robustez"
        ],
        help="Nome do teste a ser executado."
    )
    parser.add_argument("--N_particulas", type=int, default=300, help="Número de partículas / realizações.")
    parser.add_argument("--t_max", type=float, default=1.0, help="Tempo máximo de simulação.")
    parser.add_argument("--seed", type=int, default=42, help="Semente de números aleatórios.")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" SUÍTE COMPLETA DE TESTES NUMÉRICOS DA DINÂMICA TURBULENTA")
    print("=" * 70)

    if args.teste in ["todos", "convergencia_dt"]:
        executar_teste_convergencia_dt(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "convergencia_N"]:
        executar_teste_convergencia_N(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "estocasticidade"]:
        executar_teste_estocasticidade(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "estatisticas_temporais"]:
        executar_teste_estatisticas_temporais(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "difusao"]:
        executar_teste_difusao(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "comparacao"]:
        executar_teste_comparacao_simuladores(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "transicao_temporal"]:
        executar_teste_transicao_temporal(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "distribuicoes"]:
        executar_teste_distribuicoes(N_particulas=args.N_particulas, t_max=args.t_max, seed=args.seed)

    if args.teste in ["todos", "robustez"]:
        executar_teste_robustez(N_fields=3, t_max=args.t_max, seed=args.seed)

    print("\nExecução concluída! Todos os resultados e gráficos foram salvos na pasta 'resultados/'.\n")


if __name__ == "__main__":
    main()
