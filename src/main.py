"""Punto de entrada: SOLO lee el comando y se lo pasa al controlador de consola.
La lógica está en services/, el acceso a la base en daos/ y las clases del problema en domain/.

Uso (desde src/):
    python main.py preprocess                  # lee los datos y arma el caché
    python main.py baseline                    # resultado de recomendar los populares
    python main.py train                       # entrena, evalúa y guarda (modelo > artifacts/, resto > base)
    python main.py evaluate                    # re-mide precision/recall/F1 sin reentrenar
    python main.py experiments [--solo base dim64] [--max-epochs 3]
    python main.py names                       # títulos para buscar álbumes por nombre
    python main.py recommend CODIGO1 CODIGO2   # recomienda desde la consola
    python main.py server                      # abre http://127.0.0.1:8000
    python main.py import-legacy               # pasa a la base los .json/.csv de la versión anterior
"""
import argparse
import sys

from controllers.console_controller import ConsoleController


def build_parser():
    parser = argparse.ArgumentParser(description="Recomendador de álbumes")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("preprocess", help="leer los datos y armar el caché")
    subcommands.add_parser("baseline", help="resultado de recomendar siempre los populares")
    subcommands.add_parser("train", help="entrenar, evaluar y guardar los resultados")
    subcommands.add_parser("evaluate", help="re-medir precision/recall/F1 sin reentrenar")

    experiments_parser = subcommands.add_parser("experiments", help="probar varias configuraciones")
    experiments_parser.add_argument("--solo", nargs="+", help="nombres de las configuraciones a correr")
    experiments_parser.add_argument("--max-epochs", type=int, default=None)

    subcommands.add_parser("names", help="guardar en la base los títulos para buscar por nombre")

    recommend_parser = subcommands.add_parser("recommend", help="recomendar desde la consola")
    recommend_parser.add_argument("albums", nargs="+", help="uno o más códigos de álbum (asin)")
    recommend_parser.add_argument("-k", type=int, default=10)

    subcommands.add_parser("server", help="abrir la web local")

    subcommands.add_parser("import-legacy", help="pasar a la base los .json/.csv de la versión anterior")
    return parser


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # para que no explote con acentos
    except Exception:
        pass
    args = build_parser().parse_args()
    ConsoleController().run(args)


if __name__ == "__main__":
    main()
