"""Punto de entrada: SOLO dispara cada parte del sistema. La lógica está en services/ y controllers/.

Uso (desde training/):
    python main.py preprocess                  # lee los datos y arma el caché
    python main.py baseline                    # resultado de recomendar los populares
    python main.py train                       # entrena, evalúa y guarda todo en models/
    python main.py evaluate                    # re-mide precision/recall/F1 sin reentrenar
    python main.py experiments [--solo base dim64] [--max-epochs 3]
    python main.py names                       # títulos para buscar álbumes por nombre
    python main.py recommend CODIGO1 CODIGO2   # recomienda desde la consola
    python main.py server                      # abre http://127.0.0.1:8000
    python main.py coverage ../data/vynl/database.sqlite
"""
import argparse
import sys


def build_parser():
    p = argparse.ArgumentParser(description="Recomendador de álbumes (Vynl)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="leer los datos y armar el caché")
    sub.add_parser("baseline", help="resultado de recomendar siempre los populares")
    sub.add_parser("train", help="entrenar, evaluar y guardar en models/")
    sub.add_parser("evaluate", help="re-medir precision/recall/F1 sin reentrenar")

    e = sub.add_parser("experiments", help="probar varias configuraciones")
    e.add_argument("--solo", nargs="+", help="nombres de las configuraciones a correr")
    e.add_argument("--max-epochs", type=int, default=None)

    sub.add_parser("names", help="armar los títulos para buscar por nombre")

    r = sub.add_parser("recommend", help="recomendar desde la consola")
    r.add_argument("albums", nargs="+", help="uno o más códigos de álbum (asin)")
    r.add_argument("-k", type=int, default=10)

    sub.add_parser("server", help="abrir la web local")

    c = sub.add_parser("coverage", help="cobertura del catálogo de Vynl")
    c.add_argument("sqlite", help="ruta a database.sqlite de Vynl")
    return p


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # para que no explote con acentos
    except Exception:
        pass
    args = build_parser().parse_args()

    # Los imports van adentro de cada comando: así "preprocess" no carga TensorFlow.
    if args.command == "preprocess":
        from services.data_service import DataService
        DataService().build_data()
    elif args.command == "baseline":
        from services.evaluation_service import EvaluationService
        EvaluationService().run_baseline()
    elif args.command == "train":
        from services.training_service import TrainingService
        TrainingService().run()
    elif args.command == "evaluate":
        from services.evaluation_service import EvaluationService
        EvaluationService().run_evaluation()
    elif args.command == "experiments":
        from config import MAX_EPOCHS
        from services.experiment_service import ExperimentService
        ExperimentService().run(solo=args.solo, max_epochs=args.max_epochs or MAX_EPOCHS)
    elif args.command == "names":
        from services.catalog_service import CatalogService
        CatalogService.build_files()
    elif args.command == "recommend":
        from services.recommendation_service import RecommendationService
        RecommendationService().run_cli(args.albums, args.k)
    elif args.command == "server":
        from controllers.web_controller import WebController
        WebController().run()
    elif args.command == "coverage":
        from services.coverage_service import CoverageService
        CoverageService().run(args.sqlite)


if __name__ == "__main__":
    main()