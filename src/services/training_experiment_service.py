"""Capa de servicios: prueba varias configuraciones y deja un registro en la tabla `experiment` de la base.
Esto es el "feedback loop" de la consigna: entrenar > evaluar > si no mejora, cambiar un ajuste > reentrenar.

- Cada configuración cambia UN solo ajuste respecto de "base" (así se ve qué lo causó).
- Si una configuración ya está registrada, se saltea (podés cortar y seguir otro día).
- La MEJOR se elige con la métrica de VALIDACIÓN. Test se guarda solo para informar:
  si eligiéramos por test, estaríamos "espiando" el examen final.
- No pisa artifacts/model.keras: el modelo final sigue siendo el de AlbumModelTrainingService.run()."""
import time
from datetime import datetime

from config import BATCH, DIM, K, LR, MAX_EPOCHS, NEG_POWER, NEG_RATIO, PATIENCE
from daos.experiment_dao import ExperimentDAO
from domain.experiment import Experiment
from services.album_model_training_service import AlbumModelTrainingService
from services.album_recommendation_service import AlbumRecommendationService
from services.ratings_dataset_service import RatingsDatasetService
from services.recommendation_evaluation_service import RecommendationEvaluationService

BASE = {"dim": DIM, "batch": BATCH, "neg_ratio": NEG_RATIO, "neg_power": NEG_POWER, "lr": LR}

# Editá esta lista para probar otras cosas
CONFIGS = [
    {"name": "base", **BASE},
    {"name": "dim16", **BASE, "dim": 16},
    {"name": "dim64", **BASE, "dim": 64},
    {"name": "neg2", **BASE, "neg_ratio": 2},
    {"name": "neg8", **BASE, "neg_ratio": 8},
    {"name": "power0", **BASE, "neg_power": 0.0},
    {"name": "power1", **BASE, "neg_power": 1.0},
    {"name": "batch1024", **BASE, "batch": 1024},
]


class TrainingExperimentService:

    def __init__(self, experiment_dao=None):
        self.experiment_dao = experiment_dao or ExperimentDAO()
        self.training = AlbumModelTrainingService()
        self.evaluation = RecommendationEvaluationService()

    def run_config(self, data, cfg, max_epochs):
        """Entrena una configuración, la mide en test y devuelve el Experiment (todavía sin guardar)."""
        t0 = time.time()
        run = self.training.train_model(
            data, dim=cfg["dim"], batch=cfg["batch"], neg_ratio=cfg["neg_ratio"],
            neg_power=cfg["neg_power"], lr=cfg["lr"], max_epochs=max_epochs,
            patience=PATIENCE, k=K, verbose=True)
        emb_norm = AlbumRecommendationService.normalize(run.model.get_layer("album_emb").get_weights()[0])
        test_hit = AlbumRecommendationService.hit_rate_at_k(emb_norm, data.test_inputs, data.test_targets, K)
        rank = self.evaluation.evaluate_ranking(
            emb_norm, data.popularity, data.test_inputs, data.test_relevant, K)["model"]
        return Experiment(
            name=cfg["name"], run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            dim=cfg["dim"], batch=cfg["batch"], neg_ratio=cfg["neg_ratio"],
            neg_power=cfg["neg_power"], lr=cfg["lr"],
            epochs_run=len(run.history["loss"]), best_epoch=run.best_epoch,
            val_hit10=round(run.best_hr, 4), test_hit10=round(test_hit, 4),
            test_precision10=round(rank["precision"], 4),
            test_recall10=round(rank["recall"], 4),
            test_f1_10=round(rank["f1"], 4),
            minutes=round((time.time() - t0) / 60, 1))

    def run(self, solo=None, max_epochs=MAX_EPOCHS):
        """Corre las configuraciones (todas, o solo las nombradas en `solo`)."""
        ya_corridas = self.experiment_dao.find_names()
        pendientes = [c for c in CONFIGS if (not solo or c["name"] in solo)]
        data = RatingsDatasetService().build_data()   # los datos se arman una sola vez para todas las pruebas

        for n, cfg in enumerate(pendientes, 1):
            if cfg["name"] in ya_corridas:
                print(f"[{n}/{len(pendientes)}] {cfg['name']}: ya está registrada, la salteo")
                continue
            print(f"[{n}/{len(pendientes)}] {cfg['name']}: {({k: v for k, v in cfg.items() if k != 'name'})}")
            exp = self.run_config(data, cfg, max_epochs)
            self.experiment_dao.insert(exp)
            print(f"    validación {exp.val_hit10} | test hit {exp.test_hit10} | "
                  f"F1 {exp.test_f1_10} | {exp.minutes} min")

        # Resumen: la mejor según VALIDACIÓN
        mejor = self.experiment_dao.find_best_by_validation()
        if mejor:
            print(f"\nMejor configuración según validación: {mejor.name} "
                  f"(val {mejor.val_hit10}, test {mejor.test_hit10}, F1 {mejor.test_f1_10})")
            print("Registro completo en la base de datos (tabla experiment).")
