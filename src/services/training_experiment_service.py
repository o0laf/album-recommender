"""Capa de servicios: prueba varias configuraciones y deja un registro en la tabla `experiment` de la base.
Esto es el "feedback loop" de la consigna: entrenar > evaluar > si no mejora, cambiar un ajuste > reentrenar.

- Cada configuración cambia UN solo ajuste respecto de "base" (así se ve qué lo causó).
- Si una configuración ya está registrada, se saltea (podés cortar y seguir otro día).
- La MEJOR se elige con la métrica de VALIDACIÓN. Test se guarda solo para informar:
  si eligiéramos por test, estaríamos "espiando" el examen final.
- No pisa artifacts/model.keras: el modelo final sigue siendo el de AlbumModelTrainingService.run()."""
import time
from datetime import datetime

from config import (EMBEDDING_DIMENSION, TRAINING_BATCH_SIZE, RECOMMENDATION_COUNT,
                    LEARNING_RATE, MAX_EPOCHS, NEGATIVE_SAMPLING_POWER,
                    NEGATIVE_SAMPLE_RATIO, PATIENCE)
from daos.experiment_dao import ExperimentDAO
from domain.experiment import Experiment
from services.album_model_training_service import AlbumModelTrainingService
from services.album_recommendation_service import AlbumRecommendationService
from services.ratings_dataset_service import RatingsDatasetService
from services.recommendation_evaluation_service import RecommendationEvaluationService

BASE = {
    "dim": EMBEDDING_DIMENSION,
    "batch": TRAINING_BATCH_SIZE,
    "neg_ratio": NEGATIVE_SAMPLE_RATIO,
    "neg_power": NEGATIVE_SAMPLING_POWER,
    "lr": LEARNING_RATE,
}

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

    def run_config(self, data, configuration, max_epochs):
        """Entrena una configuración, la mide en test y devuelve el Experiment (todavía sin guardar)."""
        start_time = time.time()
        run = self.training.train_model(
            data, dim=configuration["dim"], batch=configuration["batch"],
            neg_ratio=configuration["neg_ratio"], neg_power=configuration["neg_power"],
            lr=configuration["lr"], max_epochs=max_epochs,
            patience=PATIENCE, k=RECOMMENDATION_COUNT, verbose=True)
        emb_norm = AlbumRecommendationService.normalize(run.model.get_layer("album_emb").get_weights()[0])
        test_hit = AlbumRecommendationService.hit_rate_at_k(
            emb_norm, data.test_inputs, data.test_targets, RECOMMENDATION_COUNT)
        rank = self.evaluation.evaluate_ranking(
            emb_norm, data.popularity, data.test_inputs, data.test_relevant,
            RECOMMENDATION_COUNT)["model"]
        return Experiment(
            name=configuration["name"], run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            dim=configuration["dim"], batch=configuration["batch"],
            neg_ratio=configuration["neg_ratio"], neg_power=configuration["neg_power"],
            lr=configuration["lr"],
            epochs_run=len(run.history["loss"]), best_epoch=run.best_epoch,
            val_hit10=round(run.best_hr, 4), test_hit10=round(test_hit, 4),
            test_precision10=round(rank["precision"], 4),
            test_recall10=round(rank["recall"], 4),
            test_f1_10=round(rank["f1"], 4),
            minutes=round((time.time() - start_time) / 60, 1))

    def run(self, solo=None, max_epochs=MAX_EPOCHS):
        """Corre las configuraciones (todas, o solo las nombradas en `solo`)."""
        completed_names = self.experiment_dao.find_names()
        pending_configurations = [c for c in CONFIGS if (not solo or c["name"] in solo)]
        data = RatingsDatasetService().build_data()   # los datos se arman una sola vez para todas las pruebas

        for position, configuration in enumerate(pending_configurations, 1):
            if configuration["name"] in completed_names:
                print(f"[{position}/{len(pending_configurations)}] {configuration['name']}: ya está registrada, la salteo")
                continue
            settings = {key: value for key, value in configuration.items() if key != "name"}
            print(f"[{position}/{len(pending_configurations)}] {configuration['name']}: {settings}")
            experiment = self.run_config(data, configuration, max_epochs)
            self.experiment_dao.insert(experiment)
            print(f"    validación {experiment.val_hit10} | test hit {experiment.test_hit10} | "
                  f"F1 {experiment.test_f1_10} | {experiment.minutes} min")

        # Resumen: la mejor según VALIDACIÓN
        best_experiment = self.experiment_dao.find_best_by_validation()
        if best_experiment:
            print(f"\nMejor configuración según validación: {best_experiment.name} "
                  f"(val {best_experiment.val_hit10}, test {best_experiment.test_hit10}, "
                  f"F1 {best_experiment.test_f1_10})")
            print("Registro completo en la base de datos (tabla experiment).")
