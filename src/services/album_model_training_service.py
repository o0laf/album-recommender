"""Capa de servicios: entrena la red, la evalúa contra el baseline y guarda los resultados.
Archivos pesados (modelo y gráficos) > artifacts/. Todo lo demás (álbumes, ajustes, épocas, métricas) > base de datos."""
import time
from datetime import datetime
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")   # dibujar a archivo, sin ventana
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from config import (ARTIFACTS_DIR, EMBEDDING_DIMENSION, LEARNING_RATE, MAX_EPOCHS,
                    NEGATIVE_SAMPLE_RATIO, NEGATIVE_SAMPLING_POWER, PATIENCE,
                    RECOMMENDATION_COUNT, SEED, TRAINING_BATCH_SIZE)
from daos.album_dao import AlbumDAO
from daos.metric_dao import MetricDAO
from daos.training_run_dao import TrainingRunDAO
from domain.epoch_record import EpochRecord
from domain.metric import Metric
from domain.training_run import TrainingRun
from ml.album_embedding_network import AlbumEmbeddingNetwork
from services.album_recommendation_service import AlbumRecommendationService
from services.ratings_dataset_service import RatingsDatasetService
from services.recommendation_evaluation_service import RecommendationEvaluationService

normalize = AlbumRecommendationService.normalize
hit_rate_at_k = AlbumRecommendationService.hit_rate_at_k


class AlbumModelTrainingService:

    def __init__(self, album_dao=None, training_run_dao=None, metric_dao=None):
        self.evaluation = RecommendationEvaluationService()
        self.album_dao = album_dao or AlbumDAO()
        self.training_run_dao = training_run_dao or TrainingRunDAO()
        self.metric_dao = metric_dao or MetricDAO()

    def make_epoch(self, data, rng, neg_probs, neg_ratio):
        """Cada época: los datos reales + negativos nuevos sorteados."""
        pos_users = data.train_users[data.train_labels == 1]
        neg_users = np.repeat(pos_users, neg_ratio)
        neg_albums = rng.choice(data.n_albums, size=len(neg_users), p=neg_probs).astype(np.int32)
        users = np.concatenate([data.train_users, neg_users])
        albums = np.concatenate([data.train_albums, neg_albums])
        labels = np.concatenate([data.train_labels, np.zeros(len(neg_users), dtype=np.float32)])
        return users.reshape(-1, 1), albums.reshape(-1, 1), labels

    def train_model(self, data, dim=EMBEDDING_DIMENSION, batch=TRAINING_BATCH_SIZE,
                    neg_ratio=NEGATIVE_SAMPLE_RATIO, neg_power=NEGATIVE_SAMPLING_POWER,
                    lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                    k=RECOMMENDATION_COUNT, seed=SEED, verbose=True):
        """Entrena con early stopping y deja el modelo con los pesos de la mejor época.
        La mejor época se elige con VALIDACIÓN (nunca con test).
        Lo usan run() y TrainingExperimentService."""
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)

        # Probabilidad de sortear cada álbum como negativo
        w = (data.popularity + 1.0) ** neg_power
        neg_probs = w / w.sum()

        model = AlbumEmbeddingNetwork.build(data.n_users, data.n_albums, dim, lr)
        history = {"loss": [], "val_hit_rate": []}
        best_hr, best_epoch, best_weights, wait = -1.0, 0, None, 0

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()
            users, albums, labels = self.make_epoch(data, rng, neg_probs, neg_ratio)
            h = model.fit([users, albums], labels, batch_size=batch, epochs=1, shuffle=True, verbose=0)
            loss = float(h.history["loss"][0])

            emb_norm = normalize(model.get_layer("album_emb").get_weights()[0])
            val_hr = hit_rate_at_k(emb_norm, data.val_inputs, data.val_targets, k)
            history["loss"].append(loss)
            history["val_hit_rate"].append(val_hr)
            if verbose:
                elapsed_seconds = time.time() - start_time
                print(f"Época {epoch:2d} | loss {loss:.4f} | hit rate@{k} validación {val_hr:.4f} | {elapsed_seconds:.0f}s")

            if val_hr > best_hr:
                best_hr, best_epoch, best_weights, wait = val_hr, epoch, model.get_weights(), 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print("La validación dejó de mejorar: corto el entrenamiento.")
                    break

        model.set_weights(best_weights)   # quedarme con la mejor época
        return SimpleNamespace(model=model, history=history, best_hr=best_hr, best_epoch=best_epoch)

    def save_model_files(self, model, data):
        """El modelo va a artifacts/ (es pesado); qué álbum es cada posición va a la base."""
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        model.save(str(ARTIFACTS_DIR / "model.keras"))
        self.album_dao.save_vocabulary(data.idx_to_album)

    def save_plots(self, history, base_test, test_hr, rank_test):
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["loss"]); ax[0].set_title("Loss de entrenamiento"); ax[0].set_xlabel("época")
        ax[1].plot(history["val_hit_rate"]); ax[1].set_title("Hit rate@10 en validación"); ax[1].set_xlabel("época")
        fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "training_curves.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Populares", "Red"], [base_test, test_hr], color=["gray", "tab:blue"])
        ax.set_title("Hit rate@10 en test"); ax.set_ylabel("hit rate")
        fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "hit_rate_comparison.png", dpi=120); plt.close(fig)

        # Precision, recall y F1 lado a lado
        fig, ax = plt.subplots(figsize=(6, 4))
        names = ["precision", "recall", "f1"]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [rank_test["baseline"][n] for n in names], 0.4, label="Populares", color="gray")
        ax.bar(x + 0.2, [rank_test["model"][n] for n in names], 0.4, label="Red", color="tab:blue")
        ax.set_xticks(x); ax.set_xticklabels(["precision@10", "recall@10", "F1@10"])
        ax.set_title("Métricas en test"); ax.legend()
        fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "ranking_metrics.png", dpi=120); plt.close(fig)

    def run(self):
        """Entrenamiento completo: datos > entrenar > evaluar en test > guardar."""
        data = RatingsDatasetService().build_data()
        run = self.train_model(data)
        model, history, best_hr = run.model, run.history, run.best_hr
        emb_norm = normalize(model.get_layer("album_emb").get_weights()[0])
        ev = self.evaluation

        # --- Evaluación final en test (usuarios que la red nunca vio) ---
        test_hr = hit_rate_at_k(emb_norm, data.test_inputs, data.test_targets, RECOMMENDATION_COUNT)
        base_val = ev.baseline_hit_rate(data.popularity, data.val_inputs, data.val_targets, RECOMMENDATION_COUNT)
        base_test = ev.baseline_hit_rate(data.popularity, data.test_inputs, data.test_targets, RECOMMENDATION_COUNT)
        top10 = [data.idx_to_album[i] for i in ev.popular_ranking(data.popularity)[:RECOMMENDATION_COUNT]]

        print(f"\nTEST  | hit rate@{RECOMMENDATION_COUNT} | red: {test_hr:.4f} | baseline populares: {base_test:.4f}")

        # Precision / recall / F1 en validación y en test
        rank_val = ev.evaluate_ranking(emb_norm, data.popularity, data.val_inputs, data.val_relevant, RECOMMENDATION_COUNT)
        rank_test = ev.evaluate_ranking(emb_norm, data.popularity, data.test_inputs, data.test_relevant, RECOMMENDATION_COUNT)
        print("\nTEST  | precision, recall y F1:")
        ev.print_table(rank_test, RECOMMENDATION_COUNT)

        # --- Guardar: archivos pesados > artifacts/, todo lo demás > base de datos ---
        self.save_model_files(model, data)
        self.save_plots(history, base_test, test_hr, rank_test)

        training_run = self.training_run_dao.insert(
            TrainingRun(
                dim=EMBEDDING_DIMENSION, batch=TRAINING_BATCH_SIZE,
                neg_ratio=NEGATIVE_SAMPLE_RATIO, neg_power=NEGATIVE_SAMPLING_POWER,
                lr=LEARNING_RATE,
                epochs_run=len(history["loss"]), best_epoch=run.best_epoch,
                n_train_users=data.n_users, n_albums=data.n_albums,
                n_val_users_evaluated=int(len(data.val_inputs)),
                n_test_users_evaluated=int(len(data.test_inputs)),
                run_date=datetime.now().strftime("%Y-%m-%d %H:%M")),
            [EpochRecord(epoch=i, loss=loss, val_hit_rate=hr)
             for i, (loss, hr) in enumerate(zip(history["loss"], history["val_hit_rate"]), 1)])

        metrics = [
            Metric(subset="val", source="model", name="hit_rate", value=best_hr, k=RECOMMENDATION_COUNT),
            Metric(subset="test", source="model", name="hit_rate", value=test_hr, k=RECOMMENDATION_COUNT),
            Metric(subset="val", source="baseline", name="hit_rate", value=base_val, k=RECOMMENDATION_COUNT),
            Metric(subset="test", source="baseline", name="hit_rate", value=base_test, k=RECOMMENDATION_COUNT),
        ]
        metrics += ev.to_metrics("val", rank_val, RECOMMENDATION_COUNT) + ev.to_metrics("test", rank_test, RECOMMENDATION_COUNT)
        self.metric_dao.save_all(training_run.id, metrics)

        print(f"\nCódigos de los 10 más populares (para probar 'recommend'): {' '.join(top10)}")
        print(f"Guardé el modelo y los gráficos en {ARTIFACTS_DIR}")
        print(f"Guardé los álbumes, ajustes, épocas y métricas en la base (entrenamiento #{training_run.id}).")
