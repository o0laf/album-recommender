"""Capa de servicios: entrena la red, la evalúa contra el baseline y guarda todo en models/."""
import json
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")   # dibujar a archivo, sin ventana
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from config import (BATCH, DIM, K, LR, MAX_EPOCHS, MODELS_DIR, NEG_POWER, NEG_RATIO,
                    PATIENCE, SEED)
from model import build_model
from services.data_service import DataService
from services.evaluation_service import EvaluationService
from services.recommendation_service import RecommendationService

normalize = RecommendationService.normalize
hit_rate_at_k = RecommendationService.hit_rate_at_k


class TrainingService:

    def __init__(self):
        self.evaluation = EvaluationService()

    def make_epoch(self, data, rng, neg_probs, neg_ratio):
        """Cada época: los datos reales + negativos nuevos sorteados."""
        pos_users = data.train_users[data.train_labels == 1]
        neg_users = np.repeat(pos_users, neg_ratio)
        neg_albums = rng.choice(data.n_albums, size=len(neg_users), p=neg_probs).astype(np.int32)
        users = np.concatenate([data.train_users, neg_users])
        albums = np.concatenate([data.train_albums, neg_albums])
        labels = np.concatenate([data.train_labels, np.zeros(len(neg_users), dtype=np.float32)])
        return users.reshape(-1, 1), albums.reshape(-1, 1), labels

    def train_model(self, data, dim=DIM, batch=BATCH, neg_ratio=NEG_RATIO, neg_power=NEG_POWER,
                    lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE, k=K, seed=SEED, verbose=True):
        """Entrena con early stopping y deja el modelo con los pesos de la mejor época.
        La mejor época se elige con VALIDACIÓN (nunca con test).
        Lo usan run() y ExperimentService."""
        keras.utils.set_random_seed(seed)
        rng = np.random.default_rng(seed)

        # Probabilidad de sortear cada álbum como negativo
        w = (data.popularity + 1.0) ** neg_power
        neg_probs = w / w.sum()

        model = build_model(data.n_users, data.n_albums, dim, lr)
        history = {"loss": [], "val_hit_rate": []}
        best_hr, best_epoch, best_weights, wait = -1.0, 0, None, 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            users, albums, labels = self.make_epoch(data, rng, neg_probs, neg_ratio)
            h = model.fit([users, albums], labels, batch_size=batch, epochs=1, shuffle=True, verbose=0)
            loss = float(h.history["loss"][0])

            emb_norm = normalize(model.get_layer("album_emb").get_weights()[0])
            val_hr = hit_rate_at_k(emb_norm, data.val_inputs, data.val_targets, k)
            history["loss"].append(loss)
            history["val_hit_rate"].append(val_hr)
            if verbose:
                print(f"Época {epoch:2d} | loss {loss:.4f} | hit rate@{k} validación {val_hr:.4f} | {time.time() - t0:.0f}s")

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
        MODELS_DIR.mkdir(exist_ok=True)
        model.save(str(MODELS_DIR / "model.keras"))
        with open(MODELS_DIR / "idx_to_album.json", "w", encoding="utf-8") as f:
            json.dump(data.idx_to_album, f)
        with open(MODELS_DIR / "album_to_idx.json", "w", encoding="utf-8") as f:
            json.dump({a: i for i, a in enumerate(data.idx_to_album)}, f)

    def save_plots(self, history, base_test, test_hr, rank_test):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["loss"]); ax[0].set_title("Loss de entrenamiento"); ax[0].set_xlabel("época")
        ax[1].plot(history["val_hit_rate"]); ax[1].set_title("Hit rate@10 en validación"); ax[1].set_xlabel("época")
        fig.tight_layout(); fig.savefig(MODELS_DIR / "training_curves.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Populares", "Red"], [base_test, test_hr], color=["gray", "tab:blue"])
        ax.set_title("Hit rate@10 en test"); ax.set_ylabel("hit rate")
        fig.tight_layout(); fig.savefig(MODELS_DIR / "hit_rate_comparison.png", dpi=120); plt.close(fig)

        # Precision, recall y F1 lado a lado
        fig, ax = plt.subplots(figsize=(6, 4))
        names = ["precision", "recall", "f1"]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [rank_test["baseline"][n] for n in names], 0.4, label="Populares", color="gray")
        ax.bar(x + 0.2, [rank_test["model"][n] for n in names], 0.4, label="Red", color="tab:blue")
        ax.set_xticks(x); ax.set_xticklabels(["precision@10", "recall@10", "F1@10"])
        ax.set_title("Métricas en test"); ax.legend()
        fig.tight_layout(); fig.savefig(MODELS_DIR / "ranking_metrics.png", dpi=120); plt.close(fig)

    def run(self):
        """Entrenamiento completo: datos > entrenar > evaluar en test > guardar."""
        data = DataService().build_data()
        run = self.train_model(data)
        model, history, best_hr = run.model, run.history, run.best_hr
        emb_norm = normalize(model.get_layer("album_emb").get_weights()[0])
        ev = self.evaluation

        # --- Evaluación final en test (usuarios que la red nunca vio) ---
        test_hr = hit_rate_at_k(emb_norm, data.test_inputs, data.test_targets, K)
        base_val = ev.baseline_hit_rate(data.popularity, data.val_inputs, data.val_targets, K)
        base_test = ev.baseline_hit_rate(data.popularity, data.test_inputs, data.test_targets, K)
        top10 = [data.idx_to_album[i] for i in ev.popular_ranking(data.popularity)[:K]]

        print(f"\nTEST  | hit rate@{K} | red: {test_hr:.4f} | baseline populares: {base_test:.4f}")

        # Precision / recall / F1 en validación y en test
        rank_val = ev.evaluate_ranking(emb_norm, data.popularity, data.val_inputs, data.val_relevant, K)
        rank_test = ev.evaluate_ranking(emb_norm, data.popularity, data.test_inputs, data.test_relevant, K)
        print("\nTEST  | precision, recall y F1:")
        ev.print_table(rank_test, K)

        # --- Guardar ---
        self.save_model_files(model, data)
        metrics = {
            "hit_rate_at_10": {
                "model_val": best_hr, "model_test": test_hr,
                "baseline_val": base_val, "baseline_test": base_test,
            },
            "ranking_at_10": {
                "val": rank_val, "test": rank_test,
                "n_val_users_evaluated": int(len(data.val_inputs)),
                "n_test_users_evaluated": int(len(data.test_inputs)),
            },
            "n_train_users": data.n_users, "n_albums": data.n_albums,
            "n_val_users_evaluated": int(len(data.val_inputs)),
            "n_test_users_evaluated": int(len(data.test_inputs)),
            "epochs_run": len(history["loss"]),
            "best_epoch": run.best_epoch,
            "settings": {"dim": DIM, "batch": BATCH, "neg_ratio": NEG_RATIO, "neg_power": NEG_POWER, "lr": LR},
            "history": history,
            "baseline_top10_asins": top10,   # códigos para probar "recommend"
        }
        with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self.save_plots(history, base_test, test_hr, rank_test)
        print(f"Guardé modelo, diccionarios, metrics.json y gráficos en {MODELS_DIR}")