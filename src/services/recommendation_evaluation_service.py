"""Capa de servicios: baseline (populares) y métricas precision@K, recall@K, F1@K.

Cómo se mide (para cada usuario de test, que la red NUNCA vio en el entrenamiento):
  1. Le damos 4 álbumes que le gustaron.
  2. Pedimos K recomendaciones (sin repetir los 4).
  3. "Acierto" = una recomendación que a ese usuario también le gustó (4-5 estrellas)
     y que no le dimos como entrada.
  precision@K = aciertos / K                        (de lo que recomendé, ¿cuánto sirvió?)
  recall@K    = aciertos / total de álbumes que le gustaron (¿cuánto de lo que le gusta encontré?)
  F1@K        = el promedio "parejo" de los dos: 2*P*R / (P+R)
Después se promedia entre todos los usuarios."""
import numpy as np

from config import RECOMMENDATION_COUNT
from daos.metric_dao import MetricDAO
from daos.training_run_dao import TrainingRunDAO
from domain.metric import Metric
from services.album_recommendation_service import AlbumRecommendationService
from services.ratings_dataset_service import RatingsDatasetService


class RecommendationEvaluationService:

    def __init__(self, training_run_dao=None, metric_dao=None):
        self.training_run_dao = training_run_dao or TrainingRunDAO()
        self.metric_dao = metric_dao or MetricDAO()

    # ---------- Baseline: los más populares ----------

    @staticmethod
    def popular_ranking(popularity):
        return np.argsort(-popularity, kind="stable")   # de más a menos popular

    def baseline_hit_rate(self, popularity, inputs, targets, k=RECOMMENDATION_COUNT):
        ranking = self.popular_ranking(popularity)[: k + inputs.shape[1]].tolist()
        hits = 0
        for input_albums, target in zip(inputs, targets):
            supplied_albums = set(input_albums.tolist())
            recommendations = [album for album in ranking if album not in supplied_albums][:k]
            hits += int(target in recommendations)
        return hits / len(inputs)

    def top_k_popular(self, popularity, inputs, k=RECOMMENDATION_COUNT):
        """Baseline: los k más populares, saltando los que el usuario ya puso."""
        ranking = self.popular_ranking(popularity)[: k + inputs.shape[1]].tolist()
        rankings = []
        for input_albums in inputs:
            supplied_albums = set(input_albums.tolist())
            rankings.append([album for album in ranking if album not in supplied_albums][:k])
        return np.array(rankings)

    # ---------- Red ----------

    @staticmethod
    def top_k_model(emb_norm, inputs, k=RECOMMENDATION_COUNT, batch=500):
        """Los k álbumes recomendados por la red para cada usuario (matriz usuarios x k)."""
        user_rankings = []
        for start in range(0, len(inputs), batch):           # por tandas, para no llenar la RAM
            input_batch = inputs[start:start + batch]
            query = AlbumRecommendationService.normalize(emb_norm[input_batch].mean(axis=1))
            scores = query @ emb_norm.T
            scores[np.arange(len(input_batch))[:, None], input_batch] = -np.inf
            user_rankings.append(np.argpartition(scores, -k, axis=1)[:, -k:])
        return np.vstack(user_rankings)

    # ---------- Métricas ----------

    @staticmethod
    def precision_recall_f1(rankings, relevants, k=RECOMMENDATION_COUNT):
        """Promedia precision, recall y F1 entre usuarios.
        top: matriz usuarios x k. relevants: lista con los álbumes que le gustaron a cada uno."""
        precisions, recalls, f1s = [], [], []
        for recommendations, relevant_albums in zip(rankings, relevants):
            hits = int(np.isin(recommendations, relevant_albums).sum())
            p = hits / k
            recall = hits / len(relevant_albums)
            f1 = 2 * p * recall / (p + recall) if hits > 0 else 0.0
            precisions.append(p)
            recalls.append(recall)
            f1s.append(f1)
        return {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
        }

    def evaluate_ranking(self, emb_norm, popularity, inputs, relevants, k=RECOMMENDATION_COUNT):
        """Devuelve las métricas de la red y del baseline de populares."""
        return {
            "model": self.precision_recall_f1(self.top_k_model(emb_norm, inputs, k), relevants, k),
            "baseline": self.precision_recall_f1(self.top_k_popular(popularity, inputs, k), relevants, k),
        }

    @staticmethod
    def print_table(results, k=RECOMMENDATION_COUNT):
        print(f"              precision@{k}  recall@{k}   F1@{k}")
        for name, label in (("model", "Red     "), ("baseline", "Populares")):
            metrics = results[name]
            print(f"  {label}   {metrics['precision']:.4f}       {metrics['recall']:.4f}     {metrics['f1']:.4f}")

    @staticmethod
    def to_metrics(subset, results, k=RECOMMENDATION_COUNT):
        """Pasa el resultado de evaluate_ranking() a una lista de Metric (para guardar en la base)."""
        return [Metric(subset=subset, source=source, name=name, value=results[source][name], k=k)
                for source in ("model", "baseline") for name in ("precision", "recall", "f1")]

    # ---------- Comandos que dispara la consola ----------

    def run_baseline(self):
        """Muestra el resultado de recomendar siempre los populares."""
        data = RatingsDatasetService().build_data()
        top10 = [data.idx_to_album[i] for i in self.popular_ranking(data.popularity)[:10]]
        print("Top 10 populares:", top10)
        print(f"Hit rate@10 validación: {self.baseline_hit_rate(data.popularity, data.val_inputs, data.val_targets):.4f}")
        print(f"Hit rate@10 test:       {self.baseline_hit_rate(data.popularity, data.test_inputs, data.test_targets):.4f}")

    def run_evaluation(self):
        """Re-mide precision/recall/F1 sin reentrenar (necesita artifacts/model.keras y un entrenamiento
        registrado en la base). Guarda las métricas nuevas en el último entrenamiento."""
        run = self.training_run_dao.find_latest()
        if run is None:
            raise SystemExit("No hay ningún entrenamiento registrado en la base. "
                             "Corré 'python main.py train' (o 'python main.py import-legacy' si ya tenías resultados).")
        data = RatingsDatasetService().build_data()
        artifacts = AlbumRecommendationService().art
        results = self.evaluate_ranking(
            artifacts.emb_norm,
            data.popularity,
            data.test_inputs,
            data.test_relevant,
        )
        print("\nTEST (usuarios que la red nunca vio):")
        self.print_table(results)

        # Se guarda en la base, sobre el último entrenamiento (si ya había esas métricas, se pisan)
        self.metric_dao.save_all(run.id, self.to_metrics("test", results))
        print(f"Métricas guardadas en la base (entrenamiento #{run.id}).")