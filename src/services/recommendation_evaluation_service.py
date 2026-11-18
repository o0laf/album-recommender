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

from config import K
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

    def baseline_hit_rate(self, popularity, inputs, targets, k=K):
        ranking = self.popular_ranking(popularity)[: k + inputs.shape[1]].tolist()
        hits = 0
        for ins, target in zip(inputs, targets):
            ya_puestos = set(ins.tolist())
            recs = [x for x in ranking if x not in ya_puestos][:k]   # no recomendar lo que ya puso
            hits += int(target in recs)
        return hits / len(inputs)

    def top_k_popular(self, popularity, inputs, k=K):
        """Baseline: los k más populares, saltando los que el usuario ya puso."""
        ranking = self.popular_ranking(popularity)[: k + inputs.shape[1]].tolist()
        out = []
        for ins in inputs:
            ya_puestos = set(ins.tolist())
            out.append([x for x in ranking if x not in ya_puestos][:k])
        return np.array(out)

    # ---------- Red ----------

    @staticmethod
    def top_k_model(emb_norm, inputs, k=K, batch=500):
        """Los k álbumes recomendados por la red para cada usuario (matriz usuarios x k)."""
        out = []
        for start in range(0, len(inputs), batch):           # por tandas, para no llenar la RAM
            ins = inputs[start:start + batch]
            q = AlbumRecommendationService.normalize(emb_norm[ins].mean(axis=1))   # promedio de los 4 álbumes
            scores = q @ emb_norm.T
            scores[np.arange(len(ins))[:, None], ins] = -np.inf   # no recomendar los 4 dados
            out.append(np.argpartition(scores, -k, axis=1)[:, -k:])
        return np.vstack(out)

    # ---------- Métricas ----------

    @staticmethod
    def precision_recall_f1(top, relevants, k=K):
        """Promedia precision, recall y F1 entre usuarios.
        top: matriz usuarios x k. relevants: lista con los álbumes que le gustaron a cada uno."""
        precisions, recalls, f1s = [], [], []
        for recs, rel in zip(top, relevants):
            hits = int(np.isin(recs, rel).sum())
            p = hits / k
            r = hits / len(rel)
            f1 = 2 * p * r / (p + r) if hits > 0 else 0.0     # sin aciertos: F1 = 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return {
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
        }

    def evaluate_ranking(self, emb_norm, popularity, inputs, relevants, k=K):
        """Devuelve las métricas de la red y del baseline de populares."""
        return {
            "model": self.precision_recall_f1(self.top_k_model(emb_norm, inputs, k), relevants, k),
            "baseline": self.precision_recall_f1(self.top_k_popular(popularity, inputs, k), relevants, k),
        }

    @staticmethod
    def print_table(res, k=K):
        print(f"              precision@{k}  recall@{k}   F1@{k}")
        for name, label in (("model", "Red     "), ("baseline", "Populares")):
            m = res[name]
            print(f"  {label}   {m['precision']:.4f}       {m['recall']:.4f}     {m['f1']:.4f}")

    @staticmethod
    def to_metrics(subset, res, k=K):
        """Pasa el resultado de evaluate_ranking() a una lista de Metric (para guardar en la base)."""
        return [Metric(subset=subset, source=source, name=name, value=res[source][name], k=k)
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
        art = AlbumRecommendationService().art
        res = self.evaluate_ranking(art.emb_norm, data.popularity, data.test_inputs, data.test_relevant)
        print("\nTEST (usuarios que la red nunca vio):")
        self.print_table(res)

        # Se guarda en la base, sobre el último entrenamiento (si ya había esas métricas, se pisan)
        self.metric_dao.save_all(run.id, self.to_metrics("test", res))
        print(f"Métricas guardadas en la base (entrenamiento #{run.id}).")
        print(f"Guardé las métricas en {path}")