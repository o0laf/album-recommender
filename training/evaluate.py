"""Precision@K, recall@K y F1@K sobre los usuarios de test, y comparación con el baseline.

Cómo se mide (para cada usuario de test, que la red NUNCA vio en el entrenamiento):
  1. Le damos 4 álbumes que le gustaron.
  2. Pedimos K recomendaciones (sin repetir los 4).
  3. "Acierto" = una recomendación que a ese usuario también le gustó (4-5 estrellas)
     y que no le dimos como entrada.
  precision@K = aciertos / K                        (de lo que recomendé, ¿cuánto sirvió?)
  recall@K    = aciertos / total de álbumes que le gustaron (¿cuánto de lo que le gusta encontré?)
  F1@K        = el promedio "parejo" de los dos: 2*P*R / (P+R)
Después se promedia entre todos los usuarios.

Uso para re-evaluar sin volver a entrenar (necesita models/model.keras):
    python evaluate.py
"""
import json

import numpy as np

from preprocessing import MODELS_DIR
from recommend import normalize

K = 10


def top_k_model(emb_norm, inputs, k=K, batch=500):
    """Los k álbumes recomendados por la red para cada usuario (matriz usuarios x k)."""
    out = []
    for start in range(0, len(inputs), batch):           # por tandas, para no llenar la RAM
        ins = inputs[start:start + batch]
        q = normalize(emb_norm[ins].mean(axis=1))        # promedio de los 4 álbumes
        scores = q @ emb_norm.T
        scores[np.arange(len(ins))[:, None], ins] = -np.inf   # no recomendar los 4 dados
        out.append(np.argpartition(scores, -k, axis=1)[:, -k:])
    return np.vstack(out)


def top_k_popular(popularity, inputs, k=K):
    """Baseline: los k más populares, saltando los que el usuario ya puso."""
    ranking = np.argsort(-popularity, kind="stable")[: k + inputs.shape[1]].tolist()
    out = []
    for ins in inputs:
        ya_puestos = set(ins.tolist())
        out.append([x for x in ranking if x not in ya_puestos][:k])
    return np.array(out)


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


def evaluate_ranking(emb_norm, popularity, inputs, relevants, k=K):
    """Devuelve las métricas de la red y del baseline de populares."""
    return {
        "model": precision_recall_f1(top_k_model(emb_norm, inputs, k), relevants, k),
        "baseline": precision_recall_f1(top_k_popular(popularity, inputs, k), relevants, k),
    }


def print_table(res, k=K):
    print(f"              precision@{k}  recall@{k}   F1@{k}")
    for name, label in (("model", "Red     "), ("baseline", "Populares")):
        m = res[name]
        print(f"  {label}   {m['precision']:.4f}       {m['recall']:.4f}     {m['f1']:.4f}")


if __name__ == "__main__":
    from preprocessing import build_data
    from recommend import load_artifacts

    data = build_data()
    art = load_artifacts()
    res = evaluate_ranking(art.emb_norm, data.popularity, data.test_inputs, data.test_relevant)
    print("\nTEST (usuarios que la red nunca vio):")
    print_table(res)

    # Lo sumo a metrics.json sin pisar lo que ya había
    path = MODELS_DIR / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics["ranking_at_10"] = {"test": res, "n_test_users_evaluated": int(len(data.test_inputs))}
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Guardé las métricas en {path}")