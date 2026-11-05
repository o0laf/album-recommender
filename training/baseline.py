"""Baseline: recomendar siempre los 10 álbumes más populares (más "me gusta" en train)."""
import numpy as np

from preprocessing import build_data


def popular_ranking(popularity):
    return np.argsort(-popularity, kind="stable")   # de más a menos popular


def baseline_hit_rate(popularity, inputs, targets, k=10):
    ranking = popular_ranking(popularity)[: k + inputs.shape[1]].tolist()
    hits = 0
    for ins, target in zip(inputs, targets):
        ya_puestos = set(ins.tolist())
        recs = [x for x in ranking if x not in ya_puestos][:k]   # no recomendar lo que ya puso
        hits += int(target in recs)
    return hits / len(inputs)


if __name__ == "__main__":
    data = build_data()
    top10 = [data.idx_to_album[i] for i in popular_ranking(data.popularity)[:10]]
    print("Top 10 populares:", top10)
    print(f"Hit rate@10 validación: {baseline_hit_rate(data.popularity, data.val_inputs, data.val_targets):.4f}")
    print(f"Hit rate@10 test:       {baseline_hit_rate(data.popularity, data.test_inputs, data.test_targets):.4f}")
