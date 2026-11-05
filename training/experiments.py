"""Prueba varias configuraciones y deja un registro en models/experiments.csv.
Esto es el "feedback loop" de la consigna: entrenar > evaluar > si no mejora, cambiar un ajuste > reentrenar.

Uso (desde training/):
    python experiments.py                       # corre todas las configuraciones de abajo
    python experiments.py --solo base dim64     # corre solo esas
    python experiments.py --max-epochs 3        # prueba rápida (para ver que anda)

- Cada configuración cambia UN solo ajuste respecto de "base" (así se ve qué lo causó).
- Si una configuración ya está en el csv, se saltea (podés cortar y seguir otro día).
- La MEJOR se elige con la métrica de VALIDACIÓN. Test se guarda solo para informar:
  si eligiéramos por test, estaríamos "espiando" el examen final.
- No pisa models/model.keras: el modelo final sigue siendo el de train.py.
"""
import argparse
import csv
import time
from datetime import datetime

from evaluate import evaluate_ranking
from preprocessing import MODELS_DIR, build_data
from recommend import hit_rate_at_k, normalize
from train import BATCH, DIM, K, LR, MAX_EPOCHS, NEG_POWER, NEG_RATIO, PATIENCE, train_model

CSV_PATH = MODELS_DIR / "experiments.csv"
COLUMNS = ["fecha", "nombre", "dim", "batch", "neg_ratio", "neg_power", "lr",
           "epocas", "mejor_epoca", "val_hit10", "test_hit10",
           "test_precision10", "test_recall10", "test_f1_10", "minutos"]

BASE = {"dim": DIM, "batch": BATCH, "neg_ratio": NEG_RATIO, "neg_power": NEG_POWER, "lr": LR}

# Editá esta lista para probar otras cosas
CONFIGS = [
    {"nombre": "base", **BASE},
    {"nombre": "dim16", **BASE, "dim": 16},
    {"nombre": "dim64", **BASE, "dim": 64},
    {"nombre": "neg2", **BASE, "neg_ratio": 2},
    {"nombre": "neg8", **BASE, "neg_ratio": 8},
    {"nombre": "power0", **BASE, "neg_power": 0.0},
    {"nombre": "power1", **BASE, "neg_power": 1.0},
    {"nombre": "batch1024", **BASE, "batch": 1024},
]


def read_log():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_row(row):
    nuevo = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel lo abre bien
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if nuevo:
            w.writeheader()
        w.writerow(row)


def run_config(data, cfg, max_epochs):
    t0 = time.time()
    run = train_model(data, dim=cfg["dim"], batch=cfg["batch"], neg_ratio=cfg["neg_ratio"],
                      neg_power=cfg["neg_power"], lr=cfg["lr"], max_epochs=max_epochs,
                      patience=PATIENCE, k=K, verbose=True)
    emb_norm = normalize(run.model.get_layer("album_emb").get_weights()[0])
    test_hit = hit_rate_at_k(emb_norm, data.test_inputs, data.test_targets, K)
    rank = evaluate_ranking(emb_norm, data.popularity, data.test_inputs, data.test_relevant, K)["model"]
    return {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "nombre": cfg["nombre"],
        "dim": cfg["dim"], "batch": cfg["batch"], "neg_ratio": cfg["neg_ratio"],
        "neg_power": cfg["neg_power"], "lr": cfg["lr"],
        "epocas": len(run.history["loss"]), "mejor_epoca": run.best_epoch,
        "val_hit10": round(run.best_hr, 4), "test_hit10": round(test_hit, 4),
        "test_precision10": round(rank["precision"], 4),
        "test_recall10": round(rank["recall"], 4),
        "test_f1_10": round(rank["f1"], 4),
        "minutos": round((time.time() - t0) / 60, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solo", nargs="+", help="nombres de las configuraciones a correr")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = p.parse_args()

    ya_corridas = {r["nombre"] for r in read_log()}
    pendientes = [c for c in CONFIGS if (not args.solo or c["nombre"] in args.solo)]
    data = build_data()   # los datos se arman una sola vez para todas las pruebas

    for n, cfg in enumerate(pendientes, 1):
        if cfg["nombre"] in ya_corridas:
            print(f"[{n}/{len(pendientes)}] {cfg['nombre']}: ya está en el csv, la salteo")
            continue
        print(f"[{n}/{len(pendientes)}] {cfg['nombre']}: {({k: v for k, v in cfg.items() if k != 'nombre'})}")
        row = run_config(data, cfg, args.max_epochs)
        append_row(row)
        print(f"    validación {row['val_hit10']} | test hit {row['test_hit10']} | "
              f"F1 {row['test_f1_10']} | {row['minutos']} min")

    # Resumen: la mejor según VALIDACIÓN
    filas = read_log()
    if filas:
        mejor = max(filas, key=lambda r: float(r["val_hit10"]))
        print(f"\nMejor configuración según validación: {mejor['nombre']} "
              f"(val {mejor['val_hit10']}, test {mejor['test_hit10']}, F1 {mejor['test_f1_10']})")
        print(f"Registro completo en {CSV_PATH}")


if __name__ == "__main__":
    main()