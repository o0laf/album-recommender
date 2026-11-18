"""Capa de servicios: pasa a la base los archivos de la versión anterior (los .json y .csv de models/),
para no perder resultados que ya tardaste horas en conseguir.

Lee de artifacts/ (la carpeta que antes se llamaba models/):
  idx_to_album.json, album_to_idx.json, album_titles.json, album_pop.json  >  tabla album
  experiments.csv                                                          >  tabla experiment
  metrics.json                                                             >  tablas training_run, epoch_record y metric
No borra ni cambia ningún archivo, y no pisa lo que ya esté en la base."""
import csv
import json
from datetime import datetime
from pathlib import Path

from config import ARTIFACTS_DIR, BATCH, DIM, LR, NEG_POWER, NEG_RATIO
from daos.album_dao import AlbumDAO
from daos.experiment_dao import ExperimentDAO
from daos.metric_dao import MetricDAO
from daos.training_run_dao import TrainingRunDAO
from domain.album import Album
from domain.epoch_record import EpochRecord
from domain.experiment import Experiment
from domain.metric import Metric
from domain.training_run import TrainingRun


class LegacyArtifactsImportService:

    def __init__(self, folder=None, album_dao=None, experiment_dao=None, training_run_dao=None, metric_dao=None):
        self.folder = Path(folder or ARTIFACTS_DIR)
        self.album_dao = album_dao or AlbumDAO()
        self.experiment_dao = experiment_dao or ExperimentDAO()
        self.training_run_dao = training_run_dao or TrainingRunDAO()
        self.metric_dao = metric_dao or MetricDAO()

    def read_json(self, name):
        path = self.folder / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ---------- Álbumes ----------

    def import_albums(self):
        asins = self.read_json("idx_to_album.json")
        if asins is None:
            to_idx = self.read_json("album_to_idx.json")
            if to_idx is None:
                return False
            asins = [a for a, _ in sorted(to_idx.items(), key=lambda kv: kv[1])]
        if self.album_dao.count_in_model() > 0:
            print("Álbumes: la base ya tiene los álbumes del modelo, no los piso.")
            return True

        self.album_dao.save_vocabulary(asins)
        titles = self.read_json("album_titles.json") or {}
        pop = self.read_json("album_pop.json") or {}
        if titles:
            self.album_dao.replace_catalog(
                [Album(asin=a, title=t, popularity=int(pop.get(a, 0))) for a, t in titles.items()])
        print(f"Álbumes: {len(asins):,} del modelo y {len(titles):,} con título.")
        return True

    # ---------- Experimentos ----------

    def import_experiments(self):
        path = self.folder / "experiments.csv"
        if not path.exists():
            return False
        ya_estan = self.experiment_dao.find_names()
        nuevos = 0
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["nombre"] in ya_estan:
                    continue
                self.experiment_dao.insert(Experiment(
                    name=r["nombre"], run_date=r["fecha"],
                    dim=int(r["dim"]), batch=int(r["batch"]), neg_ratio=int(r["neg_ratio"]),
                    neg_power=float(r["neg_power"]), lr=float(r["lr"]),
                    epochs_run=int(r["epocas"]), best_epoch=int(r["mejor_epoca"]),
                    val_hit10=float(r["val_hit10"]), test_hit10=float(r["test_hit10"]),
                    test_precision10=float(r["test_precision10"]), test_recall10=float(r["test_recall10"]),
                    test_f1_10=float(r["test_f1_10"]), minutes=float(r["minutos"])))
                nuevos += 1
        print(f"Experimentos: {nuevos} importados.")
        return True

    # ---------- Entrenamiento y métricas ----------

    def import_training(self):
        path = self.folder / "metrics.json"
        m = self.read_json("metrics.json")
        if m is None:
            return False
        if self.training_run_dao.find_latest() is not None:
            print("Entrenamiento: la base ya tiene un entrenamiento, no lo piso.")
            return True

        s = m.get("settings", {})
        ranking = m.get("ranking_at_10", {})
        history = m.get("history", {})
        epochs = [EpochRecord(epoch=i, loss=float(loss), val_hit_rate=float(hr))
                  for i, (loss, hr) in enumerate(zip(history.get("loss", []), history.get("val_hit_rate", [])), 1)]
        run = self.training_run_dao.insert(
            TrainingRun(
                dim=int(s.get("dim", DIM)), batch=int(s.get("batch", BATCH)),
                neg_ratio=int(s.get("neg_ratio", NEG_RATIO)), neg_power=float(s.get("neg_power", NEG_POWER)),
                lr=float(s.get("lr", LR)),
                epochs_run=int(m.get("epochs_run", len(epochs))), best_epoch=int(m.get("best_epoch", 0)),
                n_train_users=int(m.get("n_train_users", 0)), n_albums=int(m.get("n_albums", 0)),
                n_val_users_evaluated=int(m.get("n_val_users_evaluated", ranking.get("n_val_users_evaluated", 0))),
                n_test_users_evaluated=int(m.get("n_test_users_evaluated", ranking.get("n_test_users_evaluated", 0))),
                run_date=datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")),
            epochs)

        metrics = []
        hit = m.get("hit_rate_at_10", {})
        for subset, source, key in (("val", "model", "model_val"), ("test", "model", "model_test"),
                                    ("val", "baseline", "baseline_val"), ("test", "baseline", "baseline_test")):
            if key in hit:
                metrics.append(Metric(subset=subset, source=source, name="hit_rate", value=float(hit[key]), k=10))
        for subset in ("val", "test"):
            res = ranking.get(subset) or {}
            for source in ("model", "baseline"):
                for name in ("precision", "recall", "f1"):
                    if name in res.get(source, {}):
                        metrics.append(Metric(subset=subset, source=source, name=name,
                                              value=float(res[source][name]), k=10))
        self.metric_dao.save_all(run.id, metrics)
        print(f"Entrenamiento: 1 importado (#{run.id}) con {len(epochs)} épocas y {len(metrics)} métricas.")
        return True

    def run(self):
        print(f"Buscando archivos de la versión anterior en {self.folder}")
        found = [self.import_albums(), self.import_experiments(), self.import_training()]
        if not any(found):
            print("No encontré ningún archivo para importar (idx_to_album.json, experiments.csv, metrics.json...).")
            return
        print("Listo. Ya podés borrar de artifacts/ los .json y .csv que se importaron (el modelo y los .png quedan).")
