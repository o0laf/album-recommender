"""DAO de entrenamientos: guarda y lee las tablas `training_run` y `epoch_record`.
Reemplaza a la parte de metrics.json con los ajustes, las épocas y el historial."""
from database.database_connection import DatabaseConnection
from domain.epoch_record import EpochRecord
from domain.training_run import TrainingRun

RUN_COLUMNS = ("run_date", "dim", "batch", "neg_ratio", "neg_power", "lr", "epochs_run", "best_epoch",
               "n_train_users", "n_albums", "n_val_users_evaluated", "n_test_users_evaluated")


class TrainingRunDAO:

    def __init__(self, db=None):
        self.db = db or DatabaseConnection()

    def insert(self, run, epochs):
        """Guarda el entrenamiento y todas sus épocas juntos (o no guarda nada si algo falla)."""
        marks = ", ".join("?" for _ in RUN_COLUMNS)
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"INSERT INTO training_run ({', '.join(RUN_COLUMNS)}) VALUES ({marks})",
                [getattr(run, c) for c in RUN_COLUMNS])
            run.id = cur.lastrowid
            conn.executemany(
                "INSERT INTO epoch_record (training_run_id, epoch, loss, val_hit_rate) VALUES (?, ?, ?, ?)",
                [(run.id, e.epoch, e.loss, e.val_hit_rate) for e in epochs])
        return run

    def find_latest(self):
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM training_run ORDER BY id DESC LIMIT 1").fetchone()
        return TrainingRun(**dict(row)) if row else None

    def find_epochs(self, run_id):
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT epoch, loss, val_hit_rate FROM epoch_record WHERE training_run_id = ? ORDER BY epoch",
                (run_id,)).fetchall()
        return [EpochRecord(epoch=r["epoch"], loss=r["loss"], val_hit_rate=r["val_hit_rate"]) for r in rows]
