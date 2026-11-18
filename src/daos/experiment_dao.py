"""DAO de experimentos: guarda y lee la tabla `experiment`. Reemplaza a experiments.csv."""
from database.database_connection import DatabaseConnection
from domain.experiment import Experiment

COLUMNS = ("name", "run_date", "dim", "batch", "neg_ratio", "neg_power", "lr", "epochs_run",
           "best_epoch", "val_hit10", "test_hit10", "test_precision10", "test_recall10",
           "test_f1_10", "minutes")


class ExperimentDAO:

    def __init__(self, db=None):
        self.db = db or DatabaseConnection()

    @staticmethod
    def _to_experiment(row):
        return Experiment(**dict(row))

    def insert(self, experiment):
        marks = ", ".join("?" for _ in COLUMNS)
        values = [getattr(experiment, c) for c in COLUMNS]
        with self.db.transaction() as conn:
            cur = conn.execute(f"INSERT INTO experiment ({', '.join(COLUMNS)}) VALUES ({marks})", values)
            experiment.id = cur.lastrowid
        return experiment

    def find_all(self):
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM experiment ORDER BY id").fetchall()
        return [self._to_experiment(r) for r in rows]

    def find_names(self):
        """Nombres de las configuraciones que ya se corrieron."""
        with self.db.transaction() as conn:
            return {r["name"] for r in conn.execute("SELECT name FROM experiment")}

    def find_best_by_validation(self):
        """La mejor configuración según VALIDACIÓN (nunca según test)."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM experiment ORDER BY val_hit10 DESC, id LIMIT 1").fetchone()
        return self._to_experiment(row) if row else None
