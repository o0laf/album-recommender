"""DAO de métricas: guarda y lee la tabla `metric` (hit rate, precision, recall y F1 de cada entrenamiento)."""
from database.database_connection import DatabaseConnection
from domain.metric import Metric


class MetricDAO:

    def __init__(self, db=None):
        self.db = db or DatabaseConnection()

    def save_all(self, training_run_id, metrics):
        """Guarda las métricas del entrenamiento. Si una ya existía (mismo conjunto, origen, nombre y k),
        la pisa con el valor nuevo."""
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO metric (training_run_id, subset, source, name, k, value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(training_run_id, m.subset, m.source, m.name, m.k, m.value) for m in metrics])

    def find_by_run(self, training_run_id):
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM metric WHERE training_run_id = ? ORDER BY subset, source, name, k",
                (training_run_id,)).fetchall()
        return [Metric(id=r["id"], subset=r["subset"], source=r["source"], name=r["name"],
                       k=r["k"], value=r["value"]) for r in rows]
