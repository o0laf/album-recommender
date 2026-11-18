"""Base de datos SQLite del proyecto: abre las conexiones y crea las tablas si no existen.
Solo los DAOs usan esta clase; ningún otro archivo escribe SQL."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS album (
    asin        TEXT PRIMARY KEY,
    model_index INTEGER UNIQUE,              -- posición en la red (NULL = la red no lo conoce)
    title       TEXT,
    popularity  INTEGER NOT NULL DEFAULT 0   -- cantidad de calificaciones de 4-5 estrellas
);

CREATE TABLE IF NOT EXISTS training_run (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date               TEXT    NOT NULL,
    dim                    INTEGER NOT NULL,
    batch                  INTEGER NOT NULL,
    neg_ratio              INTEGER NOT NULL,
    neg_power              REAL    NOT NULL,
    lr                     REAL    NOT NULL,
    epochs_run             INTEGER NOT NULL,
    best_epoch             INTEGER NOT NULL,
    n_train_users          INTEGER NOT NULL,
    n_albums               INTEGER NOT NULL,
    n_val_users_evaluated  INTEGER NOT NULL,
    n_test_users_evaluated INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS epoch_record (
    training_run_id INTEGER NOT NULL REFERENCES training_run(id) ON DELETE CASCADE,
    epoch           INTEGER NOT NULL,
    loss            REAL    NOT NULL,
    val_hit_rate    REAL    NOT NULL,
    PRIMARY KEY (training_run_id, epoch)
);

CREATE TABLE IF NOT EXISTS metric (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    training_run_id INTEGER NOT NULL REFERENCES training_run(id) ON DELETE CASCADE,
    subset          TEXT    NOT NULL CHECK (subset IN ('val', 'test')),
    source          TEXT    NOT NULL CHECK (source IN ('model', 'baseline')),
    name            TEXT    NOT NULL,
    k               INTEGER NOT NULL,
    value           REAL    NOT NULL,
    UNIQUE (training_run_id, subset, source, name, k)
);

CREATE TABLE IF NOT EXISTS experiment (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL UNIQUE,
    run_date         TEXT    NOT NULL,
    dim              INTEGER NOT NULL,
    batch            INTEGER NOT NULL,
    neg_ratio        INTEGER NOT NULL,
    neg_power        REAL    NOT NULL,
    lr               REAL    NOT NULL,
    epochs_run       INTEGER NOT NULL,
    best_epoch       INTEGER NOT NULL,
    val_hit10        REAL    NOT NULL,
    test_hit10       REAL    NOT NULL,
    test_precision10 REAL    NOT NULL,
    test_recall10    REAL    NOT NULL,
    test_f1_10       REAL    NOT NULL,
    minutes          REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS vynl_coverage (
    vynl_album_id TEXT PRIMARY KEY,
    title         TEXT,
    artist        TEXT,
    matched_asins TEXT NOT NULL DEFAULT ''   -- códigos separados por espacio; vacío = no está en el modelo
);
"""


class DatabaseConnection:

    def __init__(self, path=None):
        self.path = Path(path or config.DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self):
        """Abre una conexión, guarda los cambios si todo salió bien (o los deshace si hubo error) y la cierra."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
