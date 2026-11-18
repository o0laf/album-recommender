"""Clase de dominio: una prueba del feedback loop (una configuración entrenada y medida)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Experiment:
    name: str
    dim: int
    batch: int
    neg_ratio: int
    neg_power: float
    lr: float
    epochs_run: int
    best_epoch: int
    val_hit10: float
    test_hit10: float
    test_precision10: float
    test_recall10: float
    test_f1_10: float
    minutes: float
    run_date: str = ""          # "2026-11-17 21:30"
    id: Optional[int] = None
