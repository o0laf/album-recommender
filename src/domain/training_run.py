"""Clase de dominio: un entrenamiento completo de la red (los ajustes usados y cómo terminó)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingRun:
    dim: int
    batch: int
    neg_ratio: int
    neg_power: float
    lr: float
    epochs_run: int
    best_epoch: int
    n_train_users: int
    n_albums: int
    n_val_users_evaluated: int
    n_test_users_evaluated: int
    run_date: str = ""          # "2026-11-17 21:30"
    id: Optional[int] = None
