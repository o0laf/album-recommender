"""Clase de dominio: lo que pasó en una época de un entrenamiento."""
from dataclasses import dataclass


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    val_hit_rate: float
