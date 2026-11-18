"""Clase de dominio: una métrica medida en un entrenamiento."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Metric:
    """subset: "val" o "test".  source: "model" (la red) o "baseline" (los populares).
    name: "hit_rate", "precision", "recall" o "f1".  k: cuántas recomendaciones se miraron."""
    subset: str
    source: str
    name: str
    value: float
    k: int = 10
    id: Optional[int] = None
