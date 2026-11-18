"""Clase de dominio: un álbum del catálogo."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Album:
    """`asin` es el código de Amazon. `model_index` es la posición del álbum dentro de la red
    (None si la red no lo conoce). `popularity` = cuántos "me gusta" (4-5 estrellas) tiene."""
    asin: str
    model_index: Optional[int] = None
    title: str = ""
    popularity: int = 0
