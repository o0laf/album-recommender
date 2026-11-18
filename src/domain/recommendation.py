"""Clase de dominio: un álbum recomendado."""
from dataclasses import asdict, dataclass


@dataclass
class Recommendation:
    asin: str
    title: str
    score: float      # similitud coseno con lo que puso la persona (más alto = más parecido)

    def to_dict(self):
        return asdict(self)
