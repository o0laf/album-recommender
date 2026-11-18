"""Clase de dominio: un álbum de Vynl y los códigos del modelo que le corresponden."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VynlCoverageEntry:
    vynl_album_id: str
    title: str
    artist: str
    asins: List[str] = field(default_factory=list)   # vacío = Vynl lo tiene y el modelo no

    @property
    def found(self):
        return bool(self.asins)
