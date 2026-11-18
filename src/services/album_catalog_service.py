"""Capa de servicios: títulos de los álbumes y búsqueda por nombre."""
import gzip
import html
import json
import unicodedata

import numpy as np

from config import CACHE_PATH, META_PATH
from daos.album_dao import AlbumDAO
from domain.album import Album


class AlbumCatalogService:
    """Búsqueda por nombre. Necesita los títulos (que guarda build_catalog en la base)."""

    def __init__(self, album_dao=None):
        self.album_dao = album_dao or AlbumDAO()
        # Índice de búsqueda, con los más populares primero
        albums = [a for a in self.album_dao.find_all_in_model() if a.title]
        self.index = sorted(
            ((a.popularity, a.asin, a.title, self.simplify(a.title)) for a in albums),
            reverse=True,
        )

    @staticmethod
    def simplify(text):
        """Minúsculas y sin acentos, para que 'rosalia' encuentre 'ROSALÍA'."""
        text = unicodedata.normalize("NFKD", text.lower())
        return "".join(c for c in text if not unicodedata.combining(c))

    @property
    def enabled(self):
        return bool(self.index)

    def search(self, query, limit=8):
        words = self.simplify(query).split()
        if not words:
            return []
        out = []
        for _, asin, title, norm in self.index:
            if all(w in norm for w in words):
                out.append({"asin": asin, "title": title})
                if len(out) >= limit:
                    break
        return out

    def build_catalog(self):
        """Guarda en la base el título y la popularidad de cada álbum que conoce la red:
          - título: "título - artista" (sale de la metadata de Amazon)
          - popularidad: cuántos "me gusta" tiene
        Necesita data/metadata/meta_CDs_and_Vinyl.json.gz y haber entrenado antes."""
        wanted = {a.asin for a in self.album_dao.find_all_in_model()}   # solo guardo los álbumes que conoce el modelo
        if not wanted:
            raise SystemExit("La base no tiene los álbumes del modelo. Corré primero 'python main.py train' "
                             "(o 'python main.py import-legacy').")

        titles = {}
        with gzip.open(META_PATH, "rt", encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                r = json.loads(line)
                asin = r.get("asin")
                if asin not in wanted:
                    continue
                title = html.unescape((r.get("title") or "").strip())
                artist = html.unescape((r.get("brand") or "").strip())
                if not title or len(title) > 150:   # a veces el título viene con basura
                    continue
                titles[asin] = f"{title} - {artist}" if artist else title
                if n % 200_000 == 0:
                    print(f"  {n:,} líneas leídas")

        # Popularidad: cuántas calificaciones de 4-5 estrellas tiene cada álbum
        pop = {}
        if CACHE_PATH.exists():
            c = np.load(CACHE_PATH)
            codes, counts = np.unique(c["albums"][c["stars"] >= 4], return_counts=True)
            pop = {a: int(k) for a, k in zip(codes.tolist(), counts.tolist()) if a in titles}

        self.album_dao.replace_catalog(
            [Album(asin=asin, title=title, popularity=pop.get(asin, 0)) for asin, title in titles.items()])
        print(f"Guardé {len(titles):,} títulos de {len(wanted):,} álbumes en la base.")
