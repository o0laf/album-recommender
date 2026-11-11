"""Capa de servicios: títulos de los álbumes y búsqueda por nombre."""
import gzip
import html
import json
import unicodedata

import numpy as np

from config import CACHE_PATH, META_PATH, MODELS_DIR


class CatalogService:
    """Búsqueda por nombre. Necesita los títulos (que crea build_files)."""

    def __init__(self, titles):
        pop = {}
        pop_path = MODELS_DIR / "album_pop.json"
        if pop_path.exists():
            with open(pop_path, encoding="utf-8") as f:
                pop = json.load(f)
        # Índice de búsqueda, con los más populares primero
        self.index = sorted(
            ((pop.get(asin, 0), asin, title, self.simplify(title)) for asin, title in titles.items()),
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

    @staticmethod
    def build_files():
        """Arma en models/ los archivos que permiten buscar por nombre:
          - album_titles.json: código -> "título - artista"
          - album_pop.json:    código -> cuántos "me gusta" tiene
        Necesita data/metadata/meta_CDs_and_Vinyl.json.gz y haber entrenado antes."""
        with open(MODELS_DIR / "album_to_idx.json", encoding="utf-8") as f:
            wanted = set(json.load(f))      # solo guardo los álbumes que conoce el modelo

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

        with open(MODELS_DIR / "album_titles.json", "w", encoding="utf-8") as f:
            json.dump(titles, f, ensure_ascii=False)
        with open(MODELS_DIR / "album_pop.json", "w", encoding="utf-8") as f:
            json.dump(pop, f)
        print(f"Guardé {len(titles):,} títulos de {len(wanted):,} álbumes.")