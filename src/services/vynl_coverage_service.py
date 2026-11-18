"""Capa de servicios: mide cuántos álbumes de Vynl se pueden encontrar en el modelo (por título + artista).
Necesita: los álbumes del modelo en la base (los guarda el entrenamiento) y data/metadata/meta_CDs_and_Vinyl.json.gz.
El resultado queda en la tabla `vynl_coverage`."""
import gzip
import html
import json
import re
import sqlite3
import unicodedata

from config import META_PATH
from daos.album_dao import AlbumDAO
from daos.vynl_coverage_dao import VynlCoverageDAO
from domain.vynl_coverage_entry import VynlCoverageEntry


class VynlCoverageService:

    def __init__(self, album_dao=None, coverage_dao=None):
        self.album_dao = album_dao or AlbumDAO()
        self.coverage_dao = coverage_dao or VynlCoverageDAO()

    @staticmethod
    def norm(text):
        """Minúsculas, sin acentos, sin (paréntesis) ni [corchetes], sin signos."""
        text = html.unescape(text or "").lower()
        text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)           # (Remastered 1996), [Explicit]
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        text = re.sub(r"\b(deluxe|remastered|edition|expanded|explicit)\b", " ", text)
        return " ".join(text.split())

    def run(self, sqlite_path, meta=META_PATH):
        norm = self.norm

        # 1) Álbumes de Vynl (solo lectura)
        db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        try:
            vynl = db.execute(
                "SELECT a.id, a.title, ar.name FROM albums a JOIN artists ar ON ar.id = a.artist_id"
            ).fetchall()
        finally:
            db.close()
        if not vynl:
            raise SystemExit("La base de Vynl no tiene álbumes.")
        wanted = {(norm(t), norm(n)) for _, t, n in vynl}

        # 2) Códigos que el modelo conoce
        in_model = {a.asin for a in self.album_dao.find_all_in_model()}
        if not in_model:
            raise SystemExit("La base no tiene los álbumes del modelo. Corré primero 'python main.py train'.")

        # 3) Recorrer la metadata y quedarme solo con lo que Vynl busca
        found = {}      # (título, artista) normalizados -> lista de asins
        with gzip.open(meta, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                asin = r.get("asin")
                if asin not in in_model:
                    continue
                key = (norm(r.get("title")), norm(r.get("brand")))
                if key in wanted:
                    found.setdefault(key, []).append(asin)

        # 4) Resultado
        entries = [VynlCoverageEntry(vynl_album_id=str(vid), title=title, artist=artist,
                                     asins=found.get((norm(title), norm(artist)), []))
                   for vid, title, artist in vynl]
        hits = sum(e.found for e in entries)
        self.coverage_dao.replace_all(entries)

        print(f"Álbumes de Vynl encontrados en el modelo: {hits} de {len(vynl)} ({hits / len(vynl):.0%})")
        print("Veredicto:", "vale la pena integrarlo" if hits / len(vynl) >= 0.5
              else "cobertura baja: mejor dejarlo como prototipo")
        print("Detalle guardado en la base (tabla vynl_coverage).")
