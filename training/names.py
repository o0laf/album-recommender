"""Opcional: arma los archivos que permiten buscar álbumes por nombre.
Crea en models/:
  - album_titles.json: código -> "título - artista"
  - album_pop.json:    código -> cuántos "me gusta" tiene (sirve para ordenar la búsqueda)
Necesita data/metadata/meta_CDs_and_Vinyl.json.gz (y haber corrido train.py)."""
import gzip
import html
import json

import numpy as np

from preprocessing import CACHE_PATH, MODELS_DIR, ROOT

META_PATH = ROOT / "data" / "metadata" / "meta_CDs_and_Vinyl.json.gz"

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

# Popularidad: cuántas calificaciones de 4-5 estrellas tiene cada álbum (del caché de train)
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