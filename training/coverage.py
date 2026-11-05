"""Mide cuántos álbumes de Vynl se pueden encontrar en el modelo (por título + artista).
Uso (desde training/):
    python coverage.py ../data/vynl/database.sqlite
Necesita: models/album_to_idx.json (lo crea train.py) y data/metadata/meta_CDs_and_Vinyl.json.gz
"""
import argparse
import csv
import gzip
import html
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm(text):
    """Minúsculas, sin acentos, sin (paréntesis) ni [corchetes], sin signos."""
    text = html.unescape(text or "").lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)           # (Remastered 1996), [Explicit]
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(deluxe|remastered|edition|expanded|explicit)\b", " ", text)
    return " ".join(text.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sqlite", help="ruta a database.sqlite de Vynl")
    p.add_argument("--meta", default=str(ROOT / "data" / "metadata" / "meta_CDs_and_Vinyl.json.gz"))
    p.add_argument("--model-albums", default=str(ROOT / "models" / "album_to_idx.json"))
    p.add_argument("--out", default=str(ROOT / "models" / "coverage.csv"))
    a = p.parse_args()

    # 1) Álbumes de Vynl (solo lectura)
    db = sqlite3.connect(f"file:{a.sqlite}?mode=ro", uri=True)
    vynl = db.execute(
        "SELECT a.id, a.title, ar.name FROM albums a JOIN artists ar ON ar.id = a.artist_id"
    ).fetchall()
    wanted = {(norm(t), norm(n)) for _, t, n in vynl}

    # 2) Códigos que el modelo conoce
    with open(a.model_albums, encoding="utf-8") as f:
        in_model = set(json.load(f))

    # 3) Recorrer la metadata y quedarme solo con lo que Vynl busca
    found = {}      # (título, artista) normalizados -> lista de asins
    with gzip.open(a.meta, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            asin = r.get("asin")
            if asin not in in_model:
                continue
            key = (norm(r.get("title")), norm(r.get("brand")))
            if key in wanted:
                found.setdefault(key, []).append(asin)

    # 4) Resultado
    rows, hits = [], 0
    for vid, title, artist in vynl:
        asins = found.get((norm(title), norm(artist)), [])
        hits += bool(asins)
        rows.append([vid, title, artist, " ".join(asins)])

    Path(a.out).parent.mkdir(exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel lo abre bien
        w = csv.writer(f)
        w.writerow(["vynl_album_id", "titulo", "artista", "asins_encontrados"])
        w.writerows(rows)

    print(f"Álbumes de Vynl encontrados en el modelo: {hits} de {len(vynl)} ({hits / len(vynl):.0%})")
    print("Veredicto:", "vale la pena integrarlo" if hits / len(vynl) >= 0.5
          else "cobertura baja: mejor dejarlo como prototipo")
    print(f"Detalle en {a.out}")


if __name__ == "__main__":
    main()
