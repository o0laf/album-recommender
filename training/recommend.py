"""Recomienda álbumes parecidos a 1 o más álbumes.
Uso: python recommend.py B000XXXXXX B000YYYYYY"""
import argparse
import html
import json
import re
import sys
import unicodedata
from types import SimpleNamespace

import numpy as np

from preprocessing import MODELS_DIR

# --- Filtro de recomendaciones feas ---
CANDIDATE_FACTOR = 3        # pido k*3 candidatos (30 si k=10) y me quedo con los primeros k buenos
DEDUPE_EDITIONS = True      # no repetir el mismo álbum en otra edición (ni el que ya puso)
# Pedazos de texto que delatan basura pegada (HTML, links, etc.)
BAD_FRAGMENTS = ("<", ">", "{", "}", "http", "www.", "&#", "&amp", "&nbsp")


def normalize(x):
    """Deja cada vector con largo 1 (así el producto punto = similitud coseno)."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def recommend_from_indices(emb_norm, idxs, k=10):
    """Promedia los vectores de los álbumes dados y busca los más cercanos."""
    query = normalize(emb_norm[idxs].mean(axis=0))
    scores = emb_norm @ query
    scores[idxs] = -np.inf                       # excluir los que ya puso
    top = np.argpartition(scores, -k)[-k:]       # los k mejores (sin ordenar)
    top = top[np.argsort(-scores[top])]          # ordenarlos
    return [(int(i), float(scores[i])) for i in top]


def hit_rate_at_k(emb_norm, inputs, targets, k=10, batch=500):
    """Para cada usuario: ¿el álbum escondido aparece entre los k recomendados?"""
    hits = 0
    for start in range(0, len(inputs), batch):      # por tandas, para no llenar la RAM
        ins = inputs[start:start + batch]
        tg = targets[start:start + batch]
        q = normalize(emb_norm[ins].mean(axis=1))   # promedio de los 4 álbumes
        scores = q @ emb_norm.T
        scores[np.arange(len(ins))[:, None], ins] = -np.inf   # excluir los 4 dados
        top = np.argpartition(scores, -k, axis=1)[:, -k:]
        hits += int((top == tg[:, None]).any(axis=1).sum())
    return hits / len(inputs)


def load_artifacts():
    from tensorflow import keras   # import acá adentro: tarda en cargar
    model = keras.models.load_model(MODELS_DIR / "model.keras")
    emb = model.get_layer("album_emb").get_weights()[0]
    with open(MODELS_DIR / "idx_to_album.json", encoding="utf-8") as f:
        idx_to_album = json.load(f)
    titles = {}
    titles_path = MODELS_DIR / "album_titles.json"    # existe solo si corriste names.py
    if titles_path.exists():
        with open(titles_path, encoding="utf-8") as f:
            titles = json.load(f)
    return SimpleNamespace(
        emb_norm=normalize(emb),
        idx_to_album=idx_to_album,
        album_to_idx={a: i for i, a in enumerate(idx_to_album)},
        titles=titles,
    )


def title_problem(title):
    """Devuelve por qué NO conviene recomendar este título, o None si está bien."""
    t = (title or "").strip()
    if not t:
        return "sin título"
    low = t.lower()
    if re.search(r"\bsingle\b", low):          # "single" suelto; "Singles" (compilado) pasa
        return "single"
    if any(frag in low for frag in BAD_FRAGMENTS) or any(ord(c) < 32 for c in t):
        return "texto raro"                      # (se mira antes que la barra: los links tienen "//")
    if "//" in t or " / " in t:                  # típico de singles: "Tema A / Tema B"
        return "doble tema con barra"
    letras = sum(c.isalnum() for c in t)
    if len(t) < 3 or letras < len(t) / 2:        # casi todo símbolos
        return "texto raro"
    return None


def edition_key(title):
    """Clave para detectar el mismo álbum en otra edición:
    minúsculas, sin acentos, sin (paréntesis) ni [corchetes], sin 'deluxe', 'remastered', etc."""
    text = html.unescape(title or "").lower()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\b(deluxe|remastered|edition|expanded|explicit)\b", " ", text)
    return " ".join(text.split())


def filter_candidates(candidates, art, seed_idxs, k):
    """De la lista de candidatos (ya ordenada de mejor a peor) deja los primeros k
    que pasan el filtro. Si no hay títulos (no corriste names.py) no se puede filtrar."""
    if not art.titles:
        return candidates[:k]
    vistos = {edition_key(art.titles.get(art.idx_to_album[i], "")) for i in seed_idxs}
    vistos.discard("")
    out = []
    for i, sc in candidates:
        if sc == -np.inf:                         # eran los álbumes ya puestos
            continue
        title = art.titles.get(art.idx_to_album[i], "")
        if title_problem(title):
            continue
        key = edition_key(title)
        if DEDUPE_EDITIONS and key:
            if key in vistos:
                continue
            vistos.add(key)
        out.append((i, sc))
        if len(out) == k:
            break
    return out


def recommend(codes, art, k=10):
    """Devuelve (códigos que no conozco, lista de recomendaciones)."""
    idxs, unknown = [], []
    for c in codes:
        c = c.strip().upper()
        if c in art.album_to_idx:
            idxs.append(art.album_to_idx[c])
        else:
            unknown.append(c)
    if not idxs:
        return unknown, []
    idxs = np.array(list(dict.fromkeys(idxs)))   # sin repetidos
    # Pido de más (k*3) porque después el filtro tira los singles, los sin título, etc.
    n_cand = min(k * CANDIDATE_FACTOR, len(art.emb_norm))
    candidates = recommend_from_indices(art.emb_norm, idxs, n_cand)
    recs = filter_candidates(candidates, art, idxs, k)
    return unknown, [
        {"asin": art.idx_to_album[i], "title": art.titles.get(art.idx_to_album[i], ""), "score": sc}
        for i, sc in recs
    ]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # para que no explote con acentos
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("albums", nargs="+", help="uno o más códigos de álbum (asin)")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    art = load_artifacts()
    unknown, recs = recommend(args.albums, art, args.k)
    if unknown:
        print("No conozco estos códigos (el modelo no los vio en train):", unknown)
    if not recs:
        sys.exit("No hay ningún álbum válido para recomendar.")
    for n, r in enumerate(recs, 1):
        print(f"{n:2d}. {r['asin']}  {r['title']}  (similitud {r['score']:.3f})")