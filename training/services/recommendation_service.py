"""Capa de servicios: la lógica de recomendar (promediar vectores, buscar vecinos, filtrar)."""
import html
import json
import re
import unicodedata
from types import SimpleNamespace

import numpy as np

from config import (BAD_FRAGMENTS, CANDIDATE_FACTOR, DEDUPE_EDITIONS, MODELS_DIR)


class RecommendationService:
    """Cargar el modelo (tarda) y recomendar. Las funciones matemáticas son estáticas:
    el entrenamiento y la evaluación las usan sin necesidad de cargar nada."""

    def __init__(self):
        self.art = self.load_artifacts()

    # ---------- Matemática (no necesita modelo cargado) ----------

    @staticmethod
    def normalize(x):
        """Deja cada vector con largo 1 (así el producto punto = similitud coseno)."""
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, 1e-9)

    @staticmethod
    def recommend_from_indices(emb_norm, idxs, k=10):
        """Promedia los vectores de los álbumes dados y busca los más cercanos."""
        query = RecommendationService.normalize(emb_norm[idxs].mean(axis=0))
        scores = emb_norm @ query
        scores[idxs] = -np.inf                       # excluir los que ya puso
        top = np.argpartition(scores, -k)[-k:]       # los k mejores (sin ordenar)
        top = top[np.argsort(-scores[top])]          # ordenarlos
        return [(int(i), float(scores[i])) for i in top]

    @staticmethod
    def hit_rate_at_k(emb_norm, inputs, targets, k=10, batch=500):
        """Para cada usuario: ¿el álbum escondido aparece entre los k recomendados?"""
        hits = 0
        for start in range(0, len(inputs), batch):      # por tandas, para no llenar la RAM
            ins = inputs[start:start + batch]
            tg = targets[start:start + batch]
            q = RecommendationService.normalize(emb_norm[ins].mean(axis=1))   # promedio de los 4 álbumes
            scores = q @ emb_norm.T
            scores[np.arange(len(ins))[:, None], ins] = -np.inf   # excluir los 4 dados
            top = np.argpartition(scores, -k, axis=1)[:, -k:]
            hits += int((top == tg[:, None]).any(axis=1).sum())
        return hits / len(inputs)

    # ---------- Carga del modelo ----------

    def load_artifacts(self):
        from tensorflow import keras   # import acá adentro: tarda en cargar
        model = keras.models.load_model(MODELS_DIR / "model.keras")
        emb = model.get_layer("album_emb").get_weights()[0]
        with open(MODELS_DIR / "idx_to_album.json", encoding="utf-8") as f:
            idx_to_album = json.load(f)
        titles = {}
        titles_path = MODELS_DIR / "album_titles.json"    # existe solo si corriste "names"
        if titles_path.exists():
            with open(titles_path, encoding="utf-8") as f:
                titles = json.load(f)
        return SimpleNamespace(
            emb_norm=self.normalize(emb),
            idx_to_album=idx_to_album,
            album_to_idx={a: i for i, a in enumerate(idx_to_album)},
            titles=titles,
        )

    # ---------- Filtro de recomendaciones feas ----------

    @staticmethod
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

    @staticmethod
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

    def filter_candidates(self, candidates, seed_idxs, k):
        """De la lista de candidatos (ya ordenada de mejor a peor) deja los primeros k
        que pasan el filtro. Si no hay títulos (no corriste names) no se puede filtrar."""
        art = self.art
        if not art.titles:
            return candidates[:k]
        vistos = {self.edition_key(art.titles.get(art.idx_to_album[i], "")) for i in seed_idxs}
        vistos.discard("")
        out = []
        for i, sc in candidates:
            if sc == -np.inf:                         # eran los álbumes ya puestos
                continue
            title = art.titles.get(art.idx_to_album[i], "")
            if self.title_problem(title):
                continue
            key = self.edition_key(title)
            if DEDUPE_EDITIONS and key:
                if key in vistos:
                    continue
                vistos.add(key)
            out.append((i, sc))
            if len(out) == k:
                break
        return out

    # ---------- Lo que usan los controladores ----------

    def recommend(self, codes, k=10):
        """Devuelve (códigos que no conozco, lista de recomendaciones)."""
        art = self.art
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
        candidates = self.recommend_from_indices(art.emb_norm, idxs, n_cand)
        recs = self.filter_candidates(candidates, idxs, k)
        return unknown, [
            {"asin": art.idx_to_album[i], "title": art.titles.get(art.idx_to_album[i], ""), "score": sc}
            for i, sc in recs
        ]

    def run_cli(self, codes, k=10):
        """Recomienda desde la consola e imprime el resultado."""
        unknown, recs = self.recommend(codes, k)
        if unknown:
            print("No conozco estos códigos (el modelo no los vio en train):", unknown)
        if not recs:
            raise SystemExit("No hay ningún álbum válido para recomendar.")
        for n, r in enumerate(recs, 1):
            print(f"{n:2d}. {r['asin']}  {r['title']}  (similitud {r['score']:.3f})")