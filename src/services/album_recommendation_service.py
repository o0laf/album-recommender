"""Capa de servicios: la lógica de recomendar (promediar vectores, buscar vecinos, filtrar)."""
import html
import re
import unicodedata
from types import SimpleNamespace

import numpy as np

from config import ARTIFACTS_DIR, BAD_FRAGMENTS, CANDIDATE_FACTOR, DEDUPE_EDITIONS
from daos.album_dao import AlbumDAO
from domain.recommendation import Recommendation


class AlbumRecommendationService:
    """Cargar el modelo (tarda) y recomendar. Las funciones matemáticas son estáticas:
    el entrenamiento y la evaluación las usan sin necesidad de cargar nada."""

    def __init__(self, album_dao=None):
        self.album_dao = album_dao or AlbumDAO()
        self.art = self.load_artifacts()

    # ---------- Matemática (no necesita modelo cargado) ----------

    @staticmethod
    def normalize(vectors):
        """Deja cada vector con largo 1 (así el producto punto = similitud coseno)."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    @staticmethod
    def recommend_from_indices(emb_norm, seed_indices, k=10):
        """Promedia los vectores de los álbumes dados y busca los más cercanos."""
        query = AlbumRecommendationService.normalize(emb_norm[seed_indices].mean(axis=0))
        scores = emb_norm @ query
        scores[seed_indices] = -np.inf               # excluir los que ya puso
        top = np.argpartition(scores, -k)[-k:]       # los k mejores (sin ordenar)
        top = top[np.argsort(-scores[top])]          # ordenarlos
        return [(int(i), float(scores[i])) for i in top]

    @staticmethod
    def hit_rate_at_k(emb_norm, inputs, targets, k=10, batch=500):
        """Para cada usuario: ¿el álbum escondido aparece entre los k recomendados?"""
        hits = 0
        for start in range(0, len(inputs), batch):      # por tandas, para no llenar la RAM
            input_batch = inputs[start:start + batch]
            target_batch = targets[start:start + batch]
            query = AlbumRecommendationService.normalize(emb_norm[input_batch].mean(axis=1))
            scores = query @ emb_norm.T
            scores[np.arange(len(input_batch))[:, None], input_batch] = -np.inf
            top = np.argpartition(scores, -k, axis=1)[:, -k:]
            hits += int((top == target_batch[:, None]).any(axis=1).sum())
        return hits / len(inputs)

    # ---------- Carga del modelo ----------

    def load_artifacts(self):
        """El modelo se lee de artifacts/model.keras; los códigos y títulos de los álbumes, de la base."""
        model_path = ARTIFACTS_DIR / "model.keras"
        if not model_path.exists():
            raise SystemExit(f"No encuentro {model_path}. Corré primero: python main.py train")
        albums = self.album_dao.find_all_in_model()
        if not albums:
            raise SystemExit("La base no tiene los álbumes del modelo. Corré 'python main.py train' "
                             "(o 'python main.py import-legacy' si ya tenías los archivos .json).")

        from tensorflow import keras   # import acá adentro: tarda en cargar
        model = keras.models.load_model(model_path)
        emb = model.get_layer("album_emb").get_weights()[0]
        if emb.shape[0] != len(albums):
            raise SystemExit(f"El modelo tiene {emb.shape[0]} álbumes y la base {len(albums)}: no son del "
                             "mismo entrenamiento. Volvé a correr 'python main.py train'.")
        idx_to_album = [a.asin for a in albums]
        titles = {a.asin: a.title for a in albums if a.title}    # hay títulos solo si corriste "names"
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
        trimmed_title = (title or "").strip()
        if not trimmed_title:
            return "sin título"
        normalized_title = trimmed_title.lower()
        if re.search(r"\bsingle\b", normalized_title):
            return "single"
        if any(fragment in normalized_title for fragment in BAD_FRAGMENTS) or any(ord(c) < 32 for c in trimmed_title):
            return "texto raro"
        if "//" in trimmed_title or " / " in trimmed_title:
            return "doble tema con barra"
        alphanumeric_character_count = sum(c.isalnum() for c in trimmed_title)
        if len(trimmed_title) < 3 or alphanumeric_character_count < len(trimmed_title) / 2:
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

    def filter_candidates(self, candidates, seed_indices, k):
        """De la lista de candidatos (ya ordenada de mejor a peor) deja los primeros k
        que pasan el filtro. Si no hay títulos (no corriste names) no se puede filtrar."""
        artifacts = self.art
        if not artifacts.titles:
            return candidates[:k]
        seen_editions = {
            self.edition_key(artifacts.titles.get(artifacts.idx_to_album[index], ""))
            for index in seed_indices
        }
        seen_editions.discard("")
        filtered_candidates = []
        for index, score in candidates:
            if score == -np.inf:
                continue
            title = artifacts.titles.get(artifacts.idx_to_album[index], "")
            if self.title_problem(title):
                continue
            edition_key = self.edition_key(title)
            if DEDUPE_EDITIONS and edition_key:
                if edition_key in seen_editions:
                    continue
                seen_editions.add(edition_key)
            filtered_candidates.append((index, score))
            if len(filtered_candidates) == k:
                break
        return filtered_candidates

    # ---------- Lo que usan los controladores ----------

    def recommend(self, codes, k=10):
        """Devuelve (códigos que no conozco, lista de Recommendation)."""
        if k < 1:
            raise ValueError("k debe ser al menos 1")
        artifacts = self.art
        seed_indices, unknown_codes = [], []
        for code in codes:
            normalized_code = code.strip().upper()
            if normalized_code in artifacts.album_to_idx:
                seed_indices.append(artifacts.album_to_idx[normalized_code])
            else:
                unknown_codes.append(normalized_code)
        if not seed_indices:
            return unknown_codes, []
        seed_indices = np.array(list(dict.fromkeys(seed_indices)))
        # Pido de más (k*3) porque después el filtro tira los singles, los sin título, etc.
        candidate_count = min(k * CANDIDATE_FACTOR, len(artifacts.emb_norm))
        candidates = self.recommend_from_indices(artifacts.emb_norm, seed_indices, candidate_count)
        recommendations = self.filter_candidates(candidates, seed_indices, k)
        return unknown_codes, [
            Recommendation(
                asin=artifacts.idx_to_album[index],
                title=artifacts.titles.get(artifacts.idx_to_album[index], ""),
                score=score,
            )
            for index, score in recommendations
        ]

    def run_cli(self, codes, k=10):
        """Recomienda desde la consola e imprime el resultado."""
        unknown_codes, recommendations = self.recommend(codes, k)
        if unknown_codes:
            print("No conozco estos códigos (el modelo no los vio en train):", unknown_codes)
        if not recommendations:
            raise SystemExit("No hay ningún álbum válido para recomendar.")
        for position, recommendation in enumerate(recommendations, 1):
            print(f"{position:2d}. {recommendation.asin}  {recommendation.title}  "
                  f"(similitud {recommendation.score:.3f})")