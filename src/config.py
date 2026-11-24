"""Ajustes del proyecto: rutas y valores que se pueden cambiar en un solo lugar."""
from pathlib import Path

# --- Rutas (relativas a este archivo: funciona corras desde donde corras) ---
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_PATH = ROOT / "data" / "reviews" / "CDs_and_Vinyl_5.json.gz"
META_PATH = ROOT / "data" / "metadata" / "meta_CDs_and_Vinyl.json.gz"
CACHE_PATH = ROOT / "data" / "ratings_cache.npz"    # copia rápida de las 3 columnas
DB_PATH = ROOT / "data" / "album_recommender.db"    # base SQLite: álbumes, entrenamientos, métricas, experimentos
ARTIFACTS_DIR = ROOT / "artifacts"                  # archivos pesados del entrenamiento: model.keras y gráficos
VIEWS_DIR = SRC_DIR / "views"

# --- Datos ---
LIKE_RATING_THRESHOLD = 4     # 4 o 5 estrellas = le gustó
EVALUATION_INPUT_COUNT = 4    # álbumes que le damos a cada usuario en la evaluación
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10

# --- Entrenamiento ---
EMBEDDING_DIMENSION = 32      # largo de cada embedding
TRAINING_BATCH_SIZE = 4096
NEGATIVE_SAMPLE_RATIO = 4     # negativos inventados por cada "me gusta"
NEGATIVE_SAMPLING_POWER = 0.75 # 0 = totalmente al azar; 0.75 = más a menudo los populares
LEARNING_RATE = 0.003
MAX_EPOCHS = 20
PATIENCE = 3        # corta si la validación no mejora en 3 épocas
RECOMMENDATION_COUNT = 10     # cuántas recomendaciones se piden
SEED = 42

# --- Recomendación ---
CANDIDATE_FACTOR = 3        # pido k*3 candidatos y me quedo con los primeros k buenos
DEDUPE_EDITIONS = True      # no repetir el mismo álbum en otra edición
# Pedazos de texto que delatan basura pegada (HTML, links, etc.)
BAD_FRAGMENTS = ("<", ">", "{", "}", "http", "www.", "&#", "&amp", "&nbsp")

# --- Servidor web ---
HOST, PORT = "127.0.0.1", 8000
