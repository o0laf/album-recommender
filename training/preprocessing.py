"""Lee el .gz, arma los ids de usuarios y álbumes y divide en train / validación / test."""
import gzip
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Rutas relativas a este archivo (así funciona corras desde donde corras)
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews" / "CDs_and_Vinyl_5.json.gz"
CACHE_PATH = ROOT / "data" / "ratings_cache.npz"   # copia rápida de las 3 columnas
MODELS_DIR = ROOT / "models"

LIKE_THRESHOLD = 4     # 4 o 5 estrellas = le gustó
N_INPUT = 4            # álbumes que le damos a cada usuario en la evaluación
SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def read_ratings():
    """Devuelve (usuarios, álbumes, estrellas). La primera vez tarda unos minutos;
    después usa el caché."""
    if CACHE_PATH.exists():
        c = np.load(CACHE_PATH)
        return c["users"], c["albums"], c["stars"]

    users, albums, stars = [], [], []
    t0 = time.time()
    # "rt" + utf-8: lee el .gz como texto sin descomprimirlo a disco
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)  # un JSON por línea
            users.append(r["reviewerID"])
            albums.append(r["asin"])
            stars.append(int(r["overall"]))
            if n % 200_000 == 0:
                print(f"  {n:,} líneas leídas ({time.time() - t0:.0f}s)")

    users = np.array(users)
    albums = np.array(albums)
    stars = np.array(stars, dtype=np.int8)
    np.savez_compressed(CACHE_PATH, users=users, albums=albums, stars=stars)
    print(f"Lectura lista en {time.time() - t0:.0f}s. Guardé un caché para la próxima.")
    return users, albums, stars


def make_eval_set(u, a, s, row_split, which, old_to_new, rng):
    """Para cada usuario de test/val: N_INPUT álbumes que le gustaron (entrada),
    1 más que también le gustó (escondido, para el hit rate) y la lista de TODOS
    los que le gustaron y no se dieron como entrada (para precision / recall / F1)."""
    mask = (row_split == which) & (s >= LIKE_THRESHOLD)
    eu = u[mask]
    ea = old_to_new[a[mask]]
    keep = ea >= 0                      # solo álbumes que el modelo conoce
    eu, ea = eu[keep], ea[keep]

    order = np.argsort(eu, kind="stable")   # agrupar por usuario
    eu, ea = eu[order], ea[order]
    groups = np.split(ea, np.flatnonzero(np.diff(eu)) + 1)

    inputs, targets, relevants = [], [], []
    for g in groups:
        if len(g) < N_INPUT + 1:        # muy pocos favoritos: no se evalúa
            continue
        perm = rng.permutation(g)       # mismo sorteo de siempre: el hit rate no cambia
        inputs.append(perm[:N_INPUT])
        targets.append(perm[N_INPUT])
        relevants.append(perm[N_INPUT:].astype(np.int64))   # el escondido + todos los demás que le gustaron
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64), relevants


def build_data():
    users, albums, stars = read_ratings()

    # 1) Códigos de texto -> números (0, 1, 2, ...)
    _, u = np.unique(users, return_inverse=True)
    asins, a = np.unique(albums, return_inverse=True)

    # 2) Sacar calificaciones repetidas (mismo usuario, mismo álbum): me quedo con la primera
    key = u.astype(np.int64) * len(asins) + a
    _, first = np.unique(key, return_index=True)
    u, a, s = u[first], a[first], stars[first]

    # 3) Dividir USUARIOS (no filas): 80% train, 10% val, 10% test
    rng = np.random.default_rng(SEED)
    n_all_users = int(u.max()) + 1
    perm = rng.permutation(n_all_users)
    n_val = int(VAL_FRACTION * n_all_users)
    n_test = int(TEST_FRACTION * n_all_users)
    split = np.zeros(n_all_users, dtype=np.int8)       # 0 = train
    split[perm[:n_val]] = 1                            # 1 = validación
    split[perm[n_val:n_val + n_test]] = 2              # 2 = test
    row_split = split[u]

    # 4) Con los usuarios de train armo el vocabulario (solo lo que la red puede aprender)
    m = row_split == 0
    tr_u, tr_a, tr_s = u[m], a[m], s[m]
    train_albums_old = np.unique(tr_a)
    old_to_new = np.full(len(asins), -1, dtype=np.int64)
    old_to_new[train_albums_old] = np.arange(len(train_albums_old))

    train_albums = old_to_new[tr_a].astype(np.int32)
    _, train_users = np.unique(tr_u, return_inverse=True)
    train_users = train_users.astype(np.int32)
    train_labels = (tr_s >= LIKE_THRESHOLD).astype(np.float32)

    n_users = int(train_users.max()) + 1
    n_albums = len(train_albums_old)
    idx_to_album = asins[train_albums_old].tolist()
    popularity = np.bincount(train_albums[train_labels == 1], minlength=n_albums)

    # 5) Conjuntos de evaluación
    val_inputs, val_targets, val_relevant = make_eval_set(u, a, s, row_split, 1, old_to_new, rng)
    test_inputs, test_targets, test_relevant = make_eval_set(u, a, s, row_split, 2, old_to_new, rng)

    print(f"Calificaciones únicas: {len(u):,} | usuarios: {n_all_users:,} | álbumes: {len(asins):,}")
    print(f"Le gustó (4-5 estrellas): {(s >= LIKE_THRESHOLD).mean():.1%}")
    print(f"Train: {n_users:,} usuarios, {n_albums:,} álbumes, {len(train_users):,} calificaciones")
    print(f"Usuarios evaluables -> validación: {len(val_inputs):,} | test: {len(test_inputs):,}")

    return SimpleNamespace(
        train_users=train_users, train_albums=train_albums, train_labels=train_labels,
        n_users=n_users, n_albums=n_albums, idx_to_album=idx_to_album,
        popularity=popularity,
        val_inputs=val_inputs, val_targets=val_targets, val_relevant=val_relevant,
        test_inputs=test_inputs, test_targets=test_targets, test_relevant=test_relevant,
    )


if __name__ == "__main__":
    build_data()