"""DAO de álbumes: guarda y lee la tabla `album`.
Reemplaza a idx_to_album.json, album_to_idx.json, album_titles.json y album_pop.json."""
from database.database_connection import DatabaseConnection
from domain.album import Album


class AlbumDAO:

    def __init__(self, db=None):
        self.db = db or DatabaseConnection()

    def save_vocabulary(self, asins):
        """Guarda qué álbumes conoce la red y en qué posición (la lista viene en el orden de la red).
        Reemplaza el vocabulario anterior; los títulos ya guardados no se tocan."""
        asins = list(asins)
        with self.db.transaction() as conn:
            conn.execute("UPDATE album SET model_index = NULL")
            conn.executemany("INSERT OR IGNORE INTO album (asin) VALUES (?)", [(a,) for a in asins])
            conn.executemany("UPDATE album SET model_index = ? WHERE asin = ?",
                             [(i, a) for i, a in enumerate(asins)])

    def find_all_in_model(self):
        """Todos los álbumes que la red conoce, ordenados por su posición en la red."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT asin, model_index, title, popularity FROM album "
                "WHERE model_index IS NOT NULL ORDER BY model_index"
            ).fetchall()
        return [Album(asin=r["asin"], model_index=r["model_index"], title=r["title"] or "",
                      popularity=r["popularity"]) for r in rows]

    def replace_catalog(self, albums):
        """Reemplaza los títulos y la popularidad de todos los álbumes por los de la lista."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE album SET title = NULL, popularity = 0")
            conn.executemany("UPDATE album SET title = ?, popularity = ? WHERE asin = ?",
                             [(a.title, a.popularity, a.asin) for a in albums])

    def count_in_model(self):
        with self.db.transaction() as conn:
            return conn.execute("SELECT COUNT(*) FROM album WHERE model_index IS NOT NULL").fetchone()[0]
