"""DAO de cobertura de Vynl: guarda y lee la tabla `vynl_coverage`. Reemplaza a coverage.csv."""
from database.database_connection import DatabaseConnection
from domain.vynl_coverage_entry import VynlCoverageEntry


class VynlCoverageDAO:

    def __init__(self, db=None):
        self.db = db or DatabaseConnection()

    def replace_all(self, entries):
        """Cada medición reemplaza a la anterior."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM vynl_coverage")
            conn.executemany(
                "INSERT INTO vynl_coverage (vynl_album_id, title, artist, matched_asins) VALUES (?, ?, ?, ?)",
                [(e.vynl_album_id, e.title, e.artist, " ".join(e.asins)) for e in entries])

    def find_all(self):
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM vynl_coverage ORDER BY rowid").fetchall()
        return [VynlCoverageEntry(vynl_album_id=r["vynl_album_id"], title=r["title"], artist=r["artist"],
                                  asins=r["matched_asins"].split()) for r in rows]
