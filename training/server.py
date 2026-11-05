"""Servidor web local con la librería estándar. Abrir http://127.0.0.1:8000"""
import json
import unicodedata
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from preprocessing import MODELS_DIR
from recommend import load_artifacts, recommend

HOST, PORT = "127.0.0.1", 8000
HTML = (Path(__file__).resolve().parent / "index.html").read_bytes()

print("Cargando modelo...")
ART = load_artifacts()


def simplify(text):
    """Minúsculas y sin acentos, para que 'rosalia' encuentre 'ROSALÍA'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Índice de búsqueda: solo hay títulos si corriste names.py
POP = {}
pop_path = MODELS_DIR / "album_pop.json"
if pop_path.exists():
    with open(pop_path, encoding="utf-8") as f:
        POP = json.load(f)
INDEX = sorted(
    ((POP.get(asin, 0), asin, title, simplify(title)) for asin, title in ART.titles.items()),
    reverse=True,   # los más populares primero
)
print(f"Búsqueda por nombre: {'lista' if INDEX else 'no disponible (corré names.py)'}")


def search(query, limit=8):
    words = simplify(query).split()
    if not words:
        return []
    out = []
    for _, asin, title, norm in INDEX:
        if all(w in norm for w in words):
            out.append({"asin": asin, "title": title})
            if len(out) >= limit:
                break
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
        elif url.path == "/search":
            q = parse_qs(url.query).get("q", [""])[0]
            self._json(200, {"enabled": bool(INDEX), "results": search(q)})
        else:
            self._json(404, {"error": "No encontrado"})

    def do_POST(self):
        if self.path != "/recommend":
            return self._json(404, {"error": "No encontrado"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            codes = [str(c).strip() for c in body.get("albums", []) if str(c).strip()][:20]
        except Exception:
            return self._json(400, {"error": "Pedido inválido"})
        if not codes:
            return self._json(400, {"error": "Poné al menos un álbum"})

        unknown, recs = recommend(codes, ART, k=10)
        if not recs:
            return self._json(400, {"error": "Ningún álbum está en el modelo", "unknown": unknown})
        self._json(200, {"unknown": unknown, "recommendations": recs})


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Listo: http://{HOST}:{PORT}  (Ctrl+C para frenar)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChau.")