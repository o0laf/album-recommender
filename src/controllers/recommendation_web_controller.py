"""Controlador web (servidor local con la librería estándar).
Recibe los pedidos HTTP, se los pasa a los servicios y devuelve la respuesta.
No tiene lógica de negocio: la vista es views/index.html y la lógica está en services/."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from config import HOST, PORT, VIEWS_DIR
from services.album_catalog_service import AlbumCatalogService
from services.album_recommendation_service import AlbumRecommendationService


class RecommendationWebController:

    def __init__(self):
        self.html = (VIEWS_DIR / "index.html").read_bytes()
        print("Cargando modelo...")
        self.recommender = AlbumRecommendationService()
        self.catalog = AlbumCatalogService()
        print(f"Búsqueda por nombre: {'lista' if self.catalog.enabled else 'no disponible (corré: python main.py names)'}")

    def make_handler(self):
        controller = self

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
                    self._send(200, controller.html, "text/html; charset=utf-8")
                elif url.path == "/search":
                    q = parse_qs(url.query).get("q", [""])[0]
                    self._json(200, {"enabled": controller.catalog.enabled,
                                     "results": controller.catalog.search(q)})
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

                unknown, recs = controller.recommender.recommend(codes, k=10)
                if not recs:
                    return self._json(400, {"error": "Ningún álbum está en el modelo", "unknown": unknown})
                self._json(200, {"unknown": unknown, "recommendations": [r.to_dict() for r in recs]})

        return Handler

    def run(self):
        server = HTTPServer((HOST, PORT), self.make_handler())
        print(f"Listo: http://{HOST}:{PORT}  (Ctrl+C para frenar)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nChau.")
