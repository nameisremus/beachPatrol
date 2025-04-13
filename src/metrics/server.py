from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Callable
import logging

LOG = logging.getLogger(__name__)

class _Handler(BaseHTTPRequestHandler):
    _payload: Callable[[], bytes] | None = None

    def do_GET(self):
        if self.path in ("/metrics", "/api/metrics"):
            assert self._payload is not None
            data = self._payload()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(data)
        elif self.path in ("/health", "/readiness"):
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_):
        return

class MetricsServer:
    # Starts the background thread
    def __init__(self, port: int, payload_supplier: Callable[[], bytes]):
        _Handler._payload = payload_supplier
        self._server = HTTPServer(("", port), _Handler)
        LOG.info("Metrics server configured on port %d", port)

    def start(self):
        LOG.info("Starting metrics server thread…")
        Thread(target=self._server.serve_forever, daemon=True).start()
