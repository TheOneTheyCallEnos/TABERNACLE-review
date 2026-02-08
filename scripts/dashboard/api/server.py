import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Tuple

from dashboard.api import routes
from dashboard.config import DashboardConfig


RouteHandler = Callable[[], Tuple[int, Dict[str, object]]]


class DashboardHandler(BaseHTTPRequestHandler):
    routes: Dict[str, RouteHandler] = {}

    def do_GET(self) -> None:
        handler = self.routes.get(self.path)
        if not handler:
            self._send_json(404, {"error": "not_found"})
            return
        status, payload = handler()
        self._send_json(status, payload)

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(config: DashboardConfig, host: str = "127.0.0.1", port: int = 8081) -> HTTPServer:
    DashboardHandler.routes = {
        "/health": lambda: (200, routes.get_health()),
        "/metrics": lambda: (200, {"metrics": routes.get_latest_metrics(str(config.db_path))}),
    }
    return HTTPServer((host, port), DashboardHandler)


if __name__ == "__main__":
    server = build_server(DashboardConfig())
    server.serve_forever()
