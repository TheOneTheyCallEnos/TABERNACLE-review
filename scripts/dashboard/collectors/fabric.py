import socket
import time
from typing import Any, Dict

from dashboard.collectors.base import BaseCollector


class FabricCollector(BaseCollector):
    """L1 collector: network and Redis availability."""

    name = "fabric"
    layer_id = 1

    def collect(self) -> Dict[str, Any]:
        redis_ok = self._probe_port("127.0.0.1", 6379, timeout=0.2)
        payload = {
            "layer": "L1",
            "layer_id": self.layer_id,
            "timestamp": int(time.time()),
            "redis_up": float(redis_ok),
        }
        return payload

    @staticmethod
    def _probe_port(host: str, port: int, timeout: float) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
