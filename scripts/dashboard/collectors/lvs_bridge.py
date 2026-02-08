import json
import time
from pathlib import Path
from typing import Any, Dict

from dashboard.collectors.base import BaseCollector
from dashboard.config import DashboardConfig


class LVSBridgeCollector(BaseCollector):
    """Collector that ingests holarchy_audit output."""

    name = "lvs_bridge"
    layer_id = 5

    def __init__(self, config: DashboardConfig) -> None:
        self._path = Path(config.lvs_output_path)

    def collect(self) -> Dict[str, Any]:
        data = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        payload = {
            "layer": "L5",
            "layer_id": self.layer_id,
            "timestamp": int(time.time()),
            "lvs": data,
        }
        return payload
