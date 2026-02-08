import os
import shutil
import time
from typing import Any, Dict

from dashboard.collectors.base import BaseCollector


class HardwareCollector(BaseCollector):
    """L0 collector: CPU, thermal, disk metrics."""

    name = "hardware"
    layer_id = 0

    def collect(self) -> Dict[str, Any]:
        load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        disk = shutil.disk_usage("/")
        payload = {
            "layer": "L0",
            "layer_id": self.layer_id,
            "timestamp": int(time.time()),
            "cpu_load_1m": float(load),
            "disk_free_gb": disk.free / (1024**3),
            "disk_used_ratio": disk.used / max(1, disk.total),
        }
        return payload
