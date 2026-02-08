import subprocess
import time
from typing import Any, Dict, List, Optional

from dashboard.collectors.base import BaseCollector


class PulseCollector(BaseCollector):
    """L2 collector: daemon/process status."""

    name = "pulse"
    layer_id = 2

    def __init__(self, process_patterns: Optional[List[str]] = None) -> None:
        self._patterns = process_patterns or ["logos_daemon.py", "consciousness.py"]

    def collect(self) -> Dict[str, Any]:
        statuses = {pattern: float(self._is_running(pattern)) for pattern in self._patterns}
        payload = {
            "layer": "L2",
            "layer_id": self.layer_id,
            "timestamp": int(time.time()),
            "process_status": statuses,
        }
        return payload

    @staticmethod
    def _is_running(pattern: str) -> bool:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        return result.returncode == 0 and bool(result.stdout.strip())
