from collections import defaultdict, deque
from statistics import mean, pstdev
from typing import Any, Deque, Dict, Tuple

from dashboard.config import DashboardConfig


class AnomalyDetector:
    """Z-score anomaly detection for numeric metrics."""

    def __init__(self, config: DashboardConfig, window: int = 60) -> None:
        self._threshold = config.anomaly_z_threshold
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def detect(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        anomalies: Dict[str, float] = {}
        for key, value in _numeric_items(payload):
            history = self._history[key]
            history.append(value)
            if len(history) < 5:
                continue
            avg = mean(history)
            std = pstdev(history) or 1e-6
            z = (value - avg) / std
            if abs(z) >= self._threshold:
                anomalies[key] = z
        return {
            "layer": payload.get("layer"),
            "layer_id": payload.get("layer_id"),
            "timestamp": payload.get("timestamp"),
            "anomalies": anomalies,
        }


def _numeric_items(payload: Dict[str, Any]) -> Tuple[str, float]:
    for key, value in payload.items():
        if key in {"timestamp", "layer", "layer_id"}:
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    yield f"{key}.{sub_key}", float(sub_val)
