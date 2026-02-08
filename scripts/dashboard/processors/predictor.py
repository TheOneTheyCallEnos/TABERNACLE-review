from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Tuple

from dashboard.config import DashboardConfig


class TrendPredictor:
    """Linear regression trend estimator for metrics."""

    def __init__(self, config: DashboardConfig, window: int = 60) -> None:
        self._window = window
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        trends: Dict[str, Dict[str, float]] = {}
        for key, value in _numeric_items(payload):
            series = self._history[key]
            series.append(value)
            slope, intercept, confidence = _linear_regression(list(series))
            trends[key] = {
                "slope": slope,
                "intercept": intercept,
                "confidence": confidence,
            }
        return {
            "layer": payload.get("layer"),
            "layer_id": payload.get("layer_id"),
            "timestamp": payload.get("timestamp"),
            "trends": trends,
        }


def _linear_regression(values: List[float]) -> Tuple[float, float, float]:
    n = len(values)
    if n < 2:
        return 0.0, values[0] if values else 0.0, 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs) or 1e-6
    slope = num / den
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in values) or 1e-6
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, values))
    r2 = max(0.0, 1.0 - (ss_res / ss_tot))
    return slope, intercept, r2


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
