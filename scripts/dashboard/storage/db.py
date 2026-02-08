import sqlite3
from pathlib import Path
from typing import Any, Dict

from dashboard.config import DashboardConfig


class DashboardStore:
    """SQLite adapter for dashboard telemetry storage."""

    def __init__(self, config: DashboardConfig) -> None:
        self._config = config
        self._db_path = Path(config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        schema = Path(self._config.schema_path)
        schema.parent.mkdir(parents=True, exist_ok=True)
        if schema.exists():
            ddl = schema.read_text(encoding="utf-8")
        else:
            ddl = ""
        with sqlite3.connect(self._db_path) as conn:
            if ddl:
                conn.executescript(ddl)

    def insert_raw(self, payload: Dict[str, Any]) -> None:
        ts = int(payload.get("timestamp", 0))
        layer_id = int(payload.get("layer_id", -1))
        items = _flatten_metrics(payload)
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT INTO raw (ts, layer_id, metric_key, value) VALUES (?, ?, ?, ?)",
                [(ts, layer_id, key, value) for key, value in items],
            )

    def insert_processed(self, payload: Dict[str, Any]) -> None:
        ts = int(payload.get("raw", {}).get("timestamp", 0))
        layer_id = int(payload.get("raw", {}).get("layer_id", -1))
        derived = {}
        if "coherence" in payload:
            derived.update({"coherence.cpgi": payload["coherence"].get("cpgi", 0.0)})
        if "anomaly" in payload:
            for key, value in payload["anomaly"].get("anomalies", {}).items():
                derived[f"anomaly.{key}"] = value
        if "trend" in payload:
            for key, trend in payload["trend"].get("trends", {}).items():
                derived[f"trend.{key}.slope"] = trend.get("slope", 0.0)
                derived[f"trend.{key}.confidence"] = trend.get("confidence", 0.0)
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT INTO raw (ts, layer_id, metric_key, value) VALUES (?, ?, ?, ?)",
                [(ts, layer_id, key, float(value)) for key, value in derived.items()],
            )


def _flatten_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    for key, value in payload.items():
        if key in {"timestamp", "layer", "layer_id"}:
            continue
        if isinstance(value, (int, float)):
            flattened[key] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    flattened[f"{key}.{sub_key}"] = float(sub_val)
    return flattened
