import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dashboard.config import DashboardConfig
from dashboard.processors.coherence import CoherenceAggregator
from dashboard.storage.db import DashboardStore


class MetaMonitor:
    """
    Meta-monitor for dashboard self-health (L5 holon).

    Methods return structured dicts with drift, latency, completeness,
    and self-healing status.
    """

    DRIFT_THRESHOLD = 0.05
    DATA_STALE_S = 60

    def __init__(self, collectors: Iterable[Any], config: DashboardConfig) -> None:
        self._collectors = list(collectors)
        self._config = config
        self._store = DashboardStore(config)
        self._coherence = CoherenceAggregator(config)
        self._alert_path = Path("scripts/dashboard/data/meta_alert.log")
        self._alert_path.parent.mkdir(parents=True, exist_ok=True)

    def check_accuracy(self) -> Dict[str, Any]:
        displayed = self._latest_cpgi()
        actual = self._actual_cpgi()
        drift = abs(actual - displayed)
        return {
            "displayed": displayed,
            "actual": actual,
            "drift": drift,
            "hallucinating": drift > self.DRIFT_THRESHOLD,
        }

    def check_latency(self) -> Dict[str, Any]:
        last_ts = self._latest_timestamp()
        age = max(0, int(time.time()) - last_ts)
        return {"last_ts": last_ts, "age_s": age, "stale": age > self.DATA_STALE_S}

    def check_completeness(self) -> Dict[str, Any]:
        layers = self._latest_layers()
        missing = [
            layer
            for layer in self._config.layer_ids.keys()
            if layer not in layers
        ]
        return {"layers_present": sorted(layers), "missing_layers": missing, "complete": not missing}

    def heal(self) -> Dict[str, Any]:
        restarted = False
        try:
            subprocess.run(["pkill", "-f", "dashboard"], check=False)
            restarted = True
        except OSError:
            restarted = False
        self._alert_path.write_text(f"{int(time.time())}: heal_attempted\n", encoding="utf-8")
        return {"restarted": restarted, "alert_path": str(self._alert_path)}

    def audit_self(self) -> Dict[str, Any]:
        accuracy = self.check_accuracy()
        latency = self.check_latency()
        completeness = self.check_completeness()
        healthy = (
            not accuracy["hallucinating"]
            and not latency["stale"]
            and completeness["complete"]
        )
        return {
            "layer": "L5",
            "healthy": healthy,
            "accuracy": accuracy,
            "latency": latency,
            "completeness": completeness,
        }

    def _latest_cpgi(self) -> float:
        rows = self._store_query("coherence.cpgi")
        return float(rows[0][0]) if rows else 0.0

    def _latest_timestamp(self) -> int:
        rows = self._store_query("coherence.cpgi", with_ts=True)
        return int(rows[0][0]) if rows else 0

    def _latest_layers(self) -> set:
        rows = self._store_query(None, with_ts=False, select_layer=True)
        return {f"L{row[0]}" for row in rows}

    def _actual_cpgi(self) -> float:
        values = []
        for collector in self._collectors:
            try:
                payload = collector.collect()
                coherence = self._coherence.aggregate(payload)
                values.append(coherence.get("cpgi", 0.0))
            except Exception:
                continue
        return sum(values) / len(values) if values else 0.0

    def _store_query(
        self, metric_key: Optional[str], with_ts: bool = False, select_layer: bool = False
    ):
        if select_layer:
            sql = "SELECT DISTINCT layer_id FROM raw ORDER BY layer_id ASC"
        elif with_ts:
            sql = "SELECT ts FROM raw WHERE metric_key=? ORDER BY ts DESC LIMIT 1"
        else:
            sql = "SELECT value FROM raw WHERE metric_key=? ORDER BY ts DESC LIMIT 1"
        params = () if select_layer else (metric_key,)
        with self._store._db_path.open("rb"):  # ensure db exists
            pass
        return self._store_query_db(sql, params)

    def _store_query_db(self, sql: str, params: tuple):
        import sqlite3

        with sqlite3.connect(self._store._db_path) as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
