import time
from typing import Any, Dict, Iterable, List

from dashboard.config import DashboardConfig
from dashboard.core.bus import EventBus
from dashboard.core.logger import JsonLogger
from dashboard.processors.anomaly import AnomalyDetector
from dashboard.processors.coherence import CoherenceAggregator
from dashboard.processors.predictor import TrendPredictor
from dashboard.storage.db import DashboardStore


class DashboardEngine:
    """
    Main loop: collectors -> processors -> storage.

    Data flow:
    collectors emit raw layer metrics -> processors derive coherence/anomaly/trend ->
    storage persists raw + aggregates -> API/UI consume stored values.
    """

    def __init__(self, collectors: Iterable[Any], config: DashboardConfig) -> None:
        self._collectors = list(collectors)
        self._config = config
        self._bus = EventBus()
        self._logger = JsonLogger("dashboard.engine")
        self._store = DashboardStore(config)
        self._coherence = CoherenceAggregator(config)
        self._anomaly = AnomalyDetector(config)
        self._predictor = TrendPredictor(config)

    def run(self) -> None:
        self._logger.info("dashboard_engine_start")
        while True:
            cycle_started = time.time()
            for collector in self._collectors:
                try:
                    payload = collector.collect()
                except Exception as exc:
                    self._logger.error("collector_failed", collector=collector.name, error=str(exc))
                    continue

                # Raw data -> storage
                self._store.insert_raw(payload)

                # Processed signals
                coherence = self._coherence.aggregate(payload)
                anomaly = self._anomaly.detect(payload)
                trend = self._predictor.predict(payload)

                enriched = {
                    "raw": payload,
                    "coherence": coherence,
                    "anomaly": anomaly,
                    "trend": trend,
                }
                self._store.insert_processed(enriched)
                self._bus.publish("telemetry", enriched)

            elapsed = time.time() - cycle_started
            sleep_for = max(0.0, self._config.collection_interval_s - elapsed)
            time.sleep(sleep_for)
