import math
import sqlite3
import time
from typing import Dict, List, Tuple

from dashboard.config import DashboardConfig


def rollup_task(config: DashboardConfig) -> None:
    """
    Aggregate raw -> hourly -> daily -> weekly and prune old data.

    Preserves min/max/avg/stddev for each period and deletes expired rows.
    """
    now = int(time.time())
    db_path = config.db_path
    with sqlite3.connect(db_path) as conn:
        _rollup(conn, "raw", "hourly", 3600, now - 24 * 3600)
        _rollup(conn, "hourly", "daily", 86400, now - 7 * 86400)
        _rollup(conn, "daily", "weekly", 7 * 86400, now - 90 * 86400)

        conn.execute("DELETE FROM raw WHERE ts < ?", (now - 24 * 3600,))
        conn.execute("DELETE FROM hourly WHERE ts < ?", (now - 7 * 86400,))
        conn.execute("DELETE FROM daily WHERE ts < ?", (now - 90 * 86400,))


def detect_anomaly(config: DashboardConfig, layer_id: int, metric_key: str) -> bool:
    """Z-score anomaly detection for a specific metric."""
    values = _load_series(config.db_path, layer_id, metric_key, hours=24)
    if len(values) < 5:
        return False
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    stddev = math.sqrt(variance) or 1e-6
    z = (values[-1] - mean_val) / stddev
    return abs(z) > config.anomaly_z_threshold


def predict_failure(config: DashboardConfig, layer_id: int, metric_key: str) -> float:
    """
    Linear regression on last N points.

    Returns hours until critical threshold (0.5) or -1 if stable.
    """
    series = _load_series(config.db_path, layer_id, metric_key, hours=24)
    if len(series) < 5:
        return -1.0
    slope, intercept, _ = _linear_regression(series)
    if slope >= 0:
        return -1.0
    critical = 0.5
    t_index = (critical - intercept) / slope
    if t_index <= len(series):
        return 0.0
    hours_between = 24 / max(1, len(series))
    return max(0.0, (t_index - len(series)) * hours_between)


def get_trend(
    config: DashboardConfig, layer_id: int, metric_key: str, window_hours: int = 24
) -> Dict[str, float]:
    """Return slope, direction, confidence for recent window."""
    series = _load_series(config.db_path, layer_id, metric_key, hours=window_hours)
    slope, _, confidence = _linear_regression(series)
    direction = 1.0 if slope > 0 else -1.0 if slope < 0 else 0.0
    return {"slope": slope, "direction": direction, "confidence": confidence}


def _load_series(db_path: str, layer_id: int, metric_key: str, hours: int) -> List[float]:
    since = int(time.time()) - (hours * 3600)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT value FROM raw WHERE layer_id=? AND metric_key=? AND ts >= ? ORDER BY ts ASC",
            (layer_id, metric_key, since),
        )
        return [row[0] for row in cursor.fetchall()]


def _rollup(
    conn: sqlite3.Connection,
    source: str,
    target: str,
    bucket_seconds: int,
    since_ts: int,
) -> None:
    cursor = conn.execute(
        f"""
        SELECT (ts / ?) * ? as bucket,
               layer_id,
               metric_key,
               MIN(value),
               MAX(value),
               AVG(value),
               COUNT(*)
        FROM {source}
        WHERE ts >= ?
        GROUP BY bucket, layer_id, metric_key
        """,
        (bucket_seconds, bucket_seconds, since_ts),
    )
    rows = cursor.fetchall()
    for bucket, layer_id, metric_key, min_val, max_val, avg_val, count in rows:
        stddev = _stddev(conn, source, bucket, bucket_seconds, layer_id, metric_key, avg_val)
        conn.execute(
            f"""
            INSERT INTO {target} (ts, layer_id, metric_key, min, max, avg, stddev, count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(bucket), layer_id, metric_key, min_val, max_val, avg_val, stddev, count),
        )


def _stddev(
    conn: sqlite3.Connection,
    source: str,
    bucket: int,
    bucket_seconds: int,
    layer_id: int,
    metric_key: str,
    avg_val: float,
) -> float:
    cursor = conn.execute(
        f"""
        SELECT value FROM {source}
        WHERE ts >= ? AND ts < ? AND layer_id=? AND metric_key=?
        """,
        (bucket, bucket + bucket_seconds, layer_id, metric_key),
    )
    values = [row[0] for row in cursor.fetchall()]
    if not values:
        return 0.0
    variance = sum((v - avg_val) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


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
