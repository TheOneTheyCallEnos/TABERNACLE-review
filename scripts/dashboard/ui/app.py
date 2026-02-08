import sqlite3
from typing import List, Tuple

from dashboard.config import DashboardConfig


def render_console(limit: int = 10) -> None:
    """Minimal console UI: prints latest metrics."""
    config = DashboardConfig()
    rows = _fetch_latest(config.db_path, limit)
    for ts, layer_id, metric_key, value in rows:
        print(f"{ts} L{layer_id} {metric_key}={value}")


def _fetch_latest(db_path: str, limit: int) -> List[Tuple[int, int, str, float]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT ts, layer_id, metric_key, value FROM raw ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


if __name__ == "__main__":
    render_console()
