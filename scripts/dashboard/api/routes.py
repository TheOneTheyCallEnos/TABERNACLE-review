import sqlite3
import time
from typing import Dict, List


def get_health() -> Dict[str, str]:
    return {"status": "ok", "ts": str(int(time.time()))}


def get_latest_metrics(db_path: str, limit: int = 50) -> List[Dict[str, str]]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT ts, layer_id, metric_key, value FROM raw ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "ts": str(row[0]),
                "layer_id": str(row[1]),
                "metric_key": row[2],
                "value": str(row[3]),
            }
            for row in cursor.fetchall()
        ]
