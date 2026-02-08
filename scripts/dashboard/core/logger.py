import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict


class JsonLogger:
    """Structured logging for dashboard components."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def info(self, message: str, **fields: Dict[str, Any]) -> None:
        self._logger.info(self._format(message, fields))

    def warning(self, message: str, **fields: Dict[str, Any]) -> None:
        self._logger.warning(self._format(message, fields))

    def error(self, message: str, **fields: Dict[str, Any]) -> None:
        self._logger.error(self._format(message, fields))

    def _format(self, message: str, fields: Dict[str, Any]) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "fields": fields,
        }
        return json.dumps(payload, ensure_ascii=True)
