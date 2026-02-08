from collections import defaultdict
from typing import Any, Callable, Dict, List


EventHandler = Callable[[Dict[str, Any]], None]


class EventBus:
    """Simple in-process pub/sub bus for telemetry events."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        for handler in self._subscribers.get(event_type, []):
            handler(payload)
