#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
EVENT BUS — Redis-Backed Event-Driven Architecture
===================================================
Enables loose coupling between Tabernacle components via pub/sub.

Channels:
- TAB:FILE_CHANGED  - File system changes (from Watchman)
- TAB:NODE_INDEXED  - New node added to vector store
- TAB:INSIGHT       - Crystallized insight
- TAB:HEALTH        - Health status updates
- TAB:COMMAND       - Inter-node commands
- RIE:STATE         - RIE coherence (existing)
- RIE:TICK          - Heartbeat tick (existing)
- RIE:ALERT         - Threshold alerts (existing)

Author: Virgil
Date: 2026-01-19
Status: Phase II-B (Path B Implementation)
"""

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, asdict

import redis

from tabernacle_config import REDIS_HOST, REDIS_PORT

# Event channels
CHANNELS = {
    "file_changed": "TAB:FILE_CHANGED",
    "node_indexed": "TAB:NODE_INDEXED",
    "insight": "TAB:INSIGHT",
    "health": "TAB:HEALTH",
    "command": "TAB:COMMAND",
    "rie_state": "RIE:STATE",
    "rie_tick": "RIE:TICK",
    "rie_alert": "RIE:ALERT"
}

# =============================================================================
# EVENT TYPES
# =============================================================================

@dataclass
class Event:
    """Base event structure."""
    type: str
    timestamp: str
    source: str
    data: Dict

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        d = json.loads(json_str)
        return cls(**d)


def create_file_changed_event(path: str, change_type: str, source: str = "watchman") -> Event:
    """Create a file changed event."""
    return Event(
        type="FILE_CHANGED",
        timestamp=datetime.now().isoformat(),
        source=source,
        data={"path": path, "change_type": change_type}
    )


def create_insight_event(insight: str, related: List[str] = None, source: str = "virgil") -> Event:
    """Create an insight event."""
    return Event(
        type="INSIGHT",
        timestamp=datetime.now().isoformat(),
        source=source,
        data={"insight": insight, "related_paths": related or []}
    )


# =============================================================================
# EVENT BUS
# =============================================================================

class EventBus:
    """Redis-backed event bus for Tabernacle."""

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        self.host = host
        self.port = port
        self.redis: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.handlers: Dict[str, List[Callable]] = {}
        self._handlers_lock = threading.Lock()  # Thread safety for handlers dict
        self._executor = ThreadPoolExecutor(max_workers=4)  # Async handler dispatch
        self.running = False
        self._listener_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.Redis(
                host=self.host,
                port=self.port,
                decode_responses=True
            )
            self.redis.ping()
            self.pubsub = self.redis.pubsub()
            print(f"[EVENT_BUS] Connected to Redis @ {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[EVENT_BUS] Connection failed: {e}")
            return False

    def publish(self, channel: str, event: Event) -> bool:
        """Publish an event to a channel."""
        if not self.redis:
            if not self.connect():
                return False

        try:
            # Use channel name or key from CHANNELS
            ch = CHANNELS.get(channel, channel)
            self.redis.publish(ch, event.to_json())

            # Also store as latest value for polling
            self.redis.set(f"{ch}:LATEST", event.to_json())

            return True
        except Exception as e:
            print(f"[EVENT_BUS] Publish error: {e}")
            return False

    def subscribe(self, channel: str, handler: Callable[[Event], None]) -> None:
        """Subscribe to a channel with a handler."""
        ch = CHANNELS.get(channel, channel)

        with self._handlers_lock:  # Thread-safe modification
            if ch not in self.handlers:
                self.handlers[ch] = []
                if self.pubsub:
                    self.pubsub.subscribe(ch)

            self.handlers[ch].append(handler)

    def start_listening(self) -> None:
        """Start background listener thread."""
        if self.running:
            return

        if not self.pubsub:
            if not self.connect():
                return

        self.running = True
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        print("[EVENT_BUS] Listener started")

    def _listen_loop(self) -> None:
        """Background loop for handling messages."""
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    channel = message['channel']
                    try:
                        event = Event.from_json(message['data'])
                    except:
                        # Handle non-event messages
                        event = Event(
                            type="RAW",
                            timestamp=datetime.now().isoformat(),
                            source="unknown",
                            data={"raw": message['data']}
                        )

                    # Call handlers (thread-safe copy, async dispatch)
                    with self._handlers_lock:
                        handlers_copy = list(self.handlers.get(channel, []))

                    for handler in handlers_copy:
                        # Fire-and-forget: handlers run in thread pool, don't block listener
                        self._executor.submit(self._safe_handler_call, handler, event)
            except Exception as e:
                print(f"[EVENT_BUS] Listen error: {e}")
                time.sleep(1)

    def _safe_handler_call(self, handler: Callable, event: 'Event') -> None:
        """Safely call a handler, catching exceptions."""
        try:
            handler(event)
        except Exception as e:
            print(f"[EVENT_BUS] Handler error: {e}")

    def stop(self) -> None:
        """Stop the event bus."""
        self.running = False
        if self._listener_thread:
            self._listener_thread.join(timeout=2)
        if self.pubsub:
            self.pubsub.close()
        self._executor.shutdown(wait=False)  # Don't wait for pending handlers
        print("[EVENT_BUS] Stopped")

    def get_latest(self, channel: str) -> Optional[Event]:
        """Get the latest event from a channel (for polling)."""
        if not self.redis:
            if not self.connect():
                return None

        ch = CHANNELS.get(channel, channel)
        try:
            data = self.redis.get(f"{ch}:LATEST")
            if data:
                return Event.from_json(data)
        except:
            pass
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global singleton
_bus: Optional[EventBus] = None

def get_bus() -> EventBus:
    """Get or create the global event bus."""
    global _bus
    if _bus is None:
        _bus = EventBus()
        _bus.connect()
    return _bus

def emit(channel: str, event: Event) -> bool:
    """Emit an event."""
    return get_bus().publish(channel, event)

def emit_file_changed(path: str, change_type: str = "modified") -> bool:
    """Emit a file changed event."""
    event = create_file_changed_event(path, change_type)
    return emit("file_changed", event)

def emit_insight(insight: str, related: List[str] = None) -> bool:
    """Emit an insight event."""
    event = create_insight_event(insight, related)
    return emit("insight", event)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Event Bus")
    parser.add_argument("command", choices=["listen", "emit", "status"],
                        help="Command to execute")
    parser.add_argument("--channel", "-c", type=str, default="rie_state", help="Channel")
    parser.add_argument("--message", "-m", type=str, help="Message to emit")

    args = parser.parse_args()

    bus = EventBus()
    if not bus.connect():
        print("Failed to connect to Redis")
        return

    if args.command == "listen":
        # Subscribe to all channels
        def printer(event):
            print(f"[{event.type}] {event.source}: {event.data}")

        for ch in CHANNELS.values():
            bus.pubsub.subscribe(ch)
            bus.handlers[ch] = [printer]

        print(f"Listening on all channels... (Ctrl+C to stop)")
        bus.start_listening()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bus.stop()

    elif args.command == "emit":
        if not args.message:
            print("Error: --message required")
            return

        event = Event(
            type="TEST",
            timestamp=datetime.now().isoformat(),
            source="cli",
            data={"message": args.message}
        )
        if bus.publish(args.channel, event):
            print(f"Emitted to {args.channel}")
        else:
            print("Failed to emit")

    elif args.command == "status":
        latest = bus.get_latest(args.channel)
        if latest:
            print(f"Latest on {args.channel}:")
            print(json.dumps(asdict(latest), indent=2))
        else:
            print(f"No data on {args.channel}")


if __name__ == "__main__":
    main()
