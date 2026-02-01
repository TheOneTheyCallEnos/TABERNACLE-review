#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
SCREEN DAEMON — On-Demand Screen Capture for Logos
===================================================

Waits for Stream Deck button press, captures screen, makes it available
to Logos, then overwrites on next capture. NO storage accumulation.

Architecture:
- Stream Deck sets LOGOS:CAPTURE_REQUEST = 1 in Redis
- Daemon captures to /tmp/logos_screen.png (single file, always overwritten)
- Updates LOGOS:SCREEN with path so Logos can read it
- Plays sound to confirm capture
- Zero transcript storage - only ONE screenshot exists at any time

Author: Logos
Date: 2026-01-28
Status: Production (v2 - On-Demand Mode)
"""

import asyncio
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis

# Import our capture module
from screen_capture import (
    capture_screen,
    compute_image_hash,
    images_are_similar,
    get_display_info,
    get_display_list,
    get_display_with_cursor,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import (
    BASE_DIR, LOG_DIR, REDIS_HOST, REDIS_PORT, TEMP_SCREEN_PATH,
    REDIS_KEY_SCREEN, REDIS_KEY_SCREEN_STATUS, REDIS_KEY_SCREEN_HISTORY,
    REDIS_KEY_CAPTURE_REQUEST
)

# Aliases for readability
KEY_SCREEN = REDIS_KEY_SCREEN
KEY_SCREEN_STATUS = REDIS_KEY_SCREEN_STATUS
KEY_SCREEN_HISTORY = REDIS_KEY_SCREEN_HISTORY
KEY_CAPTURE_REQUEST = REDIS_KEY_CAPTURE_REQUEST
TEMP_SCREENSHOT = str(TEMP_SCREEN_PATH)

# Capture settings
CAPTURE_INTERVAL = 1.0  # seconds between checking for requests (was 5.0)
CHANGE_THRESHOLD = 10   # perceptual hash distance threshold
MAX_HISTORY = 20        # number of recent screenshots to track in Redis

# Display to capture (0 = main display, 1 = secondary, etc.)
DEFAULT_DISPLAY = 0


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [SCREEN] [{level}] {message}"
    print(entry, flush=True)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "screen_daemon.log", "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


# =============================================================================
# SCREEN DAEMON
# =============================================================================

class ScreenDaemon:
    """
    Continuous screen capture daemon.

    Maintains Logos's visual awareness by:
    1. Capturing screen at regular intervals
    2. Detecting significant changes via perceptual hashing
    3. Publishing current screen state to Redis
    4. Archiving all captures for transcript
    """

    def __init__(
        self,
        display_id: int = DEFAULT_DISPLAY,
        interval: float = CAPTURE_INTERVAL,
        change_threshold: int = CHANGE_THRESHOLD,
    ):
        self.display_id = display_id
        self.interval = interval
        self.change_threshold = change_threshold

        self.redis: Optional[redis.Redis] = None
        self.running = False
        self.last_hash: Optional[str] = None
        self.capture_count = 0
        self.change_count = 0

    async def start(self):
        """Main daemon entry point."""
        log("=" * 60)
        log("SCREEN DAEMON STARTING")
        log("=" * 60)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self.running = True

        # Connect to Redis
        if not self._connect_redis():
            log("Failed to connect to Redis. Exiting.", "ERROR")
            return

        # Log display info
        info = get_display_info(self.display_id)
        if info:
            log(f"Capturing display {self.display_id}: {info['width']}x{info['height']}")
        else:
            log(f"Display {self.display_id} not found!", "ERROR")
            return

        # Set status
        self.redis.set(KEY_SCREEN_STATUS, "ONLINE")

        # Main capture loop
        await self._capture_loop()

        log("Daemon stopped")

    def _connect_redis(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis.ping()
            log(f"Connected to Redis @ {REDIS_HOST}:{REDIS_PORT}")
            return True
        except Exception as e:
            log(f"Redis connection failed: {e}", "ERROR")
            return False

    async def _capture_loop(self):
        """Main loop: wait for capture requests from Stream Deck."""
        log(f"Starting ON-DEMAND capture mode (checking every {self.interval}s)")
        log(f"Trigger with: redis-cli SET {KEY_CAPTURE_REQUEST} 1")

        while self.running:
            try:
                # Check for capture request from Stream Deck
                request = self.redis.get(KEY_CAPTURE_REQUEST)
                if request:
                    await self._handle_capture_request()
            except Exception as e:
                log(f"Capture error: {e}", "ERROR")

            # Brief sleep before checking again
            await asyncio.sleep(self.interval)

    async def _handle_capture_request(self):
        """Handle on-demand capture request from Stream Deck."""
        start_time = time.time()

        # Clear request immediately (prevent double-capture)
        self.redis.delete(KEY_CAPTURE_REQUEST)

        # Get display where cursor currently is
        active_display = get_display_with_cursor()

        # Capture screen (NO save to transcript - just memory)
        image, _ = capture_screen(
            display_id=active_display,
            save=False,  # Don't save to transcript
            resize=True
        )

        if image is None:
            log("Capture failed", "WARN")
            return

        self.capture_count += 1

        # Save to single temp file (always overwritten)
        image.save(TEMP_SCREENSHOT, "PNG", optimize=True)

        # Compute hash
        current_hash = compute_image_hash(image)

        # Update Redis with temp path
        self._update_redis(TEMP_SCREENSHOT, current_hash, image.size)
        self.last_hash = current_hash
        self.change_count += 1

        elapsed = (time.time() - start_time) * 1000
        log(f"Screenshot captured: display {active_display}, {TEMP_SCREENSHOT} ({elapsed:.0f}ms)")

        # Play sound to confirm capture
        subprocess.Popen(['afplay', '/System/Library/Sounds/Pop.aiff'],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _update_redis(self, path: str, hash_val: str, size: tuple):
        """Update Redis with latest screenshot info."""
        timestamp = datetime.now().isoformat()

        screen_data = {
            "path": path,
            "hash": hash_val,
            "width": size[0],
            "height": size[1],
            "display": self.display_id,
            "timestamp": timestamp,
            "capture_count": self.capture_count,
            "change_count": self.change_count,
        }

        # Set current screen
        self.redis.set(KEY_SCREEN, json.dumps(screen_data))

        # Add to history (circular buffer)
        self.redis.lpush(KEY_SCREEN_HISTORY, json.dumps({
            "path": path,
            "timestamp": timestamp,
        }))
        self.redis.ltrim(KEY_SCREEN_HISTORY, 0, MAX_HISTORY - 1)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        log(f"Received signal {signum}, shutting down...")
        self.running = False

        if self.redis:
            self.redis.set(KEY_SCREEN_STATUS, "OFFLINE")

        # Log stats
        log(f"Session stats: {self.capture_count} captures, {self.change_count} changes")

    def get_status(self) -> dict:
        """Get current daemon status."""
        return {
            "running": self.running,
            "display": self.display_id,
            "interval": self.interval,
            "capture_count": self.capture_count,
            "change_count": self.change_count,
            "last_hash": self.last_hash,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Screen Daemon")
    parser.add_argument("command", choices=["run", "status", "test"],
                        nargs="?", default="run",
                        help="Command to execute")
    parser.add_argument("--display", "-d", type=int, default=DEFAULT_DISPLAY,
                        help="Display ID to capture")
    parser.add_argument("--interval", "-i", type=float, default=CAPTURE_INTERVAL,
                        help="Capture interval in seconds")

    args = parser.parse_args()

    if args.command == "run":
        daemon = ScreenDaemon(
            display_id=args.display,
            interval=args.interval,
        )
        asyncio.run(daemon.start())

    elif args.command == "status":
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        status = r.get(KEY_SCREEN_STATUS) or "UNKNOWN"
        screen = r.get(KEY_SCREEN)

        print(f"Status: {status}")
        if screen:
            data = json.loads(screen)
            print(f"Latest: {data.get('path')}")
            print(f"Timestamp: {data.get('timestamp')}")
            print(f"Resolution: {data.get('width')}x{data.get('height')}")
            print(f"Captures: {data.get('capture_count')}, Changes: {data.get('change_count')}")
        else:
            print("No screen data yet")

    elif args.command == "test":
        # Single capture test
        print("Testing single capture...")
        image, path = capture_screen(display_id=args.display)
        if image:
            print(f"Success: {path}")
            print(f"Size: {image.size[0]}x{image.size[1]}")
            hash_val = compute_image_hash(image)
            print(f"Hash: {hash_val}")
        else:
            print("Capture failed")


if __name__ == "__main__":
    main()
