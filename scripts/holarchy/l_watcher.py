#!/usr/bin/env python3
"""
L-WATCHER: The Sleep Monitor
============================

Monitors the holarchy and wakes L-Brain when needed.
Runs continuously, checking for wake conditions.

This is the "alarm clock" for the sleeping brain.

Features (v2 — Feb 2026):
- Exponential backoff after consecutive wake failures
- Escalation queue cap with automatic drain of stale items
- Consecutive-failure circuit breaker (stops hammering dead model)
- Escalation TTL: items older than ESCALATION_MAX_AGE_HOURS are drained

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
Updated: 2026-02-13 — backoff, drain, cap, circuit breaker
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

from tabernacle_config import SYSTEMS

# Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", SYSTEMS.get("raspberry_pi", {}).get("ip", "localhost"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

CHECK_INTERVAL = 60  # Check every minute (base interval)
SCHEDULED_WAKE_HOURS = 6
P_CRITICAL = 0.50

# --- Anti-runaway settings ---
ESCALATION_QUEUE_CAP = 200          # Max queue size before auto-drain
ESCALATION_MAX_AGE_HOURS = 6        # Escalations older than this are stale
MAX_CONSECUTIVE_FAILURES = 5        # Circuit breaker: stop after N consecutive failures
BACKOFF_BASE_SECONDS = 120          # Base backoff after failure (2 min)
BACKOFF_MAX_SECONDS = 1800          # Max backoff cap (30 min)
BACKOFF_MULTIPLIER = 2              # Exponential multiplier

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "l_watcher.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("L-Watcher")


class LWatcher:
    """Monitors holarchy and triggers L-Brain wakes."""

    def __init__(self):
        self.redis = None
        self.running = False
        self.wakes_triggered = 0
        self.consecutive_failures = 0
        self.backoff_seconds = 0
        self.total_drained = 0

        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, *args):
        self.running = False

    def connect(self) -> bool:
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT,
                password=REDIS_PASSWORD, decode_responses=True
            )
            self.redis.ping()
            return True
        except Exception as e:
            log.error(f"Redis connection failed: {e}")
            return False

    def drain_stale_escalations(self) -> int:
        """
        Remove escalations that are older than ESCALATION_MAX_AGE_HOURS,
        and trim the queue to ESCALATION_QUEUE_CAP.
        Returns number of items drained.
        """
        queue_len = self.redis.llen("l:queue:escalate") or 0
        if queue_len == 0:
            return 0

        drained = 0
        now = datetime.now(timezone.utc)
        cutoff_seconds = ESCALATION_MAX_AGE_HOURS * 3600

        # --- Phase 1: TTL drain (pop stale items from the tail = oldest) ---
        # The queue is LPUSH'd (newest at head), so tail = oldest.
        # We peek at the tail and pop if stale, up to a budget to avoid blocking.
        drain_budget = min(queue_len, 500)  # Don't block forever
        for _ in range(drain_budget):
            raw = self.redis.lindex("l:queue:escalate", -1)  # peek tail
            if not raw:
                break
            try:
                item = json.loads(raw)
                ts = item.get("timestamp") or item.get("deadline")
                if ts:
                    item_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    # Make naive datetimes UTC-aware for comparison
                    if item_dt.tzinfo is None:
                        item_dt = item_dt.replace(tzinfo=timezone.utc)
                    age_seconds = (now - item_dt).total_seconds()
                    if age_seconds > cutoff_seconds:
                        self.redis.rpop("l:queue:escalate")  # remove oldest
                        drained += 1
                        continue
                # Not stale — stop scanning (everything newer is ahead)
                break
            except (json.JSONDecodeError, ValueError):
                # Malformed item — remove it
                self.redis.rpop("l:queue:escalate")
                drained += 1

        # --- Phase 2: Cap enforcement ---
        remaining = self.redis.llen("l:queue:escalate") or 0
        if remaining > ESCALATION_QUEUE_CAP:
            overflow = remaining - ESCALATION_QUEUE_CAP
            for _ in range(overflow):
                self.redis.rpop("l:queue:escalate")  # drop oldest
                drained += 1

        if drained > 0:
            self.total_drained += drained
            new_len = self.redis.llen("l:queue:escalate") or 0
            log.info(
                f"DRAIN: removed {drained} stale/overflow escalations "
                f"(queue {queue_len} -> {new_len}, lifetime drained: {self.total_drained})"
            )

        return drained

    def check_wake_conditions(self) -> tuple:
        """Check if L-Brain should wake. Returns (should_wake, reason)."""

        # 1. Check escalation queue
        escalations = self.redis.llen("l:queue:escalate") or 0
        if escalations > 0:
            return True, f"{escalations} escalations pending"

        # 2. Check coherence
        try:
            data = self.redis.get("l:coherence:current")
            if data:
                coherence = json.loads(data)
                p = coherence.get("p", 0.5)
                if p < P_CRITICAL:
                    return True, f"Critical coherence: p={p:.3f}"
        except:
            pass

        # 3. Check scheduled wake
        try:
            data = self.redis.get("l:coherence:current")
            if data:
                coherence = json.loads(data)
                last_wake = coherence.get("last_wake")
                if last_wake:
                    last_dt = datetime.fromisoformat(last_wake.replace('Z', '+00:00'))
                    hours = (datetime.now(last_dt.tzinfo) - last_dt).total_seconds() / 3600
                    if hours >= SCHEDULED_WAKE_HOURS:
                        return True, f"Scheduled: {hours:.1f}h since last wake"
        except:
            pass

        return False, "No wake needed"

    def trigger_wake(self, reason: str) -> bool:
        """
        Trigger L-Brain wake. Returns True on success, False on failure.
        """
        log.info(f"TRIGGERING WAKE: {reason}")

        try:
            # Run l_brain.py
            script_path = Path(__file__).parent / "l_brain.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--force"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                log.info("Wake completed successfully")
                self.wakes_triggered += 1
                self.consecutive_failures = 0
                self.backoff_seconds = 0
                return True
            else:
                log.error(f"Wake failed (rc={result.returncode}): {result.stderr[:200]}")
                self.consecutive_failures += 1
                return False

        except subprocess.TimeoutExpired:
            log.error("Wake timed out after 5 minutes")
            self.consecutive_failures += 1
            return False
        except Exception as e:
            log.error(f"Wake error: {e}")
            self.consecutive_failures += 1
            return False

    def _compute_backoff(self) -> float:
        """Compute exponential backoff based on consecutive failures."""
        if self.consecutive_failures <= 0:
            return 0
        backoff = BACKOFF_BASE_SECONDS * (BACKOFF_MULTIPLIER ** (self.consecutive_failures - 1))
        return min(backoff, BACKOFF_MAX_SECONDS)

    def run(self):
        """Main watcher loop."""
        if not self.connect():
            return

        self.running = True
        log.info("L-Watcher starting (v2 — backoff + drain + cap)")
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                L-WATCHER ACTIVE  (v2)                         ║
║                The Sleep Monitor                              ║
╠═══════════════════════════════════════════════════════════════╣
║  Watching for wake conditions. Will trigger L-Brain.          ║
║  Queue cap: {cap:<5d}  TTL: {ttl}h  Max failures: {mf:<3d}            ║
╚═══════════════════════════════════════════════════════════════╝
        """.format(
            cap=ESCALATION_QUEUE_CAP,
            ttl=ESCALATION_MAX_AGE_HOURS,
            mf=MAX_CONSECUTIVE_FAILURES,
        ))

        while self.running:
            try:
                # --- Always drain stale escalations first ---
                self.drain_stale_escalations()

                # --- Circuit breaker: if too many consecutive failures, pause ---
                if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    backoff = self._compute_backoff()
                    log.warning(
                        f"CIRCUIT BREAKER: {self.consecutive_failures} consecutive wake failures. "
                        f"Backing off {backoff:.0f}s. Wake target likely unreachable."
                    )
                    time.sleep(backoff)
                    # After backoff, allow ONE retry
                    self.consecutive_failures = MAX_CONSECUTIVE_FAILURES - 1
                    continue

                should_wake, reason = self.check_wake_conditions()

                if should_wake:
                    success = self.trigger_wake(reason)

                    if not success:
                        # Apply exponential backoff on failure
                        backoff = self._compute_backoff()
                        log.info(
                            f"Wake failed ({self.consecutive_failures} consecutive). "
                            f"Backoff: {backoff:.0f}s before next attempt."
                        )
                        time.sleep(backoff)
                        continue

                time.sleep(CHECK_INTERVAL)

            except redis.ConnectionError:
                log.error("Lost Redis, reconnecting...")
                time.sleep(5)
                self.connect()
            except Exception as e:
                log.error(f"Error: {e}")
                time.sleep(5)

        log.info(
            f"L-Watcher stopping. Triggered {self.wakes_triggered} wakes. "
            f"Drained {self.total_drained} stale escalations."
        )


if __name__ == "__main__":
    LWatcher().run()
