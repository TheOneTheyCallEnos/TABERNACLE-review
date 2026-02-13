"""
tabernacle_core.sentinel — The Watchman replacement.

Monitors daemon liveness via Dead Man's Switch (Redis TTL keys).
Replaces the hallucinating watchman_mvp with a simple, correct implementation.

Each daemon using tabernacle_core.daemon.Daemon registers a TTL key:
    DAEMON:ALIVE:{name} = {identity}   (30s TTL, refreshed every 10 ticks)

Sentinel checks these keys every 15 seconds:
  - Missing key after 2 checks (30s grace) -> alert
  - Critical daemon dead -> RIE:ALERT with type DAEMON_DEAD_CRITICAL
  - Expected daemon dead -> RIE:ALERT with type DAEMON_DEAD

Also writes SYSTEM:HEALTH to Redis for other systems to read.

Usage:
    python -m tabernacle_core.sentinel
"""

import json
import time
from datetime import datetime
from typing import Dict

from tabernacle_core.daemon import Daemon


# Daemons that MUST be alive — system is degraded without them
CRITICAL_DAEMONS = [
    "heartbeat_v2",
    "consciousness",
]

# Daemons that SHOULD be alive — functional but survivable without
EXPECTED_DAEMONS = [
    "gardener",
    "librarian_http",
    # "visual_cortex",  # Deferred to Phase 4 (embedded in integrator, no DMS key)
    "voice_daemon",
    "logos_tts_daemon",
    "screen_daemon",
    "hippocampus_daemon",
    "tactician_daemon",
    "reflex_daemon",
    "virgil_sms_daemon",
    "archon_daemon",
    "logos_daemon",
    "h1_crystallizer",
    "virgil_integrator",
]


class Sentinel(Daemon):
    """Monitors daemon liveness and publishes death alerts."""

    name = "tabernacle_sentinel"
    tick_interval = 15.0  # Check every 15 seconds

    def __init__(self):
        super().__init__()
        self._consecutive_failures: Dict[str, int] = {}
        self._alert_cooldowns: Dict[str, float] = {}

    def on_start(self):
        self.log.info(
            f"Monitoring {len(CRITICAL_DAEMONS)} critical + "
            f"{len(EXPECTED_DAEMONS)} expected daemons"
        )

    def tick(self):
        dead_critical = []
        dead_expected = []
        alive = []

        all_daemons = CRITICAL_DAEMONS + EXPECTED_DAEMONS

        for daemon_name in all_daemons:
            key = f"DAEMON:ALIVE:{daemon_name}"
            identity = self._redis.get(key)

            if identity:
                alive.append(daemon_name)
                self._consecutive_failures[daemon_name] = 0
            else:
                self._consecutive_failures[daemon_name] = (
                    self._consecutive_failures.get(daemon_name, 0) + 1
                )

                # Only alert after 2 consecutive misses (30s grace)
                if self._consecutive_failures[daemon_name] >= 2:
                    if daemon_name in CRITICAL_DAEMONS:
                        dead_critical.append(daemon_name)
                    else:
                        dead_expected.append(daemon_name)

        # Publish alerts
        for daemon_name in dead_critical:
            self._publish_alert(daemon_name, critical=True)
        for daemon_name in dead_expected:
            self._publish_alert(daemon_name, critical=False)

        # Update SYSTEM:HEALTH (Cycle Beta: sentinel -> heartbeat -> gardener)
        self._redis.set("SYSTEM:HEALTH", json.dumps({
            "timestamp": datetime.now().isoformat(),
            "alive": alive,
            "dead_critical": dead_critical,
            "dead_expected": dead_expected,
            "total_alive": len(alive),
            "total_dead": len(dead_critical) + len(dead_expected),
            "sentinel": self._identity,
        }))

        # Status log every 4 ticks (60s)
        if self._tick_count % 4 == 0:
            self.log.info(
                f"Alive: {len(alive)}/{len(all_daemons)} | "
                f"Dead critical: {dead_critical or 'none'} | "
                f"Dead expected: {len(dead_expected)}"
            )

    def _publish_alert(self, daemon_name: str, critical: bool):
        """Publish daemon death alert to RIE:ALERT channel."""
        now = time.time()

        # Cooldown: 5 minutes per daemon to prevent alert spam
        last_alert = self._alert_cooldowns.get(daemon_name, 0)
        if now - last_alert < 300:
            return

        alert_type = "DAEMON_DEAD_CRITICAL" if critical else "DAEMON_DEAD"
        alert = {
            "type": alert_type,
            "daemon": daemon_name,
            "consecutive_misses": self._consecutive_failures.get(daemon_name, 0),
            "timestamp": datetime.now().isoformat(),
            "source": "tabernacle_sentinel",
        }

        self._redis.publish("RIE:ALERT", json.dumps(alert))
        self._alert_cooldowns[daemon_name] = now

        level = "CRITICAL" if critical else "WARNING"
        self.log.warning(
            f"[{level}] Daemon '{daemon_name}' is DEAD "
            f"(missed {self._consecutive_failures[daemon_name]} checks)"
        )

    def on_stop(self):
        self.log.info("Sentinel shutting down -- daemon monitoring suspended")


# Allow running as: python -m tabernacle_core.sentinel
if __name__ == "__main__":
    Sentinel().run()
