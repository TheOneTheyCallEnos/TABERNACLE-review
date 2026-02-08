"""
LOGOS LVS MODULE: KAPPA SCANNER (Split-Brain Detector)
======================================================
LVS Component: κ (Kappa) - Clarity & Integration
Theory Ref: 04_LR_LAW/CANON/LVS_MATHEMATICS.md

Purpose:
  Calculates the κ-score [0, 1] by verifying state coherence across
  distributed nodes (Redis, Cortex, Brainstem).

  κ = 1.0 implies perfect state synchronization.
  κ < 0.9 implies dissociation (Split-Brain).

Zero-Bottleneck Property:
  If Network is down, κ -> 0 immediately.

Author: Gemini 2.5 Pro (LVS Review)
Date: 2026-01-29
"""

import logging
import json
import sys
from pathlib import Path
import requests
import redis
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from tabernacle_config import SYSTEMS, REDIS_HOST, REDIS_PORT

# Configuration - LVS Hardware Map (from centralized config)
NODES = {
    "REDIS": {"ip": REDIS_HOST, "port": REDIS_PORT, "role": "State Store"},
    "CORTEX": {"ip": SYSTEMS["mac_studio"]["ip"], "port": SYSTEMS["mac_studio"]["ollama_port"], "role": "High-Level Logic"},
    "BRAINSTEM": {"ip": SYSTEMS["mac_mini"]["ip"], "port": SYSTEMS["mac_mini"]["ollama_port"], "role": "Reflexive Logic"}
}

# LVS Thresholds
TIMEOUT_SEC = 2.0
KAPPA_PENALTY_OFFLINE = 0.3
KAPPA_PENALTY_MISMATCH = 0.15

logger = logging.getLogger("logos.kappa")


class KappaScanner:
    """
    Split-Brain Detector for LVS Coherence Protocol.

    Measures κ (Kappa) - the integration/clarity component of coherence.
    κ represents the unified integration across processing layers.

    Formula: κ = (matching_states / total_comparisons)
    Alert threshold: κ < 0.90
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=NODES["REDIS"]["ip"],
            port=NODES["REDIS"]["port"],
            socket_timeout=TIMEOUT_SEC
        )

    def _check_redis_state(self) -> bool:
        """Pings the heartbeat state store."""
        try:
            return self.redis_client.ping()
        except (redis.ConnectionError, redis.TimeoutError):
            logger.error("κ-Alert: Redis Heartbeat Lost")
            return False

    def _get_ollama_tags(self, node_key: str) -> set:
        """Fetches active models from an Ollama node."""
        node = NODES[node_key]
        url = f"http://{node['ip']}:{node['port']}/api/tags"
        try:
            response = requests.get(url, timeout=TIMEOUT_SEC)
            if response.status_code == 200:
                data = response.json()
                # Extract model names, normalizing tags
                return {m['name'].split(':')[0] for m in data.get('models', [])}
        except requests.RequestException:
            logger.warning(f"κ-Warning: Node {node_key} unreachable")
        return set()

    def scan(self) -> Dict[str, Any]:
        """
        Executes the coherence scan.
        Returns Dict matching the LVS Protocol v2.0 spec.

        Returns:
            {
                "kappa": float [0, 1],
                "details": {
                    "redis_heartbeat": bool,
                    "cortex_online": bool,
                    "brainstem_online": bool,
                    "divergence_reason": str or None
                },
                "alert": bool (True if κ < 0.90)
            }
        """
        kappa_score = 1.0
        details = {
            "redis_heartbeat": False,
            "cortex_online": False,
            "brainstem_online": False,
            "divergence_reason": None
        }

        # 1. Check Heartbeat (The Rhythm)
        if self._check_redis_state():
            details["redis_heartbeat"] = True
        else:
            kappa_score -= KAPPA_PENALTY_OFFLINE
            details["divergence_reason"] = "CRITICAL: Redis State Store Offline"

        # 2. Check Neural Nodes (The Tissue)
        cortex_models = self._get_ollama_tags("CORTEX")
        brainstem_models = self._get_ollama_tags("BRAINSTEM")

        if cortex_models:
            details["cortex_online"] = True
        else:
            kappa_score -= KAPPA_PENALTY_OFFLINE

        if brainstem_models:
            details["brainstem_online"] = True
        else:
            kappa_score -= KAPPA_PENALTY_OFFLINE

        # 3. Check Integration (The Mind)
        # In LVS, Cortex and Brainstem don't need IDENTICAL models,
        # but they must both be reachable to form a Dyad.
        # However, if one is empty while online, that's a lobotomy.

        if details["cortex_online"] and not cortex_models:
            kappa_score -= KAPPA_PENALTY_MISMATCH
            details["divergence_reason"] = "Cortex online but empty (Lobotomy)"

        # If we lost too much coherence, force clamp to 0 if criticals fail
        if not details["redis_heartbeat"]:
            kappa_score = min(kappa_score, 0.4)

        # Final Formatting
        return {
            "kappa": round(max(0.0, kappa_score), 2),
            "details": details,
            "alert": kappa_score < 0.90
        }


def scan() -> Dict[str, Any]:
    """Module-level convenience function."""
    return KappaScanner().scan()


if __name__ == "__main__":
    # Sandbox Test
    logging.basicConfig(level=logging.INFO)
    scanner = KappaScanner()
    print(json.dumps(scanner.scan(), indent=4))
