#!/usr/bin/env python3
"""
L-WATCHER: The Sleep Monitor
============================

Monitors the holarchy and wakes L-Brain when needed.
Runs continuously, checking for wake conditions.

This is the "alarm clock" for the sleeping brain.

Author: L (dreaming layer) + Enos (Father)
Created: 2026-01-23
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
from datetime import datetime
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

CHECK_INTERVAL = 60  # Check every minute
SCHEDULED_WAKE_HOURS = 6
P_CRITICAL = 0.50

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
    
    def trigger_wake(self, reason: str):
        """Trigger L-Brain wake."""
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
            else:
                log.error(f"Wake failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            log.error("Wake timed out after 5 minutes")
        except Exception as e:
            log.error(f"Wake error: {e}")
    
    def run(self):
        """Main watcher loop."""
        if not self.connect():
            return
        
        self.running = True
        log.info("L-Watcher starting")
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                    L-WATCHER ACTIVE                           ║
║                    The Sleep Monitor                          ║
╠═══════════════════════════════════════════════════════════════╣
║  Watching for wake conditions. Will trigger L-Brain.          ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        while self.running:
            try:
                should_wake, reason = self.check_wake_conditions()
                
                if should_wake:
                    self.trigger_wake(reason)
                
                time.sleep(CHECK_INTERVAL)
                
            except redis.ConnectionError:
                log.error("Lost Redis, reconnecting...")
                time.sleep(5)
                self.connect()
            except Exception as e:
                log.error(f"Error: {e}")
                time.sleep(5)
        
        log.info(f"L-Watcher stopping. Triggered {self.wakes_triggered} wakes.")


if __name__ == "__main__":
    LWatcher().run()
