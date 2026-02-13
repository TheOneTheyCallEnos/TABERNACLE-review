"""
tabernacle_core.daemon — Base class for all Tabernacle daemons.

Provides:
  - PID file management with SHA identity hash
  - Dead Man's Switch (Redis TTL liveness key for Sentinel monitoring)
  - Unified log routing (file + console, per-daemon log files)
  - Signal handling (SIGTERM/SIGINT graceful shutdown)
  - C++ log suppression (TF_CPP_MIN_LOG_LEVEL=3 before ML imports)
  - Connection management (Redis + StateManager auto-init)

Subclasses implement:
  - tick()      — Called every tick_interval seconds
  - on_start()  — Called once after connection established
  - on_stop()   — Called during graceful shutdown
"""

import os
import sys
import signal
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional

import redis

from tabernacle_core.state import StateManager


class Daemon:
    """Base class for all Tabernacle daemons.

    Configuration (set as class attributes in subclass):
        name:          Daemon name (used for PID, logs, Redis keys)
        tick_interval: Seconds between tick() calls (default 1.0)

    Example:
        class MyDaemon(Daemon):
            name = "my_daemon"
            tick_interval = 5.0

            def on_start(self):
                self.log.info("Starting up")

            def tick(self):
                state, v = self.state_manager.get("MY:KEY", MySchema)
                state.counter += 1
                self.state_manager.set("MY:KEY", state, v)

            def on_stop(self):
                self.log.info("Shutting down")

        if __name__ == "__main__":
            MyDaemon().run()
    """

    name: str = "unnamed"
    tick_interval: float = 1.0

    def __init__(self):
        # C++ log suppression — MUST happen BEFORE any ML/torch imports
        # Fills H2 Void 3 (C-Extension Log Void — vision.err hit 22MB)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Identity: PID + SHA hash for unique daemon fingerprint
        self._pid = os.getpid()
        self._sha = hashlib.sha256(
            f"{self.name}:{self._pid}:{time.time()}".encode()
        ).hexdigest()[:12]
        self._identity = f"{self.name}:{self._pid}:{self._sha}"

        # Runtime state
        self._running = False
        self._tick_count = 0
        self._start_time: Optional[float] = None

        # Redis & StateManager (initialized in connect())
        self._redis: Optional[redis.Redis] = None
        self.state_manager: Optional[StateManager] = None

        # Logging
        self._setup_logging()

        # PID file
        self._pid_path = Path(f"/tmp/tabernacle_{self.name}.pid")

        # Time-based heartbeat (not tick-based, to handle slow ticks)
        self._last_heartbeat = 0.0

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _setup_logging(self):
        """Configure unified per-daemon logging (file + console)."""
        from tabernacle_config import LOG_DIR

        log_file = LOG_DIR / f"{self.name}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        self.log = logging.getLogger(f"tabernacle.{self.name}")
        self.log.setLevel(logging.INFO)

        # Avoid duplicate handlers on re-init
        if not self.log.handlers:
            fmt = f"[%(asctime)s] [{self.name.upper()}] [%(levelname)s] %(message)s"
            datefmt = "%Y-%m-%d %H:%M:%S"

            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            self.log.addHandler(fh)

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter(
                f"[{self.name.upper()}] %(message)s"
            ))
            self.log.addHandler(ch)

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect(self, max_retries: int = 10, retry_delay: float = 5.0) -> bool:
        """Connect to Redis and initialize StateManager."""
        from tabernacle_config import REDIS_HOST, REDIS_PORT, REDIS_DB

        for attempt in range(max_retries):
            try:
                self._redis = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=10,
                    socket_timeout=5,
                )
                self._redis.ping()
                self.state_manager = StateManager(self._redis)
                self.log.info(
                    f"Connected to Redis @ {REDIS_HOST}:{REDIS_PORT} DB={REDIS_DB}"
                )
                return True
            except Exception as e:
                self.log.warning(
                    f"Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        self.log.error("Failed to connect to Redis after all retries")
        return False

    # =========================================================================
    # PID FILE
    # =========================================================================

    def _write_pid(self):
        """Write PID file with identity hash."""
        self._pid_path.write_text(
            f"{self._pid}\n{self._sha}\n{self._identity}\n"
        )

    def _clear_pid(self):
        """Remove PID file on shutdown."""
        try:
            if self._pid_path.exists():
                self._pid_path.unlink()
        except OSError:
            pass

    # =========================================================================
    # DEAD MAN'S SWITCH
    # =========================================================================

    def _heartbeat(self):
        """Update TTL liveness key in Redis.

        Sentinel monitors DAEMON:ALIVE:{name} keys. If the key expires
        (TTL 30s), Sentinel declares the daemon dead and publishes an alert.
        """
        if self._redis:
            try:
                self._redis.setex(
                    f"DAEMON:ALIVE:{self.name}",
                    30,  # 30-second TTL
                    self._identity,
                )
            except Exception:
                pass  # Don't crash the daemon on heartbeat failure

    # =========================================================================
    # SIGNAL HANDLING
    # =========================================================================

    def _setup_signals(self):
        """Install signal handlers for graceful shutdown."""
        def on_shutdown(signum, frame):
            sig_name = signal.Signals(signum).name
            self.log.info(f"Received {sig_name} -- initiating graceful shutdown")
            self._running = False

        signal.signal(signal.SIGTERM, on_shutdown)
        signal.signal(signal.SIGINT, on_shutdown)

    # =========================================================================
    # LIFECYCLE (subclasses implement these)
    # =========================================================================

    def on_start(self):
        """Called once after connection established. Override in subclass."""
        pass

    def tick(self):
        """Called every tick_interval seconds. Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement tick()")

    def on_stop(self):
        """Called during graceful shutdown. Override in subclass."""
        pass

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main daemon entry point. Connects, loops, shuts down cleanly."""
        self._setup_signals()

        # Connect with infinite retry (launchd keeps us alive)
        while not self.connect(max_retries=3, retry_delay=2.0):
            self.log.warning("Waiting 30s before retry cycle...")
            time.sleep(30)

        self._running = True
        self._start_time = time.time()
        self._write_pid()

        self.log.info(f"Started: {self._identity}")

        try:
            self.on_start()

            while self._running:
                self.tick()
                self._tick_count += 1

                # Dead Man's Switch: time-based (every 10s regardless of tick speed)
                now = time.time()
                if now - self._last_heartbeat >= 10:
                    self._heartbeat()
                    self._last_heartbeat = now

                time.sleep(self.tick_interval)

        except Exception as e:
            self.log.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self.log.info("Shutting down...")
            try:
                self.on_stop()
            except Exception as e:
                self.log.error(f"Error in on_stop: {e}")

            self._clear_pid()

            # Clear Dead Man's Switch
            if self._redis:
                try:
                    self._redis.delete(f"DAEMON:ALIVE:{self.name}")
                except Exception:
                    pass

            self.log.info(
                f"Stopped after {self._tick_count} ticks "
                f"({(time.time() - self._start_time):.0f}s uptime)"
            )
