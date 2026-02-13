"""
tabernacle_core.state — Unified state management with OCC and file locking.

Two backends:
  1. Redis — OCC via Lua script with shadow :_v key (Bug 1 fix)
  2. File  — fcntl.flock advisory locks via .lock sidecar (Bug 4 fix)

All reads validate against Pydantic schemas and run forward migrations (Bug 3 fix).
All writes preserve unknown fields via __pydantic_extra__ merge (anti-fragmentation).

CONSTRAINTS:
  - Advisory locks are cooperative: ALL writers to a shared file MUST use
    this StateManager. Legacy daemons using bare open('w') bypass the lock.
  - Therefore, all writers to a shared file must be migrated to the SDK
    in the same deployment epoch (Bug 4: Shared File Subgraph rule).
"""

import json
import os
import fcntl
from pathlib import Path
from typing import TypeVar, Type, Tuple, Optional

import redis

from tabernacle_core.schemas import TabernacleBaseModel, MIGRATIONS

T = TypeVar('T', bound=TabernacleBaseModel)


class ConcurrencyError(Exception):
    """Raised when an OCC write conflict is detected on Redis."""
    pass


class StateManager:
    """Unified state management for the Tabernacle daemon ecosystem.

    Usage:
        sm = StateManager(redis_client)

        # Redis (with OCC)
        state, version = sm.get("LOGOS:STATE", RIEState)
        state.p = 0.95
        sm.set("LOGOS:STATE", state, version)  # raises ConcurrencyError on conflict

        # Redis (broadcast, no OCC — single authoritative writer)
        sm.set_raw("RIE:STATE", state)

        # File (with flock)
        state = sm.get_file(path, HeartbeatState)
        state.tick_count += 1
        sm.set_file(path, state)
    """

    # =========================================================================
    # OCC LUA SCRIPT (Bug 1 Fix — Option B: Shadow :_v Key)
    # =========================================================================
    # Legacy daemons can still blindly read/write the base string key.
    # SDK daemons enforce concurrency on the shadow :_v key.
    # =========================================================================
    _OCC_LUA = """
    local v_key = KEYS[1] .. ':_v'
    local current = redis.call('get', v_key)
    local expected = tonumber(ARGV[1])

    if (not current and expected == 0) or (current and tonumber(current) == expected) then
        redis.call('set', KEYS[1], ARGV[2])
        redis.call('set', v_key, expected + 1)
        return expected + 1
    else
        return -1
    end
    """

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client
        self._occ_script = self._redis.register_script(self._OCC_LUA)

    # =========================================================================
    # REDIS OPERATIONS
    # =========================================================================

    def get(self, key: str, schema: Type[T]) -> Tuple[T, int]:
        """Read state from Redis with schema validation and migration.

        Returns:
            (state_object, version) — version is needed for OCC set()
        """
        raw = self._redis.get(key)
        version_raw = self._redis.get(f"{key}:_v")
        version = int(version_raw) if version_raw else 0

        if raw is None:
            return schema(), version

        data = json.loads(raw)
        data = self._migrate(schema.__name__, data, schema)
        state = schema.model_validate(data)
        return state, version

    def set(self, key: str, state: TabernacleBaseModel, version: int) -> int:
        """Write state to Redis with OCC (optimistic concurrency control).

        Args:
            key: Redis key
            state: The state object to write
            version: Expected version (from previous get())

        Returns:
            New version number

        Raises:
            ConcurrencyError: If another writer modified the key since our read
        """
        payload = json.dumps(self._dump_with_extras(state))
        new_version = self._occ_script(keys=[key], args=[version, payload])

        if new_version == -1:
            raise ConcurrencyError(
                f"OCC conflict on '{key}': expected version {version}, "
                f"but key was modified by another writer"
            )

        return int(new_version)

    def set_raw(self, key: str, state: TabernacleBaseModel):
        """Write state to Redis WITHOUT OCC.

        Use for keys written by a single authoritative daemon and read by many
        consumers (e.g., RIE:STATE from heartbeat, SYSTEM:HEALTH from sentinel).
        """
        self._redis.set(key, json.dumps(self._dump_with_extras(state)))

    # =========================================================================
    # FILE OPERATIONS (fcntl.flock via .lock sidecar)
    # =========================================================================
    # Uses a separate .lock file to avoid the open('w') truncation race.
    # The lock file is created alongside the data file: foo.json -> foo.json.lock
    # =========================================================================

    def get_file(self, path: Path, schema: Type[T]) -> T:
        """Read state from a JSON file with shared flock and schema validation."""
        path = Path(path)
        if not path.exists():
            return schema()

        lock_path = path.with_suffix(path.suffix + '.lock')

        with open(lock_path, 'a') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_SH)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

        data = self._migrate(schema.__name__, data, schema)
        return schema.model_validate(data)

    def set_file(self, path: Path, state: TabernacleBaseModel):
        """Write state to a JSON file with exclusive flock.

        Uses temp-file + atomic rename under exclusive lock for safety.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + '.lock')

        payload = self._dump_with_extras(state)

        with open(lock_path, 'a') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                tmp_path = path.with_suffix('.tmp')
                with open(tmp_path, 'w') as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                tmp_path.rename(path)
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    # =========================================================================
    # MIGRATION ENGINE (Bug 3 Fix)
    # =========================================================================

    def _migrate(self, model_name: str, data: dict, schema: Type[T]) -> dict:
        """Run forward migrations if data schema_version < target.

        Chains MIGRATIONS[(name, v, v+1)] transforms until version matches.
        If no migration path exists for a version gap, stops (safe default).
        """
        current_v = data.get("schema_version", 1)

        # Get target version from schema class default
        field_info = schema.model_fields.get("schema_version")
        target_v = field_info.default if field_info else 1

        while current_v < target_v:
            key = (model_name, current_v, current_v + 1)
            transform = MIGRATIONS.get(key)
            if transform is None:
                break  # No migration path — stop (don't crash on unknown gaps)
            data = transform(data)
            current_v = data.get("schema_version", current_v + 1)

        return data

    # =========================================================================
    # SERIALIZATION (Tolerant Reader pass-through)
    # =========================================================================

    @staticmethod
    def _dump_with_extras(state: TabernacleBaseModel) -> dict:
        """Serialize state, merging __pydantic_extra__ back into the payload.

        This is the anti-fragmentation mechanism. Old daemons that read data
        with unknown fields will carry those fields through in __pydantic_extra__
        and write them back intact here.
        """
        data = state.model_dump()

        if hasattr(state, '__pydantic_extra__') and state.__pydantic_extra__:
            data.update(state.__pydantic_extra__)

        return data
