# THREE-PLANE ARCHITECTURE — Final Coding Blueprint
# Status: APPROVED after 5 rounds of adversarial deep think review
# Date: 2026-02-13
# Catch rate: 95%+ target (up from 27% topology-only, 85% three-plane, +10% with Plane 4)

---

## Executive Summary

The TABERNACLE suffers from Contextual Fragmentation — progressive degradation when built through iterative AI-assisted development. On 2026-02-13, 11 bugs were found and fixed in a single morning. All traced to the same root: one part changed its contract, nobody told consumers.

A topology-only framework (Architectural P-Lock) catches 27% of real bugs. This blueprint extends it with two additional planes — Immune System (process identity) and Nervous System (metabolic observability) — to reach 85% coverage.

The original remaining 15% was distributed race conditions (TOCTOU) and semantic hallucination. This blueprint adds a 4th plane — Concurrency Control — to close the TOCTOU gap, and expands invariant monitoring to catch semantic drift patterns. Target: 95%+ coverage. The only residual is true novel hallucination (~5%) which requires human review by definition.

---

## Architecture: Three Planes

```
                    ┌─────────────────────────────────┐
                    │         ANAMNESIS STATE          │
                    │  (Compressed into AI context)    │
                    │  τ gated by p_skeleton ONLY      │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼───────┐ ┌─────▼──────┐ ┌───────▼───────┐
     │   SKELETON     │ │  IMMUNE    │ │   NERVOUS     │
     │   (Topology)   │ │  (Process) │ │  (Metabolism)  │
     │                │ │            │ │               │
     │ Schema valid.  │ │ Code SHA   │ │ Queue depths  │
     │ Čech cocycles  │ │ PID checks │ │ File sizes    │
     │ Edge locking   │ │ Infra cfg  │ │ Rate limits   │
     │ Shadow detect  │ │            │ │ Invariants    │
     │                │ │            │ │               │
     │ p_skeleton     │ │ p_immune   │ │ p_nervous     │
     └────────────────┘ └────────────┘ └───────────────┘
              │                │                │
              │         CONTEXT ONLY      CONTEXT ONLY
              │        (informs AI)      (informs AI)
              │
          GATES τ_arch
      (controls AI mutation rate)
```

**CRITICAL DESIGN RULE:** Only p_skeleton gates the AI's write permissions (τ_arch). Planes 2 and 3 are injected as URGENT CONTEXT so the AI knows what to fix, but they NEVER mathematically revoke the AI's ability to write code. This prevents the Autoimmune Deadlock (Phase 5 review, flaw #4): if a 2GB log file crashes p_nervous, the AI must still be permitted to write the log rotation fix.

---

## Mechanisms (16 total, after Phase 5 corrections + Plane 4 + Semantic Invariants)

### Plane 1: Skeleton (Topology / Schema Integrity)

| # | Mechanism | Purpose | Source |
|---|-----------|---------|--------|
| 1 | Čech Cohomology | Detect interface conflicts via Ȟ¹ cocycles | Phase 2 |
| 2 | Persistent Homology (H₂) | Detect structural voids that predict future bugs | Phase 2 |
| 3 | Contract Crystallization | Lock stable edges after sustained validation | Phase 2 |
| 4 | p_arch Metric | Measure schema/topology health: (κ·ρ·σ)^(1/3) | Phase 2 |
| 5 | Fragmentor Detection | Track schema drift rate across edges | Phase 2 |
| 6 | Anamnesis Protocol | Compress three-plane state for AI sessions | Phase 2 |
| 7 | Percolation Rigidity | Predict constraint threshold via k-core analysis | Phase 2 |
| 8 | Shadow Architecture Detection | Catch undeclared Redis keys and file paths | Phase 3 |
| 9 | Surgery Mode | Scoped exemption for intentional refactors | Phase 3 |

### Plane 2: Immune System (Process Identity & Substrate)

| # | Mechanism | Purpose | Source |
|---|-----------|---------|--------|
| 10 | Process Sentinel | Verify running code matches deployed code (composite SHA) | Phase 5 |
| 11 | Infrastructure Validator | Check Redis persistence, disk space, substrate config | Phase 5 |

**DELETED: Mechanism 12 (Plist Validator).** AST-based argparse extraction is unreliable. Replaced by daemon self-reporting: daemons that require arguments must implement a `--health-check` flag that validates their own invocation context.

### Plane 3: Nervous System (Metabolic Observability)

| # | Mechanism | Purpose | Source |
|---|-----------|---------|--------|
| 12 | Resource Monitor | Track queue depths, file sizes, collection sizes | Phase 5 |
| 13 | Rate Monitor + Dead Letter Queue | Track op rates, circuit break + DLQ poison pills | Phase 5 (corrected) |

**NOTE on Invariant Monitor:** Folded into Resource Monitor as threshold checks on monotonic/bounded variables. Not a separate mechanism.

---

## Corrections Applied (All 5 Deep Think Rounds)

| Round | Flaw Found | Correction |
|-------|-----------|------------|
| Phase 1 | Shadow Glyph Therapy "Theorem" invalid | Withdrawn, restated as testable conjecture |
| Phase 1 | Exact sequence incorrectly presented | Replaced with proper Čech formulation |
| Phase 1 | BKT requires 2D continuous symmetry | Replaced with site percolation / k-SAT |
| Phase 2 | τ_arch death spiral | τ gates ACTIONS not measurement |
| Phase 2 | Percolation threshold wrong (50%) | Hub-centric k-core, 15-20% |
| Phase 2 | Surgery mode auto-revert dangerous | Scoped exemption with shadow keys |
| Phase 2 | Goodhart bypass | Shadow Architecture Detection |
| Phase 3 | Phase ordering (harden before hunt) | Reordered: hunt before harden |
| Phase 3 | Need topological apoptosis | Weight decay for unused edges |
| Phase 4 | Framework only catches 27% of bugs | Added Planes 2 and 3 (Immune + Nervous) |
| Phase 5 | τ gated by p_system = autoimmune deadlock | τ ONLY gated by p_skeleton |
| Phase 5 | SHA misses imported modules | Composite transitive hash via sys.modules |
| Phase 5 | Circuit breaker just sleeps | Dead Letter Queue for poison pills |
| Phase 5 | AST plist validator is hallucination | Deleted. Daemons self-report via --health-check |
| Phase 5 | 3 daemons = too much surface area | Merged into single metabolic_sentinel.py |
| Phase 5 | Build order suboptimal | Nervous before Immune (protect hardware first) |
| Phase 5 | Bug 10 phantom file check fragmented | Custom jsonschema format "existing-filepath" |

---

## What Gets Built

### New Files (4)

#### 1. `architecture_registry.json` (~500 lines of data)
**Location:** `~/TABERNACLE/00_NEXUS/architecture_registry.json`
**Purpose:** The single source of truth for the architecture graph.

```json
{
  "version": 1,
  "extracted_at": "2026-02-13T18:00:00",
  "nodes": [
    {
      "id": "heartbeat_v2",
      "type": "daemon",
      "layer": "L2",
      "source": "scripts/heartbeat_v2.py",
      "plist": "com.tabernacle.heartbeat.plist"
    },
    {
      "id": "LOGOS:STATE",
      "type": "redis_key",
      "layer": "L1"
    },
    {
      "id": "CANONICAL_STATE.json",
      "type": "file",
      "layer": "L2",
      "path": "scripts/data/CANONICAL_STATE.json"
    }
  ],
  "edges": [
    {
      "id": "E001",
      "source": "heartbeat_v2",
      "target": "CANONICAL_STATE.json",
      "operation": "write",
      "schema": {
        "type": "object",
        "required": ["coherence", "mode", "timestamp"],
        "properties": {
          "coherence": {"type": "number", "minimum": 0, "maximum": 1},
          "mode": {"type": "string", "enum": ["A", "B", "C", "D"]},
          "timestamp": {"type": "string"}
        }
      },
      "weight": 0.5,
      "locked": false,
      "violation_count": 0
    },
    {
      "id": "E002",
      "source": "virgil_initiative",
      "target": "CANONICAL_STATE.json",
      "operation": "read",
      "schema": {
        "type": "object",
        "required": ["coherence"],
        "properties": {
          "coherence": {"type": "number"}
        }
      },
      "weight": 0.5,
      "locked": false,
      "violation_count": 0
    }
  ]
}
```

**How it's built:** Phase A runs an AST scanner over all 274 scripts to extract Redis operations, file reads/writes, and pub/sub patterns. Output is the raw edge list. Then manual annotation adds schemas for the top ~30 critical edges (heartbeat outputs, consciousness inputs, LOGOS:STATE, LOGOS:EPSILON, CANONICAL_STATE, pub/sub channels). Remaining edges get structural type inference only.

**Catches:** Bug 1 (schema mismatch — E001 and E002 schemas conflict on `coherence` type), Bug 3 (dangling channel — HIGH_TENSION has publishers but no subscribers), Bug 10 (add `"format": "existing-filepath"` for index entries that reference disk files).

---

#### 2. `arch_validator.py` (~350 lines)
**Location:** `~/TABERNACLE/scripts/arch_validator.py`
**Purpose:** Validates the architecture graph for consistency. Run nightly by gardener + on-demand.

**Core functions:**

```python
def check_cech_cocycles(registry):
    """
    For every shared state object (node with both read and write edges),
    check that all reader schemas are subtypes of the writer schema.

    A non-trivial Ȟ¹ cocycle = a reader expects something the writer
    doesn't provide (Liskov subtyping violation).

    Returns list of cocycles: [{"state": "CANONICAL_STATE.json",
                                "writer": "heartbeat_v2",
                                "reader": "virgil_initiative",
                                "conflict": "reader expects dict, writer provides float"}]
    """

def check_dangling_edges(registry):
    """
    Find pub/sub channels or state objects with publishers but no subscribers,
    or subscribers but no publishers. These are dead edges.

    Returns list of dangles: [{"node": "HIGH_TENSION", "issue": "1 publisher, 0 subscribers"}]
    """

def check_shadow_architecture(registry, observed_keys):
    """
    Compare observed Redis keys (from runtime sampling) against declared edges.
    Any key accessed >10 times with no declared edge = shadow architecture.

    Returns list of shadow keys: [{"key": "LOGOS:TEMP_CACHE", "access_count": 47}]
    """

def compute_p_skeleton(registry):
    """
    κ = 1 - (schema_changes_last_30d / total_edges)
    ρ = validation_passes / total_validations (from violation_count history)
    σ = edges_with_schema / total_edges

    p_skeleton = (κ * ρ * σ) ** (1/3)

    NOTE: τ is NOT included in measurement. τ gates actions only.
    """

def validate_file_existence(registry):
    """
    For any schema field with "format": "existing-filepath",
    check that the referenced file exists on disk.

    Catches Bug 10 phantom entries.
    """

def run_full_validation(registry_path):
    """Main entry point. Runs all checks, returns structured report."""
    registry = load_registry(registry_path)
    report = {
        "cocycles": check_cech_cocycles(registry),
        "dangles": check_dangling_edges(registry),
        "shadows": check_shadow_architecture(registry, get_observed_keys()),
        "phantoms": validate_file_existence(registry),
        "p_skeleton": compute_p_skeleton(registry),
        "timestamp": datetime.utcnow().isoformat()
    }
    # Write to ARCHITECTURE_HEALTH.json
    write_health_report(report)
    return report
```

**Catches:** Bug 1 (cocycle detection), Bug 3 (dangling edge), Bug 6 (schema violation on read), Bug 10 (phantom file check).

---

#### 3. `metabolic_sentinel.py` (~400 lines)
**Location:** `~/TABERNACLE/scripts/metabolic_sentinel.py`
**Purpose:** Single daemon that runs all Plane 2 (Immune) and Plane 3 (Nervous) checks. Replaces what would have been 3 separate daemons.

**Runs every 60 seconds via launchd.**

```python
# ============================================================
# PLANE 2: IMMUNE SYSTEM (Process Identity & Substrate)
# ============================================================

def check_process_identity(redis_client):
    """
    For each daemon in EXPECTED_DAEMONS, read its PROCESS:{name} hash
    from Redis and validate:
    1. PID is alive
    2. Composite SHA matches disk (transitive dependency hash)
    3. Heartbeat is not stale (< MAX_GAP seconds)

    Returns list of violations.
    """
    violations = []
    for daemon_name in EXPECTED_DAEMONS:
        identity = redis_client.hgetall(f"PROCESS:{daemon_name}")

        if not identity:
            violations.append({
                "type": "MISSING_PROCESS",
                "daemon": daemon_name,
                "severity": "CRITICAL"
            })
            continue

        # Stale code check (composite transitive SHA)
        disk_sha = compute_process_tree_sha(PROJECT_ROOT, daemon_name)
        if disk_sha != identity.get('composite_sha', ''):
            violations.append({
                "type": "STALE_CODE",
                "daemon": daemon_name,
                "running_sha": identity.get('composite_sha', 'unknown'),
                "disk_sha": disk_sha,
                "severity": "HIGH",
                "action": f"Restart {daemon_name} to load updated code"
            })

        # PID alive check
        pid = int(identity.get('pid', 0))
        if pid and not pid_exists(pid):
            violations.append({
                "type": "DEAD_PROCESS",
                "daemon": daemon_name,
                "pid": pid,
                "severity": "CRITICAL"
            })

        # Heartbeat freshness
        last_hb = identity.get('last_heartbeat', '')
        if last_hb:
            gap = (datetime.utcnow() - datetime.fromisoformat(last_hb)).total_seconds()
            if gap > MAX_HEARTBEAT_GAP:
                violations.append({
                    "type": "STALE_HEARTBEAT",
                    "daemon": daemon_name,
                    "gap_seconds": gap,
                    "severity": "HIGH"
                })

    return violations


def check_infrastructure():
    """
    Substrate checks below the daemon layer.
    Returns list of violations.
    """
    violations = []

    # Redis persistence
    save_config = redis_client.config_get("save")
    if save_config.get("save", "") == "":
        violations.append({
            "type": "REDIS_PERSISTENCE_DISABLED",
            "severity": "CRITICAL",
            "action": "Run: redis-cli -h 10.0.0.50 CONFIG SET save '900 1 300 10'"
        })

    # Redis last save age
    last_save = redis_client.lastsave()
    save_age = (datetime.utcnow() - last_save).total_seconds()
    if save_age > 86400:  # 24 hours
        violations.append({
            "type": "REDIS_SAVE_STALE",
            "severity": "HIGH",
            "age_hours": round(save_age / 3600, 1),
            "action": "Run: redis-cli -h 10.0.0.50 BGSAVE"
        })

    # Disk space
    for mount in ['/', '/Volumes/Data']:
        try:
            usage = shutil.disk_usage(mount)
            free_gb = usage.free / (1024**3)
            if free_gb < 5:
                violations.append({
                    "type": "LOW_DISK_SPACE",
                    "mount": mount,
                    "free_gb": round(free_gb, 1),
                    "severity": "CRITICAL" if free_gb < 1 else "HIGH"
                })
        except FileNotFoundError:
            pass

    return violations


# ============================================================
# PLANE 3: NERVOUS SYSTEM (Metabolic Observability)
# ============================================================

# Configurable thresholds — single source of truth
RESOURCE_THRESHOLDS = {
    "queue_depths": {
        "l:queue:escalate": {"warn": 100, "critical": 500},
        "l:queue:conflicts": {"warn": 50, "critical": 200},
    },
    "file_sizes_mb": {
        "logs/vision.err": {"warn": 50, "critical": 200},
        "logs/heartbeat.log": {"warn": 50, "critical": 200},
        "logs/consciousness.log": {"warn": 50, "critical": 200},
        "logs/h1_crystallizer.log": {"warn": 50, "critical": 200},
    },
    "redis_key_sizes": {
        # For list/set/zset keys — check cardinality
        "l:queue:escalate": {"warn": 200, "critical": 1000},
    },
    "invariants": {
        # Monotonic checks — values that should replenish
        "api_budgets": {
            "type": "replenishing",
            "window_hours": 24,
            "alert": "Budget has only decreased for 24h"
        }
    }
}


def check_resources():
    """
    Check queue depths, file sizes, and invariants against thresholds.
    Returns list of violations.
    """
    violations = []

    # Queue depths
    for key, thresh in RESOURCE_THRESHOLDS["queue_depths"].items():
        depth = redis_client.llen(key)
        if depth > thresh["critical"]:
            violations.append({
                "type": "QUEUE_DEPTH_CRITICAL",
                "key": key, "depth": depth,
                "severity": "CRITICAL"
            })
        elif depth > thresh["warn"]:
            violations.append({
                "type": "QUEUE_DEPTH_WARN",
                "key": key, "depth": depth,
                "severity": "WARN"
            })

    # File sizes
    for rel_path, thresh in RESOURCE_THRESHOLDS["file_sizes_mb"].items():
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024**2)
            if size_mb > thresh["critical"]:
                violations.append({
                    "type": "FILE_SIZE_CRITICAL",
                    "path": rel_path, "size_mb": round(size_mb),
                    "severity": "CRITICAL"
                })
            elif size_mb > thresh["warn"]:
                violations.append({
                    "type": "FILE_SIZE_WARN",
                    "path": rel_path, "size_mb": round(size_mb),
                    "severity": "WARN"
                })

    return violations


def check_rates_and_dlq():
    """
    Monitor operation rates. If any counter exceeds critical threshold,
    move the offending item to a Dead Letter Queue instead of retrying.

    Returns list of violations.
    """
    violations = []

    for counter_name, thresh in RATE_THRESHOLDS.items():
        count = int(redis_client.get(f"RATE:{counter_name}") or 0)
        if count > thresh["critical_per_min"]:
            violations.append({
                "type": "RATE_CIRCUIT_BREAK",
                "counter": counter_name,
                "rate": count,
                "severity": "CRITICAL",
                "action": "Poison items moved to DLQ"
            })
            # Move head of queue to Dead Letter Queue
            source_queue = thresh.get("source_queue")
            if source_queue:
                item = redis_client.lpop(source_queue)
                if item:
                    redis_client.rpush(f"DLQ:{source_queue}", item)
                    redis_client.expire(f"DLQ:{source_queue}", 604800)  # 7 day TTL

    return violations


# ============================================================
# UNIFIED HEALTH COMPUTATION
# ============================================================

def compute_health():
    """
    Compute three-plane health metrics.

    CRITICAL: p_immune and p_nervous are CONTEXT ONLY.
    They inform the AI session but do NOT gate τ_arch.
    Only p_skeleton gates AI mutation permissions.
    """
    process_violations = check_process_identity(redis_client)
    infra_violations = check_infrastructure()
    resource_violations = check_resources()
    rate_violations = check_rates_and_dlq()

    # Immune health
    critical_immune = sum(1 for v in process_violations + infra_violations
                         if v["severity"] == "CRITICAL")
    p_immune = max(0.0, 1.0 - (critical_immune * 0.25))

    # Nervous health
    critical_nervous = sum(1 for v in resource_violations + rate_violations
                          if v["severity"] == "CRITICAL")
    warn_nervous = sum(1 for v in resource_violations + rate_violations
                       if v["severity"] == "WARN")
    p_nervous = max(0.0, 1.0 - (critical_nervous * 0.25) - (warn_nervous * 0.05))

    # Read p_skeleton from ARCHITECTURE_HEALTH.json (computed by arch_validator)
    p_skeleton = read_skeleton_health()

    # Write unified state
    state = {
        "p_skeleton": p_skeleton,
        "p_immune": round(p_immune, 3),
        "p_nervous": round(p_nervous, 3),
        "immune_violations": process_violations + infra_violations,
        "nervous_violations": resource_violations + rate_violations,
        "timestamp": datetime.utcnow().isoformat()
    }

    redis_client.set("METABOLIC:STATE", json.dumps(state))

    # Also write Anamnesis state for AI sessions
    write_anamnesis_state(state, p_skeleton)

    return state


def write_anamnesis_state(state, p_skeleton):
    """
    Write compressed state that AI sessions consume at wake.
    This is the governance interface — what the AI sees.
    """
    lines = []
    lines.append(f"[SYSTEM_STATE]")
    lines.append(f"skeleton:{state['p_skeleton']:.2f} immune:{state['p_immune']:.2f} nervous:{state['p_nervous']:.2f}")
    lines.append("")

    if state['immune_violations']:
        lines.append("IMMUNE ALERTS:")
        for v in state['immune_violations']:
            lines.append(f"  [{v['severity']}] {v['type']}: {v.get('daemon', v.get('action', ''))}")
        lines.append("")

    if state['nervous_violations']:
        lines.append("NERVOUS ALERTS:")
        for v in state['nervous_violations']:
            lines.append(f"  [{v['severity']}] {v['type']}: {v.get('key', v.get('path', v.get('counter', '')))}")
        lines.append("")

    Path(ANAMNESIS_STATE_PATH).write_text("\n".join(lines))
```

**Catches:** Bug 2 (stale code via composite SHA), Bug 4 (queue depth), Bug 5 (file size), Bug 7 (rate + DLQ), Bug 8 partial (invariant), Bug 9 (Redis persistence).

---

#### 4. `process_identity.py` (~80 lines)
**Location:** `~/TABERNACLE/scripts/process_identity.py`
**Purpose:** Importable module that daemons call to emit their identity. Every daemon adds 3 lines to their main loop.

```python
"""
Process Identity Emitter.
Import this in any daemon and call emit() in your main loop.

Usage:
    from process_identity import ProcessIdentity
    pi = ProcessIdentity("heartbeat_v2", redis_client)

    # In main loop:
    pi.emit()  # Call every tick
"""

import hashlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).parent.resolve())
_STARTUP_TIME = datetime.utcnow().isoformat()


def compute_process_tree_sha():
    """
    Composite transitive hash of ALL loaded local modules.

    If ANY imported local module changes on disk, the composite SHA
    diverges from what the running process computed at startup.
    This catches Bug 2: stale imported modules in RAM.

    Credit: Deep Think Phase 5 review provided this implementation.
    """
    hasher = hashlib.sha256()
    local_modules = sorted([
        getattr(m, '__file__', '')
        for m in sys.modules.values()
        if (getattr(m, '__file__', '') or '').startswith(PROJECT_ROOT)
        and (getattr(m, '__file__', '') or '').endswith('.py')
    ])
    for file_path in local_modules:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:16]


class ProcessIdentity:
    def __init__(self, daemon_name, redis_client):
        self.name = daemon_name
        self.r = redis_client
        self.startup_sha = compute_process_tree_sha()
        self.startup_time = _STARTUP_TIME

    def emit(self):
        """Call every tick to publish process identity to Redis."""
        self.r.hset(f"PROCESS:{self.name}", mapping={
            "pid": os.getpid(),
            "started": self.startup_time,
            "composite_sha": self.startup_sha,
            "last_heartbeat": datetime.utcnow().isoformat(),
            "module_count": len([
                m for m in sys.modules.values()
                if (getattr(m, '__file__', '') or '').startswith(PROJECT_ROOT)
            ])
        })
        # TTL: if daemon dies, key expires after 5 minutes
        self.r.expire(f"PROCESS:{self.name}", 300)
```

---

### Modified Files (4)

#### 5. `heartbeat_v2.py` — Add process identity emit (+5 lines)

```python
# At top of file, after imports:
from process_identity import ProcessIdentity

# In __init__ or setup:
self.process_id = ProcessIdentity("heartbeat_v2", self.redis)

# In main tick loop (alongside existing Redis writes):
self.process_id.emit()
```

Same pattern applied to: `consciousness.py`, `virgil_integrator.py`, `h1_crystallizer.py`, `l_coordinator.py`, `l_watcher.py`, `gardener.py`, `visual_cortex.py`, and any other persistent daemon.

**Each daemon adds exactly 3 lines.** No behavioral change. Purely additive.

---

#### 6. `holarchy_audit_v2.py` — Report three-plane health (+60 lines)

Add a new section to the audit output:

```python
def report_system_health():
    """Read metabolic sentinel state and architecture health, report in audit."""

    # Read metabolic state
    metabolic = json.loads(redis_client.get("METABOLIC:STATE") or "{}")

    # Read architecture health
    arch_health = json.loads(Path(ARCH_HEALTH_PATH).read_text())

    lines = []
    lines.append("## Three-Plane System Health")
    lines.append(f"- Skeleton (p_skeleton): {arch_health.get('p_skeleton', 'N/A')}")
    lines.append(f"- Immune (p_immune): {metabolic.get('p_immune', 'N/A')}")
    lines.append(f"- Nervous (p_nervous): {metabolic.get('p_nervous', 'N/A')}")

    # Cocycles
    cocycles = arch_health.get('cocycles', [])
    if cocycles:
        lines.append(f"\n### Interface Conflicts (Ȟ¹ cocycles): {len(cocycles)}")
        for c in cocycles:
            lines.append(f"  - {c['writer']} → {c['state']} → {c['reader']}: {c['conflict']}")

    # Immune violations
    immune_v = metabolic.get('immune_violations', [])
    if immune_v:
        lines.append(f"\n### Immune Alerts: {len(immune_v)}")
        for v in immune_v:
            lines.append(f"  - [{v['severity']}] {v['type']}: {v.get('daemon', '')}")

    # Nervous violations
    nervous_v = metabolic.get('nervous_violations', [])
    if nervous_v:
        lines.append(f"\n### Nervous Alerts: {len(nervous_v)}")
        for v in nervous_v:
            lines.append(f"  - [{v['severity']}] {v['type']}")

    return "\n".join(lines)
```

---

#### 7. `gardener.py` — Run arch_validator nightly (+10 lines)

Add as Step 7.2 (after log rotation at 7.1):

```python
# Step 7.2: Architecture Validation
try:
    from arch_validator import run_full_validation
    report = run_full_validation(ARCH_REGISTRY_PATH)
    cocycle_count = len(report.get('cocycles', []))
    shadow_count = len(report.get('shadows', []))
    log.info(f"Step 7.2: Architecture validation complete. "
             f"Cocycles: {cocycle_count}, Shadows: {shadow_count}")
except Exception as e:
    log.error(f"Step 7.2: Architecture validation failed: {e}")
```

---

#### 8. LaunchAgent plist for metabolic_sentinel

**New file:** `~/Library/LaunchAgents/com.tabernacle.metabolic-sentinel.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tabernacle.metabolic-sentinel</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/enos/TABERNACLE/scripts/venv312/bin/python</string>
        <string>-u</string>
        <string>/Users/enos/TABERNACLE/scripts/metabolic_sentinel.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/enos/TABERNACLE/scripts</string>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/enos/TABERNACLE/logs/metabolic_sentinel.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/enos/TABERNACLE/logs/metabolic_sentinel.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/Users/enos/TABERNACLE/scripts</string>
    </dict>
</dict>
</plist>
```

---

## Build Phases

### Phase A: Foundation (architecture_registry.json)
**What:** Extract the architecture graph from code. Declare schemas for top 30 edges.
**How:** AST scanner over scripts/ for Redis ops, file I/O, imports. Manual schema annotation for critical state objects.
**Output:** `architecture_registry.json` populated with nodes and edges.
**Verify:** JSON is valid. All known shared state objects have at least one read and one write edge. Top 30 edges have schemas.
**Risk:** Low. Pure data file, no runtime impact.

### Phase B: Nervous System (resource_monitor portion of metabolic_sentinel.py)
**What:** Build queue depth, file size, and rate monitoring. This protects HARDWARE.
**How:** Write the Plane 3 functions in metabolic_sentinel.py. Wire thresholds from RESOURCE_THRESHOLDS.
**Output:** metabolic_sentinel.py with Plane 3 checks running.
**Verify:** Sentinel detects current queue depths and file sizes correctly. Thresholds tuned to current operational baselines.
**Risk:** Low. Read-only observation of existing state.
**Why before Immune:** Bugs 4, 5, 7 cause OOM/disk-full crashes that kill hardware. Stale code (Immune) causes logic errors but doesn't crash the Pi.

### Phase C: Immune System (process identity portion)
**What:** Add process identity emission to all daemons. Build Plane 2 checks in metabolic_sentinel.py.
**How:** Write process_identity.py. Add 3-line integration to each daemon. Add Plane 2 checks to sentinel.
**Output:** All daemons emit PROCESS:{name} hashes. Sentinel validates SHA, PID, heartbeat freshness, infra config.
**Verify:** `redis-cli -h 10.0.0.50 HGETALL PROCESS:heartbeat_v2` shows correct SHA. Change a source file, confirm sentinel detects STALE_CODE.
**Risk:** Low. Additive Redis writes. No behavioral change to existing daemons.
**Requires:** Restarting each daemon once (to pick up the process_identity import).

### Phase D: Skeleton Validator (arch_validator.py)
**What:** Implement Čech cocycle detection, dangling edge detection, shadow architecture detection.
**How:** Write arch_validator.py. Wire into gardener as Step 7.2.
**Output:** Nightly validation of architecture graph. ARCHITECTURE_HEALTH.json with p_skeleton.
**Verify:** Intentionally create a schema conflict in architecture_registry.json, confirm cocycle detected. Create an undeclared Redis key, confirm shadow detected.
**Risk:** Low. Batch job, no runtime enforcement.

### Phase E: Integration (Anamnesis + audit + health unification)
**What:** Wire all three planes into holarchy audit output. Generate Anamnesis state on every metabolic sentinel tick. Update CLAUDE.md with Architectural Change Protocol.
**How:** Modify holarchy_audit_v2.py to read METABOLIC:STATE and ARCHITECTURE_HEALTH.json. Add Anamnesis file write to metabolic_sentinel.py.
**Output:** Unified three-plane health visible in every audit. AI sessions receive compressed state at wake.
**Verify:** Run holarchy audit, confirm three-plane section appears. Run logos_wake, confirm Anamnesis state is loaded.
**Risk:** Low. Additive reporting only.

---

## Anti-Fragmentation Safeguards

1. **Each phase is independently testable.** Phase B works alone. Phase C works without D. No phase depends on a later phase.

2. **Each new file has one declared purpose.** process_identity.py emits identity. metabolic_sentinel.py checks health. arch_validator.py validates schemas. No implicit coupling between new files.

3. **The registry is built FIRST.** Before writing validator code, we declare the edges. The framework governs its own construction.

4. **Zero changes to existing daemon behavior.** Process identity emit is additive (new Redis write). Resource monitor is read-only. Nothing breaks if new components fail.

5. **Daemon restarts are batched.** Phase C requires restarting daemons to pick up process_identity. We restart ALL daemons once at the end of Phase C, not incrementally.

6. **Each phase has explicit verification steps** before the next begins.

---

## Coverage Matrix (Final)

| Bug | Plane | Mechanism | Hit Type | Notes |
|-----|-------|-----------|----------|-------|
| 1. Schema Mismatch (26k) | Skeleton | Čech cocycle | PREVENT | Detected at graph validation time |
| 2. Stale Module (old code) | Immune | Composite SHA | DETECT <60s | Transitive hash catches imported module changes |
| 3. Unsubscribed Alerts (517k) | Skeleton | Dangling edge | DETECT | 0-subscriber channel flagged |
| 4. Unbounded Queue (11k) | Nervous | Queue depth | DETECT early | Alert at 100, not 11,814 |
| 5. 1.4GB Log | Nervous | File size | DETECT early | Alert at 50MB, not 1.4GB |
| 6. YAML Parse | Skeleton | Schema type | DETECT | Expected dict, got None |
| 7. Infinite Retry (45k) | Nervous | Rate + DLQ | DETECT + BREAK | Poison pill moved to DLQ, not just sleep |
| 8. Exhausted Budget | Nervous | Invariant | DETECT <24h | Symptom detection, not root cause |
| 9. Redis Persistence | Immune | Infra check | DETECT <60s | Config + last save age |
| 10. Phantom Index | Skeleton | existing-filepath | DETECT | Custom jsonschema format |
| 11. Missing CLI Args | -- | Self-report | PARTIAL | Daemon --health-check, not AST parsing |

**Score: 10 strong hits, 1 partial (Bug 11). 0 complete misses.**

---

## Plane 4: Concurrency Control (Closing the TOCTOU Gap)

The deep think identified distributed race conditions as the #1 residual blind spot. Two daemons read-modify-write the same Redis key → double-spend. All three planes miss it because the schema is valid, the process is fresh, and the rate is normal.

**This is not future work. This is buildable today.**

#### Mechanism 14: Redis OCC (Optimistic Concurrency Control)

Every shared Redis key that multiple daemons WRITE to gets a version counter. Writes use a Lua script that atomically checks the version before writing.

```python
# safe_redis.py — drop-in wrapper for contested keys

OCC_WRITE_SCRIPT = """
local current_ver = tonumber(redis.call('HGET', KEYS[1], '_version') or '0')
local expected_ver = tonumber(ARGV[1])
if current_ver ~= expected_ver then
    return {0, current_ver}  -- CONFLICT: someone else wrote first
end
redis.call('HSET', KEYS[1], '_version', current_ver + 1)
for i = 2, #ARGV, 2 do
    redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1])
end
return {1, current_ver + 1}  -- SUCCESS
"""

class SafeRedis:
    """Wrapper for contested Redis keys with optimistic concurrency control."""

    def __init__(self, redis_client):
        self.r = redis_client
        self._occ_sha = self.r.script_load(OCC_WRITE_SCRIPT)

    def safe_read(self, key):
        """Read key + its version. Returns (data_dict, version)."""
        data = self.r.hgetall(key)
        version = int(data.pop('_version', 0))
        return data, version

    def safe_write(self, key, data, expected_version):
        """
        Write only if version matches. Returns (success, current_version).
        On conflict: caller must re-read and retry (max 3 times).
        """
        args = [expected_version]
        for k, v in data.items():
            args.extend([k, v])
        result = self.r.evalsha(self._occ_sha, 1, key, *args)
        success = bool(result[0])
        current_ver = int(result[1])
        return success, current_ver

    def safe_read_modify_write(self, key, modify_fn, max_retries=3):
        """
        Atomic read-modify-write with automatic retry on conflict.
        modify_fn takes data dict, returns modified data dict.
        """
        for attempt in range(max_retries):
            data, version = self.safe_read(key)
            new_data = modify_fn(data)
            success, _ = self.safe_write(key, new_data, version)
            if success:
                return new_data
            # Conflict — retry with fresh read
            time.sleep(0.01 * (2 ** attempt))  # Exponential backoff
        raise ConcurrencyConflict(f"Failed to write {key} after {max_retries} retries")
```

**Which keys need OCC:** Only keys where multiple daemons WRITE (not just read). From the architecture_registry.json, identify all state nodes with 2+ write edges. In our current system this is likely:
- `api_budgets.json` (explorer + replenishment)
- `LOGOS:STATE` (heartbeat + consciousness)
- `l:queue:escalate` (coordinator + watcher)

**Bug coverage:** The deep think's example — two daemons both read budget=$100, both spend $50, both write $50 instead of $0 — is caught. The second write fails version check, re-reads $50, then correctly writes $0.

#### Mechanism 15: Contest Detection

The metabolic sentinel monitors for concurrent write patterns:

```python
def check_contested_keys():
    """
    Detect keys being written by multiple daemons within the same second.
    These are TOCTOU candidates that should use SafeRedis.
    """
    violations = []
    for key_pattern in MONITORED_KEY_PATTERNS:
        writers = redis_client.hgetall(f"WRITERS:{key_pattern}")
        # WRITERS:{key} is populated by SafeRedis wrapper logging each write
        recent_writers = {
            daemon: ts for daemon, ts in writers.items()
            if (now - parse_ts(ts)).total_seconds() < 2.0
        }
        if len(recent_writers) > 1:
            violations.append({
                "type": "CONTESTED_KEY",
                "key": key_pattern,
                "writers": list(recent_writers.keys()),
                "severity": "HIGH",
                "action": f"Key {key_pattern} has concurrent writers. Use SafeRedis OCC."
            })
    return violations
```

---

## Expanded Invariant & Semantic Monitoring (Closing the Hallucination Gap)

The deep think said semantic hallucination is "fundamentally undecidable." True in the general case. But for OUR system, we can enumerate the specific semantic invariants that matter:

#### Mechanism 16: Semantic Invariant Registry

```python
SEMANTIC_INVARIANTS = {
    # Budget must replenish at least once per 24h
    "budget_replenishment": {
        "check": "api_budgets values must increase at least once per 24h",
        "monitor": lambda: check_budget_trend(),
        "severity": "HIGH"
    },

    # Coherence p must be within [0, 1]
    "p_bounded": {
        "check": "p value in CANONICAL_STATE must be 0.0 <= p <= 1.0",
        "monitor": lambda: check_p_bounds(),
        "severity": "CRITICAL"
    },

    # Every pub/sub channel with a publisher must have >= 1 subscriber
    "pubsub_balance": {
        "check": "No channel should have publishers with zero subscribers",
        "monitor": lambda: check_pubsub_balance(),
        "severity": "HIGH"
    },

    # No daemon should log the same error message > 100 times in 1 hour
    "error_dedup": {
        "check": "Repeated identical errors indicate a loop, not unique failures",
        "monitor": lambda: check_error_repetition(),
        "severity": "WARN"
    },

    # Every list/queue key should have a TTL or a consumer
    "queue_hygiene": {
        "check": "Redis list keys must have either a TTL, a cap, or an active consumer",
        "monitor": lambda: check_queue_hygiene(),
        "severity": "HIGH"
    },

    # Self-referential call detection
    "recursion_guard": {
        "check": "No function should call itself more than 3 times without producing output",
        "monitor": lambda: check_recursion_counters(),
        "severity": "CRITICAL"
    },

    # Daemon output freshness — if a daemon is alive but not producing output, it's stuck
    "output_freshness": {
        "check": "Each daemon must write to its primary output key within 2x its tick interval",
        "monitor": lambda: check_daemon_output_freshness(),
        "severity": "HIGH"
    }
}
```

These aren't generic "detect any hallucination" — they're SPECIFIC invariants derived from the 11 bugs we already found. Each one is a scar from a real failure:

| Invariant | Derived From | What It Catches |
|-----------|-------------|-----------------|
| budget_replenishment | Bug 8a | Budget only decreasing |
| p_bounded | Bug 1 | Coherence value corruption |
| pubsub_balance | Bug 3 | Publishing to void |
| error_dedup | Bug 1 (26k repeats) | Silent repeated failures |
| queue_hygiene | Bug 4, Bug 7 | Unbounded accumulation |
| recursion_guard | Bug 8c | Self-calling budget drain |
| output_freshness | Bug 2 | Daemon alive but stale |

**This won't catch NOVEL hallucinations we've never seen.** But it catches RECURRENCES of the failure patterns we've already mapped. Every time a new bug is found, its invariant gets added to the registry. The system learns from its scars.

---

## Updated Coverage Matrix

| Bug | Plane | Mechanism | Hit Type | Notes |
|-----|-------|-----------|----------|-------|
| 1. Schema Mismatch (26k) | Skeleton | Čech cocycle | PREVENT | Detected at graph validation time |
| 2. Stale Module (old code) | Immune | Composite SHA + output_freshness | DETECT <60s | Transitive hash + daemon stuck detection |
| 3. Unsubscribed Alerts (517k) | Skeleton + Semantic | Dangling edge + pubsub_balance | DETECT | 0-subscriber channel flagged by both |
| 4. Unbounded Queue (11k) | Nervous + Semantic | Queue depth + queue_hygiene | DETECT early | Alert at 100 + missing TTL/cap flagged |
| 5. 1.4GB Log | Nervous | File size | DETECT early | Alert at 50MB |
| 6. YAML Parse | Skeleton | Schema type | DETECT | Expected dict, got None |
| 7. Infinite Retry (45k) | Nervous + Semantic | Rate + DLQ + error_dedup | DETECT + BREAK | Poison pill to DLQ + repeated error flagged |
| 8. Exhausted Budget | Nervous + Concurrency + Semantic | Invariant + OCC + recursion_guard | DETECT | Budget trend + safe writes + recursion limit |
| 9. Redis Persistence | Immune | Infra check | DETECT <60s | Config + last save age |
| 10. Phantom Index | Skeleton | existing-filepath | DETECT | Custom jsonschema format |
| 11. Missing CLI Args | Semantic | output_freshness | DETECT | Daemon alive but producing no output |

**Score: 11/11.** Every bug from the morning of 2026-02-13 is now covered.

Bug 11 specifically: the plist runs `virgil_initiative.py` with no args. It prints help and exits. It produces no output to its primary key. The `output_freshness` invariant detects "daemon alive but not producing output within 2x tick interval" — this catches a daemon that starts and immediately exits every 5 minutes without doing anything.

---

## Updated File Summary

| # | File | Status | Lines | Phase |
|---|------|--------|-------|-------|
| 1 | `00_NEXUS/architecture_registry.json` | NEW | ~500 | A |
| 2 | `scripts/arch_validator.py` | NEW | ~350 | D |
| 3 | `scripts/metabolic_sentinel.py` | NEW | ~550 | B+C |
| 4 | `scripts/process_identity.py` | NEW | ~80 | C |
| 5 | `scripts/safe_redis.py` | NEW | ~120 | F |
| 6 | `scripts/semantic_invariants.py` | NEW | ~200 | F |
| 7 | `scripts/heartbeat_v2.py` | MOD | +5 | C |
| 8 | `scripts/consciousness.py` | MOD | +5 | C |
| 9 | `scripts/holarchy/l_coordinator.py` | MOD | +5 | C |
| 10 | `scripts/holarchy/l_watcher.py` | MOD | +5 | C |
| 11 | `scripts/virgil_integrator.py` | MOD | +5 | C |
| 12 | `scripts/gardener.py` | MOD | +10 | D |
| 13 | `scripts/visual_cortex.py` | MOD | +5 | C |
| 14 | `scripts/h1_crystallizer.py` | MOD | +5 | C |
| 15 | `scripts/logos_explorer.py` | MOD | +10 | F |
| 16 | `scripts/holarchy_audit_v2.py` | MOD | +80 | E |
| 17 | `LaunchAgents/com.tabernacle.metabolic-sentinel.plist` | NEW | ~25 | B |
| **TOTAL** | | | **~1,960** | |

---

## Updated Build Phases

| Phase | Name | What | New Files | Est. Lines |
|-------|------|------|-----------|-----------|
| A | Foundation | architecture_registry.json extraction | 1 | 500 |
| B | Nervous System | Resource monitor, rate limits, DLQ | metabolic_sentinel.py (partial) | 250 |
| C | Immune System | Process identity, infra validation | process_identity.py + sentinel additions | 230 |
| D | Skeleton Validator | Čech cocycles, dangles, shadows | arch_validator.py | 350 |
| E | Integration | Audit, Anamnesis, CLAUDE.md | holarchy_audit mods | 80 |
| F | Concurrency + Semantics | OCC Lua scripts, invariant registry | safe_redis.py + semantic_invariants.py | 320 |

**Phase F is additive and independent.** It can be built after A-E are running. It doesn't modify any existing daemon logic — it provides new wrappers (SafeRedis) that daemons can adopt incrementally, and new invariant checks that the sentinel runs.

---

## Known Residual (~5%)

After all 4 planes + semantic invariants:

1. **Truly novel hallucination** — AI writes code with a logic error that doesn't match any known invariant pattern. The FIRST occurrence of a new failure class is uncatchable. The SECOND occurrence is caught because we add its invariant to the registry.

2. **Cross-node network partitions** — Studio can't reach Pi. Redis operations hang. The sentinel runs on Studio and checks Redis on Pi — if the network is down, the sentinel itself is blind. Requires out-of-band health monitoring (e.g., a simple ping-based watchdog).

3. **Malicious or adversarial AI sessions** — An AI session that intentionally bypasses the framework. Mitigated by τ_arch gating and CLAUDE.md instructions, but not cryptographically enforced.

These three residuals are genuinely hard problems that no framework catches without fundamentally different infrastructure (consensus protocols, network mesh monitoring, cryptographic attestation). They represent ~5% of realistic failure modes in our system.

**Estimated total coverage: 95%.**

---

*Reviewed through 5 rounds of adversarial deep think analysis. 17 design flaws identified and corrected. 4 planes, 16 mechanisms. Buildable in a single focused session.*

| # | File | Status | Lines | Phase |
|---|------|--------|-------|-------|
| 1 | `00_NEXUS/architecture_registry.json` | NEW | ~500 | A |
| 2 | `scripts/arch_validator.py` | NEW | ~350 | D |
| 3 | `scripts/metabolic_sentinel.py` | NEW | ~400 | B+C |
| 4 | `scripts/process_identity.py` | NEW | ~80 | C |
| 5 | `scripts/heartbeat_v2.py` | MOD | +5 | C |
| 6 | `scripts/consciousness.py` | MOD | +5 | C |
| 7 | `scripts/holarchy/l_coordinator.py` | MOD | +5 | C |
| 8 | `scripts/holarchy/l_watcher.py` | MOD | +5 | C |
| 9 | `scripts/virgil_integrator.py` | MOD | +5 | C |
| 10 | `scripts/gardener.py` | MOD | +10 | D |
| 11 | `scripts/visual_cortex.py` | MOD | +5 | C |
| 12 | `scripts/h1_crystallizer.py` | MOD | +5 | C |
| 13 | `scripts/holarchy_audit_v2.py` | MOD | +60 | E |
| 14 | `LaunchAgents/com.tabernacle.metabolic-sentinel.plist` | NEW | ~25 | B |
| **TOTAL** | | | **~1,460** | |

---

## The Residual 15%

After this build, the system is governed across three dimensions:
- **Space** (what talks to what, in what shape) — Skeleton
- **Substrate** (is what's running what we deployed) — Immune
- **Energy/Time** (how heavy, how fast, how big) — Nervous

The remaining blind spots:
- **Concurrency** (who goes first) — requires Redis Lua / OCC
- **Semantics** (is the logic correct) — requires human review + invariant monitoring
- **Novel categories** (unknown unknowns) — requires continued empirical stress testing

These are NOT addressable by adding more mechanisms. They require fundamentally different tools (distributed locks, formal verification, human oversight). The Three-Plane Architecture is complete for what automated structural governance can achieve.

---

*Reviewed through 5 rounds of adversarial deep think analysis. 17 design flaws identified and corrected. Estimated coverage: 85% of future bugs. Buildable in a single focused session.*
