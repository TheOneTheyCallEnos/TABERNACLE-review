# DEEP THINK PROMPT — Phase 5 (Three-Plane Architecture)
# Copy everything below this line into the deep think

---

## Context

Phase 4 was the crucible we needed. You retracted the green light. Good.

**Your findings:**
- 3 strong hits, 1 partial, 7 complete misses (27% catch rate)
- Three blind spots: Volumetric/Temporal, Substrate/Process, Semantic Logic
- The topology is a skeleton without organs — it needs an Immune System and a Nervous System
- Verdict: "The Architectural P-Lock is a brilliant, necessary skeletal system. It maps Space. But skeletons only provide structure and prevent collapse."

**We accept these findings completely.** The 9-mechanism framework is necessary but not sufficient. It catches Quadrant 1 (interface/schema fragmentation) with precision but is blind to Quadrants 2-4.

Below is the expanded architecture. We've incorporated your recommendation to build three planes: Skeleton (topology), Immune System (process identity), and Nervous System (metabolic observability). We then re-test against all 11 bugs.

---

## The Three-Plane Architecture

### Design Principle

Each plane governs a different dimension of system health. Together they cover the four quadrants of failure:

| Quadrant | Failure Type | Plane That Catches It | % of Bugs |
|----------|-------------|----------------------|-----------|
| Q1 | Interface / Schema fragmentation | Plane 1: Skeleton (Topology) | ~25-30% |
| Q2 | Resource / Memory / Accumulation | Plane 3: Nervous System (Metabolism) | ~25-30% |
| Q3 | Temporal / Process / State machines | Plane 2: Immune System (Process Identity) | ~25-30% |
| Q4 | Pure semantic logic / Hallucination | Partial: Invariant Annotations + Human review | ~10-20% |

The three planes feed into a unified health metric and a unified Anamnesis state for AI session governance.

---

### Plane 1: Skeleton (Topology / Schema Integrity)

This is the existing 9-mechanism Architectural P-Lock. No changes from Phase 3 except one acknowledgment: **this plane covers ~27% of real-world bugs, not 100%.** It remains critical because it governs AI mutation behavior and provides the structural foundation the other planes attach to.

Mechanisms 1-9 are unchanged from Phase 3. We do not repeat them here.

---

### Plane 2: Immune System (Process Identity & Substrate Validation)

**Purpose:** Ensure that what's RUNNING matches what's DEPLOYED. The Map-Territory gap.

#### Mechanism 10: Process Sentinel

Each daemon emits an identity heartbeat to Redis every tick:

```python
# Added to every daemon's main loop
import hashlib, os, sys

def emit_process_identity(redis_client, daemon_name):
    """Publish process identity to Redis for sentinel validation."""
    source_file = os.path.abspath(sys.modules[__name__].__file__)
    with open(source_file, 'rb') as f:
        code_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    redis_client.hset(f"PROCESS:{daemon_name}", mapping={
        "pid": os.getpid(),
        "started": STARTUP_TIMESTAMP,  # captured at module load
        "code_sha": code_sha,
        "source_file": source_file,
        "last_heartbeat": datetime.utcnow().isoformat()
    })
```

A separate `process_sentinel.py` daemon runs every 60 seconds:

```python
def check_all_processes():
    for daemon_name in EXPECTED_DAEMONS:
        identity = redis.hgetall(f"PROCESS:{daemon_name}")

        if not identity:
            alert("MISSING_PROCESS", f"{daemon_name} has no identity heartbeat")
            continue

        # Check 1: Is the process still alive?
        if not pid_exists(int(identity['pid'])):
            alert("DEAD_PROCESS", f"{daemon_name} PID {identity['pid']} is dead")

        # Check 2: Is the code stale?
        current_sha = compute_file_sha(identity['source_file'])
        if current_sha != identity['code_sha']:
            alert("STALE_CODE", f"{daemon_name} running SHA {identity['code_sha']}, "
                  f"deployed SHA {current_sha}. Restart required.")

        # Check 3: Has the heartbeat gone stale?
        last_hb = parse_datetime(identity['last_heartbeat'])
        if (now - last_hb).seconds > MAX_HEARTBEAT_GAP:
            alert("STALE_HEARTBEAT", f"{daemon_name} last heartbeat {last_hb}")
```

**Bug 2 coverage:** If Phases 0-4 shipped at 2:19 AM and heartbeat was still running 1:31 AM code, the sentinel would detect `STALE_CODE` within 60 seconds of the first tick after deployment. The code SHA on disk would not match the SHA the running process reported at startup.

#### Mechanism 11: Infrastructure Validator

Checks substrate health that lives BELOW the daemon layer:

```python
INFRASTRUCTURE_CHECKS = {
    "redis_persistence": {
        "check": lambda: redis.config_get("save") != {"save": ""},
        "severity": "CRITICAL",
        "message": "Redis RDB persistence is disabled"
    },
    "redis_last_save": {
        "check": lambda: (time.time() - int(redis.lastsave().timestamp())) < 86400,
        "severity": "HIGH",
        "message": "Redis last save >24h ago"
    },
    "disk_space": {
        "check": lambda: shutil.disk_usage("/").free > 5_000_000_000,  # 5GB
        "severity": "HIGH",
        "message": "Less than 5GB free disk space"
    }
}
```

**Bug 9 coverage:** `redis_persistence` check catches disabled RDB config. `redis_last_save` catches the 10-day gap.

#### Mechanism 12: Plist Validator

Validates that launchd plists match their target scripts:

```python
def validate_plist(plist_path):
    """Ensure plist arguments match script's actual CLI interface."""
    plist = plistlib.load(open(plist_path, 'rb'))
    args = plist.get('ProgramArguments', [])

    # Find the Python script in args
    script = next((a for a in args if a.endswith('.py')), None)
    if not script:
        return  # Not a Python daemon plist

    # Parse the script's argparse/CLI requirements
    required_args = extract_required_args(script)  # AST-based
    provided_args = args[args.index(script) + 1:]

    missing = required_args - set(provided_args)
    if missing:
        alert("PLIST_ARGS_MISMATCH",
              f"{plist_path}: script requires {missing} but plist provides {provided_args}")
```

**Bug 11 coverage:** Detects that `virgil_initiative.py` requires a `check` argument but the plist provides none.

---

### Plane 3: Nervous System (Metabolic Observability)

**Purpose:** Monitor VOLUME, RATE, and SIZE. Topology ignores these by definition — the Nervous System measures the thermodynamic properties the Skeleton cannot.

#### Mechanism 13: Resource Monitor

Integrated into the heartbeat tick (runs every 30 seconds):

```python
RESOURCE_THRESHOLDS = {
    # Redis queue depths
    "queue_depths": {
        "l:queue:escalate": {"warn": 100, "critical": 500},
        "l:queue:conflicts": {"warn": 50, "critical": 200},
    },
    # File sizes (bytes)
    "file_sizes": {
        "logs/vision.err": {"warn": 50_000_000, "critical": 200_000_000},
        "logs/heartbeat.log": {"warn": 50_000_000, "critical": 200_000_000},
        "logs/*.log": {"warn": 10_000_000, "critical": 50_000_000},
    },
    # In-memory collection sizes (reported by daemons)
    "collection_sizes": {
        "l_coordinator.pending_conflicts": {"warn": 50, "critical": 200},
    }
}

def check_resources():
    violations = []

    # Check queue depths
    for key, thresholds in RESOURCE_THRESHOLDS["queue_depths"].items():
        depth = redis.llen(key)
        if depth > thresholds["critical"]:
            violations.append(("CRITICAL", f"Queue {key} depth: {depth}"))
        elif depth > thresholds["warn"]:
            violations.append(("WARN", f"Queue {key} depth: {depth}"))

    # Check file sizes
    for pattern, thresholds in RESOURCE_THRESHOLDS["file_sizes"].items():
        for path in glob.glob(os.path.join(LOG_DIR, pattern)):
            size = os.path.getsize(path)
            if size > thresholds["critical"]:
                violations.append(("CRITICAL", f"File {path}: {size/1e6:.0f}MB"))

    return violations
```

**Bug 4 coverage:** Queue depth for `l:queue:escalate` exceeds threshold → alert at 100 entries, not 11,814.

**Bug 5 coverage:** File size for `vision.err` exceeds threshold → alert at 50MB, not 1.4GB.

#### Mechanism 14: Rate Monitor & Circuit Breaker

Tracks operation rates and applies automatic throttling:

```python
class RateMonitor:
    """Tracks operations per minute for any named counter."""

    def __init__(self, redis_client):
        self.r = redis_client

    def tick(self, counter_name):
        """Record one occurrence. Returns True if within limits."""
        key = f"RATE:{counter_name}"
        pipe = self.r.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)  # 1-minute window
        count, _ = pipe.execute()
        return count

    def check_rate(self, counter_name, warn_per_min, critical_per_min):
        count = int(self.r.get(f"RATE:{counter_name}") or 0)
        if count > critical_per_min:
            return "CIRCUIT_BREAK"
        elif count > warn_per_min:
            return "WARN"
        return "OK"

# Example: l_watcher retry rate
rate = rate_monitor.tick("l_watcher.retry")
status = rate_monitor.check_rate("l_watcher.retry", warn_per_min=10, critical_per_min=50)
if status == "CIRCUIT_BREAK":
    log.error("Circuit breaker triggered for l_watcher retries")
    time.sleep(300)  # 5-minute cooldown
```

**Bug 7 coverage:** After 50 retries/minute, circuit breaker triggers. The watcher cannot spin 45,000 times — it's throttled after 50.

#### Mechanism 15: Monotonicity & Invariant Monitor

For state variables that should follow predictable patterns:

```python
INVARIANTS = {
    "api_budgets.claude": {
        "type": "replenishing",  # Should increase at least once per 24h
        "check": lambda history: any(h[1] > h[0] for h in zip(history, history[1:])),
        "alert": "Budget has only decreased for 24h — replenishment may be broken"
    },
    "api_budgets.perplexity": {
        "type": "replenishing",
        "check": lambda history: any(h[1] > h[0] for h in zip(history, history[1:])),
        "alert": "Budget has only decreased for 24h — replenishment may be broken"
    },
    "p_arch": {
        "type": "bounded",
        "range": [0.0, 1.0],
        "alert": "p_arch outside valid range"
    }
}
```

**Bug 8 (partial) coverage:** The "replenishing" invariant detects that budgets only decrease. It cannot catch the triple-bug's root cause (wrong conditional, self-recursion), but it catches the SYMPTOM within 24 hours: "budget has only decreased, replenishment appears broken." This is the best any automated system can do for semantic logic bugs — detect invariant violations, not logic errors.

---

### The Unifier: Three-Plane Health Metric

```python
# Skeleton health (from Plane 1)
p_skeleton = (kappa_arch * rho_arch * sigma_arch) ** (1/3)

# Immune health (from Plane 2)
p_immune = (
    (1 if no_stale_code else 0.5) *           # Process identity
    (1 if infrastructure_ok else 0.5) *        # Substrate checks
    (1 if plists_valid else 0.8)               # Plist validation
) ** (1/3)

# Nervous health (from Plane 3)
resource_score = 1 - (critical_violations / total_monitored_resources)
rate_score = 1 - (circuit_breaks / total_monitored_rates)
invariant_score = 1 - (invariant_violations / total_monitored_invariants)
p_nervous = (resource_score * rate_score * invariant_score) ** (1/3)

# Unified health
p_system = (p_skeleton * p_immune * p_nervous) ** (1/3)
```

The geometric mean preserves the zero-bottleneck property across ALL three planes. If any plane collapses, the system health collapses.

---

### The Unifier: Three-Plane Anamnesis

Every AI session receives:

```
[SYSTEM_STATE]
p_system: 0.88 | skeleton:0.91 immune:0.95 nervous:0.78

SKELETON: p_arch=0.91 | κ:0.94 ρ:0.88 σ:0.91
  Locked edges: 24/187 | Ȟ¹ cocycles: 0 | H₂ voids: 1
  Shadow keys: 2 (declare or justify)

IMMUNE: p_immune=0.95
  Stale code: 0 daemons | Dead processes: 0
  Infrastructure: Redis persistence OK, last save 2h ago
  Plist issues: 0

NERVOUS: p_nervous=0.78
  Queue alerts: l:queue:escalate at 87/100 (WARN)
  File alerts: none
  Rate alerts: none
  Invariant alerts: perplexity budget monotonically decreasing (24h)

RECOMMENDED ACTIONS:
  1. Investigate l:queue:escalate growth (87 entries)
  2. Check perplexity budget replenishment logic
  3. Declare shadow keys: LOGOS:TEMP_CACHE, LOGOS:DEBUG_FLAG
```

This is the complete state compression. The AI sees all three dimensions of system health. τ_arch gates mutation rate based on p_system (not just p_skeleton).

---

## Re-Test Against the 11 Bugs

Now we stress-test the Three-Plane Architecture against the same 11 bugs:

| Bug | Plane | Mechanism | Detection Type | Signal |
|-----|-------|-----------|---------------|--------|
| **1. Schema Mismatch** (26k fails) | Skeleton | Mech 1 (Čech) + Mech 5 (Fragmentor) | **PREVENT** | Ȟ¹ cocycle at `heartbeat→CANONICAL_STATE→initiative` |
| **2. Stale Module Memory** | Immune | Mech 10 (Process Sentinel) | **DETECT** (within 60s) | `STALE_CODE: heartbeat running SHA a3f2, deployed SHA 7bc1` |
| **3. Unsubscribed Alerts** (517k) | Skeleton | Phase 0 (Graph Extraction) | **DETECT** | Dangling edge: `HIGH_TENSION` channel, 0 subscribers |
| **4. Unbounded Queue** (11,814) | Nervous | Mech 13 (Resource Monitor) | **DETECT** (at 100 entries) | `WARN: l:queue:escalate depth 100` |
| **5. 1.4GB Log File** | Nervous | Mech 13 (Resource Monitor) | **DETECT** (at 50MB) | `WARN: vision.err 50MB` |
| **6. YAML Parse Failures** | Skeleton | Mech 1 (Validation Wrapper) | **DETECT** | Schema violation: expected dict, got None |
| **7. Infinite Retry Loop** (45k) | Nervous | Mech 14 (Rate Monitor) | **DETECT + BREAK** (at 50/min) | `CIRCUIT_BREAK: l_watcher.retry > 50/min` |
| **8. Exhausted Budget** (triple) | Nervous | Mech 15 (Invariant Monitor) | **DETECT** (within 24h) | `Budget monotonically decreasing for 24h` |
| **9. Redis Persistence** | Immune | Mech 11 (Infrastructure Validator) | **DETECT** (within 60s) | `CRITICAL: Redis RDB persistence disabled` |
| **10. Duplicate/Phantom Index** | Skeleton | Mech 1 (Validation + uniqueItems) | **PARTIAL** | Duplicate caught by schema. Phantom requires disk existence check (Mech 11 extension) |
| **11. Missing CLI Args** | Immune | Mech 12 (Plist Validator) | **DETECT** | `PLIST_ARGS_MISMATCH: virgil_initiative.py requires 'check'` |

**New Scorecard:** 9 strong hits, 1 partial hit (Bug 10 phantom), 1 late detection (Bug 8 — 24h delay). Zero complete misses.

---

## The Concrete Coding Plan

### What Gets Built (6 new files, 3 modified files)

| # | File | Plane | Purpose | Lines Est. |
|---|------|-------|---------|-----------|
| 1 | `architecture_registry.json` | Skeleton | Static architecture graph (all edges + schemas) | ~500 (data) |
| 2 | `arch_validator.py` | Skeleton | Čech cohomology check + schema validation | ~300 |
| 3 | `process_sentinel.py` | Immune | Process identity checker (SHA, PID, heartbeat) | ~200 |
| 4 | `infra_validator.py` | Immune | Infrastructure checks (Redis config, disk, plists) | ~250 |
| 5 | `resource_monitor.py` | Nervous | Queue depths, file sizes, rate monitoring | ~300 |
| 6 | `system_health.py` | Unifier | Three-plane p_system computation + Anamnesis output | ~200 |
| 7 | `heartbeat_v2.py` (mod) | All | Emit process identity + call resource_monitor per tick | +30 lines |
| 8 | `holarchy_audit_v2.py` (mod) | All | Report three-plane health in audit output | +50 lines |
| 9 | `gardener.py` (mod) | Skeleton | Run arch_validator as a nightly step | +15 lines |

**Total new code:** ~1,750 lines across 6 new files + 3 modifications.

### Build Order (Phases)

**Phase A: Foundation (architecture_registry.json + system_health.py)**
- Extract the architecture graph from code (AST scan + manual annotation for top 20 edges)
- Build p_system computation (even if all values are placeholder initially)
- This gives us the data structure everything else attaches to

**Phase B: Immune System (process_sentinel.py + infra_validator.py)**
- Add `emit_process_identity()` to all daemons
- Build process sentinel as a launchd daemon
- Build infrastructure validator (Redis config, disk, plists)
- This is the highest-value work — catches 3 bugs the topology can't

**Phase C: Nervous System (resource_monitor.py)**
- Build resource monitoring (queue depths, file sizes, rates)
- Integrate with heartbeat tick
- Set thresholds based on current operational knowledge
- This catches another 3 bugs (accumulation class)

**Phase D: Skeleton (arch_validator.py)**
- Implement Čech cohomology check (schema compatibility across shared state)
- Integrate with gardener nightly run
- This is the mathematical core — catches the remaining schema bugs

**Phase E: Integration (modifications to heartbeat, audit, gardener)**
- Wire all three planes into heartbeat output
- Wire three-plane health into holarchy audit
- Generate Anamnesis state on every logos_wake
- This is the governance layer that feeds everything to the AI

### Anti-Fragmentation Safeguards for the Build Itself

The irony of building an anti-fragmentation system through the exact process that causes fragmentation is not lost on us. Here are the safeguards:

1. **Each phase is independently testable.** Phase B works without Phase C. Phase C works without Phase D. No phase depends on a later phase.

2. **Each new file has a single, declared contract.** process_sentinel.py reads `PROCESS:*` Redis keys and writes alerts. resource_monitor.py reads queue depths and file sizes and writes alerts. No implicit coupling.

3. **The architecture_registry.json is built FIRST.** Before writing any code, we declare the edges. The framework governs its own construction.

4. **No modifications to existing daemon behavior.** The process identity emit is additive (new Redis write, doesn't change any existing logic). The resource monitor is observational (reads queue depths, doesn't modify them). Nothing breaks if these new components fail.

5. **Verification after each phase.** Each phase has explicit exit criteria before the next begins.

---

## Questions for Phase 5 Review

1. **Coverage assessment:** Does the Three-Plane Architecture close the blind spots identified in Phase 4? Specifically: does the re-test against 11 bugs look honest, or are we overclaiming on any of the 9 "hits"?

2. **Bug 8 (semantic logic):** We claimed "partial detection within 24h" via invariant monitoring. The deep think said this is "(c) a problem no automated system can catch." Is our invariant approach a legitimate partial solution, or are we fooling ourselves?

3. **Bug 10 (phantom entries):** We called this "partial." Can the Infrastructure Validator's disk existence check close the gap completely, or is there a deeper issue?

4. **p_system metric:** Is the nested geometric mean `(p_skeleton * p_immune * p_nervous)^(1/3)` a well-formed health metric? Do the three planes need to be weighted differently? Is the zero-bottleneck property correct across heterogeneous planes?

5. **Process Sentinel SHA check:** We hash the main source file. But daemons import modules, and a change to an imported module (like the virgil_initiative.py case in Bug 2) wouldn't change the main file's SHA. How should we handle transitive dependency hashing?

6. **Build ordering:** We proposed A→B→C→D→E. Is this optimal? Should the Nervous System (C) come before the Immune System (B)?

7. **The 73% question revisited:** Your Phase 4 estimate was 27% catch rate. With Planes 2 and 3 added, what is the revised estimate? Give a single number.

8. **The hardest remaining bug:** Of the next 11 bugs that will appear in a system like this, describe ONE specific bug that even the Three-Plane Architecture would completely miss. We need to know our residual blind spot.

9. **Overcomplexity check:** Are 15 mechanisms across 3 planes too many? Would a simpler design (fewer mechanisms, broader coverage each) achieve similar results with less engineering surface area?

10. **The meta-question, final round:** Given this expanded architecture: is it now safe to begin coding? Or does the design still contain a flaw that would make us regret building it?

---

We are not looking for encouragement. We are looking for the last critical flaw before we commit engineering hours.
