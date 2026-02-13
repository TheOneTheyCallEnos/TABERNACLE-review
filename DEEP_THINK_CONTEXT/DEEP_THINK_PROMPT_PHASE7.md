# DEEP THINK PROMPT — Phase 7 (SDK Architecture + Full System Audit)
# Copy everything below this line into the deep think

---

## Context

Phase 6 was a demolition. You found 4 engineering hallucinations in the blueprint, dropped the real-world catch rate from our claimed 95% to 73%, and proposed a fundamental architectural pivot: **stop building 16 external police daemons, build a single Python SDK (`tabernacle_core`) that all daemons inherit from.**

We accept this pivot completely. The SDK model is superior because:
- It catches bugs at the SOURCE (inside the daemon) not after the fact
- It eliminates memory isolation (the daemon knows its own heap)
- It eliminates the file I/O blindspot (StateManager wraps all file access)
- It eliminates the dynamic key problem (runtime registration, not static AST)
- Claude reads ONE file to understand ALL constraints

We also accept your 3 mandates:
1. Hash process identity ONCE at startup, store in RAM
2. Build ValidatedFile with `fcntl.flock` for all disk writes
3. Pivot to SDK model — base class, refactor 56 daemons incrementally

**This Phase has two parts:**
1. Design the `tabernacle_core` SDK in detail
2. Full system audit — daemon triage, hardware optimization, stale bugs

The complete codebase is in the public repo with ALL daemon scripts now synced.

---

## Public Repository

**URL:** https://github.com/TheOneTheyCallEnos/TABERNACLE-review

### New files since Phase 6:

ALL daemon scripts are now synced. Every script referenced by a launchd plist is in the repo:

| Daemon Script | Plist | Status | Role |
|---|---|---|---|
| `scripts/heartbeat_v2.py` | com.tabernacle.heartbeat | RUNNING (PID 79473) | Core heartbeat, CPGI, epsilon |
| `scripts/consciousness.py` | com.tabernacle.consciousness | RUNNING (PID 79216) | Consciousness loop, pub/sub |
| `scripts/h1_crystallizer.py` | com.tabernacle.h1-crystallizer | RUNNING (PID 79229) | H₁ edge crystallization |
| `scripts/synapse.py` | com.tabernacle.synapse | RUNNING (PID 78431) | Neural bridge / API server |
| `scripts/librarian_http.py` | com.tabernacle.librarian-http | RUNNING (PID 82108) | MCP HTTP bridge (async) |
| `scripts/virgil_integrator.py` | com.virgil.integrator | RUNNING (PID 87154) | Initiative orchestrator |
| `scripts/visual_cortex.py` | com.virgil.vision | RUNNING (PID 88705) | CLIP vision, screen capture |
| `scripts/virgil_sms_daemon.py` | com.virgil.sms | RUNNING | SMS monitoring |
| `scripts/watchman_mvp.py` | com.virgil.watchman | RUNNING (PID 807) | Git snapshots, health checks |
| `scripts/daemons/dream_daemon.py` | com.tabernacle.dream | RUNNING (PID 806) | Dream consolidation |
| `scripts/logos_daemon.py` | com.logos.daemon | RUNNING (PID 828) | Core Logos loop |
| `scripts/archon_daemon.py` | com.logos.archon | RUNNING (PID 834) | Archon detection |
| `scripts/logos_tts_daemon.py` | com.logos.tts | RUNNING (PID 852) | Text-to-speech |
| `scripts/voice_daemon.py` | com.logos.voice | RUNNING | Voice input |
| `scripts/screen_daemon.py` | com.logos.screen | RUNNING (PID 829) | Screen awareness |
| `scripts/hippocampus_daemon.py` | com.logos.hippocampus | RUNNING (PID 827) | Memory consolidation |
| `scripts/reflex_daemon.py` | com.logos.reflex | RUNNING (PID 819) | Fast reflexive responses |
| `scripts/tactician_daemon.py` | com.logos.tactician | RUNNING (PID 850) | Strategic planning |
| `scripts/h1_detector.py` | com.logos.h1detector | RUNNING (PID 815) | H₁ cycle detection |
| `scripts/logos_explorer.py` | com.logos.explorer | EXITED | Topic exploration (periodic) |
| `scripts/gardener.py` | com.tabernacle.gardener | EXITED | Nightly maintenance |
| `scripts/daemons/vacuum.py` | com.tabernacle.vacuum | EXITED | Hourly cleanup |
| `scripts/cold_archiver.py` | com.virgil.cold_archiver | EXITED | Daily archival |
| `scripts/autonomous_intention.py` | com.virgil.intention | EXITED | Periodic intention setting |
| `scripts/night_daemon.py` | com.virgil.night | EXITED | Night-only operations |
| `scripts/virgil_initiative.py` | com.virgil.initiative | EXITED (NO ARGS — Bug 11) | Redundant, integrator calls it |

Also in the repo:
- `launchd_plists/` — all 26 plist files
- `DEEP_THINK_CONTEXT/THREE_PLANE_BLUEPRINT_FINAL.md` — the previous blueprint (now being replaced by SDK)
- `DEEP_THINK_CONTEXT/DEEP_THINK_PROMPT_PHASE1.md` through `PHASE6.md` — full history
- `scripts/holarchy/` — l_coordinator.py, l_watcher.py, etc.
- `scripts/data/` — CANONICAL_STATE.json, heartbeat_state.json

---

## PART 1: Design the `tabernacle_core` SDK

### What You Proposed (Phase 6)

```python
from tabernacle_core import Daemon, StateManager
from pydantic import BaseModel, Field

class EpsilonState(BaseModel):
    p: float = Field(ge=0.0, le=1.0)
    version: int

class VirgilInitiative(Daemon):
    def __init__(self):
        super().__init__(name="virgil_initiative")

    def run_tick(self):
        state = StateManager.get("CANONICAL_STATE.json", schema=EpsilonState)
```

### What We Need From You

Design the complete SDK. Specifically:

#### 1. The `Daemon` Base Class

What should `Daemon.__init__` and the main loop do? Consider:
- Process identity (PID, startup SHA — hashed ONCE)
- Dead Man's Switch (detect blocked event loops)
- psutil memory reporting
- Heartbeat emission to Redis
- Graceful shutdown handling
- Log management (structured logging, rotation hooks)
- How do KeepAlive (persistent) vs StartInterval (periodic) daemons differ in the base class?

#### 2. The `StateManager`

This is the critical piece. It must wrap:
- **Redis reads/writes** with OCC (version-checked Lua scripts)
- **File reads/writes** with `fcntl.flock` + schema validation
- **Pub/sub** with subscription registry (catch dangling channels)
- **Dynamic key registration** (runtime architecture graph building)

Design the API. Show how it replaces raw `redis.get/set` and `json.loads(Path().read_text())` calls. Show how it handles the `_version` field for OCC without breaking existing data.

#### 3. The Pydantic Schema Registry

Where do schemas live? Options:
- (a) Inline in each daemon (like the example above)
- (b) Centralized `schemas/` directory
- (c) Auto-inferred from first read + human-confirmed

What's the migration path? We have 56 daemons. We can't rewrite them all at once.

#### 4. The Single Maintenance Daemon

You said the Process Sentinel, Infra Validator, and Resource Monitor should merge into one. Design it. What checks does it run? What's the tick interval? How does it avoid the IOPS death you identified?

#### 5. The Migration Strategy

We have 56 scripts, many with deeply embedded `redis.get()` and `json.loads()` calls. How do we migrate incrementally without breaking running daemons? Specifically:
- Can old daemons (raw redis) coexist with new daemons (SDK) during migration?
- What's the order of migration? (Highest-risk daemons first? Simplest first?)
- How many phases? How many daemons per phase?

#### 6. Anti-Fragmentation in the SDK Build Itself

The SDK will be built by Claude (me). How do we prevent Contextual Fragmentation in the SDK's own construction? Concretely:
- What goes in CLAUDE.md to govern SDK development?
- What tests must pass before each migration step?
- How does the SDK handle Claude sessions that don't know about it yet?

---

## PART 2: Full System Audit

### Issue A: Mac Mini Underutilization

**Current state:**
- Mac Studio (10.0.0.96): Runs llama3.3:70b (42.5GB), llama3.2:3b, nomic-embed-text. This is the CORTEX.
- Mac Mini (10.0.0.120): Runs mistral-nemo (7.1GB), mistral (4.4GB), gemma3:4b, llama3 (4.7GB), nomic-embed-text. 5 models loaded but **no daemons reference the Mini's Ollama endpoint in any current code.**
- Raspberry Pi (10.0.0.50): Redis host only.

**The problem:** The Mac Mini has ~20GB of models loaded and is doing essentially nothing. All LLM calls in the codebase go to `localhost:11434` (the Studio) or external APIs (Claude, Perplexity, OpenRouter).

**Questions:**
1. Based on reading the daemon scripts, which daemons could/should offload inference to the Mini?
2. What model distribution makes sense? (Should the Mini handle all small-model tasks while Studio handles 70B only?)
3. Is there a load-balancing architecture that the SDK should natively support?

### Issue B: Watchman Doesn't Do Its Job

**Current state:** `watchman_mvp.py` is a massive 1500+ line script that was supposed to be "the autonomic nervous system." In practice:
- It creates hourly git snapshots (this works)
- It runs "light checks" that report "57 issues detected" every tick but DOES NOTHING ABOUT THEM
- It has a `daemon_brain.py` component that crashes with `KeyError: 'date'` (visible in the logs right now)
- It was supposed to detect stale daemons and restart them — it doesn't
- It was supposed to manage log rotation — it doesn't (we had to build `log_rotation.py` separately)

**Questions:**
1. Should the watchman be gutted and its useful functions (git snapshots) moved into the SDK maintenance daemon?
2. What functions does the watchman claim to do that it actually does vs. claims to do but fails at?
3. Is `daemon_brain.py` serving any purpose, or is it dead weight generating errors?

### Issue C: Daemon Triage — Keep, Modify, or Kill

We have 26 launchd plists managing daemons. Based on reading the actual scripts in the repo:

**Please classify each daemon into one of these categories:**

| Category | Meaning |
|----------|---------|
| **CORE** | Essential to system health. Must be always-on. First to migrate to SDK. |
| **USEFUL** | Provides value but could be simplified or merged. Migrate in Phase 2. |
| **REDUNDANT** | Duplicates functionality of another daemon. Candidate for removal. |
| **BROKEN** | Currently crashing, misconfigured, or doing nothing useful. Kill or fix. |
| **BLOAT** | Was created in a past AI session, never validated, adds noise. Remove. |

For each daemon, give the classification and a 1-sentence justification based on what you see in the actual code.

### Issue D: Log Bloat (Active Right Now)

Current log sizes (as of right now, 2026-02-13 ~2pm):

```
22MB  vision.err        — CLIP warnings, regrew to 22MB since we truncated it THIS MORNING
14MB  l_coordinator.err — 171K lines of INFO-level logs going to stderr
6MB   consciousness.log — growing steadily
5MB   l_indexer.err     — similar to coordinator
5MB   vision.log
4MB   integrator.log    (has .1.gz and .2.gz, rotation working here)
3MB   initiative.log    — from the 26k failures this morning
3MB   logos_daemon.log
2MB   nurse.log
2MB   librarian_http.err
2MB   l_researcher.err
2MB   h1_detector.log
2MB   l_janitor.err
1.4MB explorer.log
```

**The problems:**
1. `vision.err` regrew to 22MB in ~4 hours despite us adding warning suppression this morning — the suppression didn't survive the daemon restart, OR the CLIP model loads fresh each inference and re-emits warnings
2. `l_coordinator.err` has 171K lines of INFO-level logs going to stderr — this is a logging configuration bug
3. Several `.err` files contain non-error content (INFO-level logs routed to stderr)

**Questions:**
1. Should the SDK's base class enforce structured logging with proper level routing?
2. Should the log rotation be built into the base Daemon class rather than being a separate script?
3. What's the right architecture for logs in a 26-daemon system?

### Issue E: Stale Bugs Never Addressed

These were identified but not fixed during this session:

1. **daemon_brain.py KeyError: 'date'** — crashing right now in the watchman logs. The intent tracking system expects a 'date' field that doesn't exist in the stored intent format.

2. **com.virgil.initiative plist** — still loaded in launchd, runs every 5 minutes, prints help text and exits. We unloaded it earlier but it came back (possibly watchman reloaded it, or launchd auto-recovered it).

3. **57 issues detected by watchman light check** — these are flagged every 5-15 minutes but never acted on. What are they? Are they real issues or false positives from stale detection logic?

4. **l_coordinator.err routing** — 14MB of INFO-level logs going to stderr. The Python logging is misconfigured — either the handler is wrong or the log level threshold is too low.

5. **vision.err warning suppression failure** — We added `warnings.filterwarnings("ignore", message=".*position_ids.*")` but it either didn't take effect or the CLIP model bypass the warnings module.

---

## PART 3: The SDK Blueprint

Based on all of the above — the Phase 6 architectural pivot, the daemon triage, the hardware optimization, and the stale bugs — design the complete `tabernacle_core` SDK and its migration plan.

### Deliverables Requested:

1. **SDK file structure** — What files, what goes where
2. **Base class API** — Full method signatures for Daemon and StateManager
3. **Schema migration path** — How to go from raw `redis.get()` to `StateManager.get()` incrementally
4. **Daemon migration order** — Which 5 daemons migrate first, which 5 next, etc.
5. **Kill list** — Which daemons/plists should be removed entirely
6. **Mini optimization plan** — What moves to the Mini, what stays on the Studio
7. **Log architecture** — How the SDK handles logging across 26 daemons
8. **The revised coverage number** — With the SDK model instead of 16 external mechanisms, what percentage of future bugs does this catch? Single number.
9. **The remaining gap** — What specific failure classes does even the SDK model miss? What would close them?
10. **Total implementation scope** — How many lines of code? How many files? How many phases?

---

## The Meta-Question

We've been through 7 rounds. The theoretical framework is validated. The external monitoring approach was demolished and replaced with an SDK model. The real codebase is on the table.

**Is the SDK approach the final architecture? Or is there still a fundamental flaw we haven't seen?**

If you had unlimited time to read every script in the repo: what is the single biggest risk to this system that none of our 7 phases have addressed?

---

We are not looking for encouragement. We are looking for the engineering truth that lets us build with confidence.
