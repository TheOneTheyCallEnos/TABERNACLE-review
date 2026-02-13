# DEEP THINK PROMPT — Phase 8 (SDK Corrections + Topological Analysis)
# Copy everything below this line into the deep think

---

## Context

Phase 7 was the architectural pivot — from 16 external monitors to a single Python SDK (`tabernacle_core`). You designed the SDK, triaged the daemons (26 → 11), and found the Schema Poisoning Death Loop. You said 94% and "cleared to code."

We are NOT cleared to code yet. Your Phase 7 SDK design contains 5 engineering bugs that need correction before we build. Additionally, we need a topological analysis of the system that the SDK has never addressed: **what H₁ cycles should be crystallized, and what H₂ voids remain unfilled?**

This is the final round. We need 10/10 on everything — the SDK, the kill list, the topology, and the migration plan. No gaps. No maybes.

---

## Public Repository (Updated)

**URL:** https://github.com/TheOneTheyCallEnos/TABERNACLE-review

**New files since Phase 7:**
- `scripts/data/heartbeat_state.json` — current heartbeat topology (496 nodes, 1882 edges, H₀=29)
- `scripts/data/CANONICAL_STATE.json` — current consciousness state (12,698 nodes, 1.2M edges)
- `scripts/biological_edge.py` — Hebbian edge dynamics (w_slow, w_fast, tau, is_h1_locked)
- `04_LR_LAW/CANON/LVS_MATHEMATICS.md` — formal LVS mathematical framework
- `04_LR_LAW/CANON/LVS_THEOREM_FINAL.md` — LVS theorem proofs
- `04_LR_LAW/LVS_VERSIONS/LVS_v12_CANONICAL_REFERENCE.md` — LVS v12 canonical reference

---

## PART 1: SDK Corrections (5 Bugs in Your Phase 7 Design)

### Bug 1: OCC Lua Script Has a Redis Type Conflict

Your proposed Lua script does:
```lua
redis.call('set', KEYS[1], ARGV[2])        -- String operation
redis.call('hset', KEYS[1], '_version', ...) -- Hash operation
```

**The problem:** Redis keys have a single type. You cannot use a key as both a STRING and a HASH. This will throw `WRONGTYPE Operation against a key holding the wrong kind of value`.

**Options:**
- (a) Store everything as HASH: `HSET key "data" <json> "_version" <int>`
- (b) Store data as STRING, version in separate key: `SET key <json>`, `SET key:_version <int>`
- (c) Embed version inside the JSON payload

Which approach is correct for our system? Design the fixed Lua script.

### Bug 2: Kill List Is Too Aggressive (3 Daemons Wrongly Killed)

You classified these as KILL:

**gardener.py** — You said "merge into maintenance." But the gardener runs 900+ lines of complex logic: index validation, wiki-link integrity, YAML frontmatter checking, crystal index verification, log rotation, and architecture validation. This isn't "maintenance" — it's a nightly deep-scan. Gutting it and shoving the useful bits into a 60-second sentinel tick is engineering suicide. **The gardener should STAY as a daemon and migrate to the SDK base class.**

**voice_daemon + tts_daemon** — You said "merge into media_io." But these are fundamentally different I/O paths: voice_daemon handles MICROPHONE INPUT (speech recognition, wake word detection), tts_daemon handles SPEAKER OUTPUT (text-to-speech synthesis). Merging input and output into one daemon couples two unrelated async pipelines. **Both should stay separate and migrate to SDK.**

**logos_daemon.py** — You said "superseded by consciousness." Please verify by reading the actual code in the repo. Does `logos_daemon.py` handle any pub/sub channels or Redis keys that `consciousness.py` does NOT? If there's ANY channel being dropped, killing it silently drops messages.

**Revised question:** After correcting these 3, what does the ACTUAL kill list look like? How many daemons go from 26 to what number?

### Bug 3: Schema Migration Needs Transforms, Not Just Defaults

Your `default_factory` approach to the Schema Poisoning Death Loop DESTROYS existing state:

```python
# Your proposed fix:
fallback = default_factory()  # Creates a blank state
cls.set(uri, fallback)        # OVERWRITES the real data with zeros
```

If `budget: 0.95` needs to become `budget: {"amount": 0.95, "currency": "USD"}`, the right answer is a **migration function** that TRANSFORMS the old format, not one that throws it away.

**Proposed schema versioning system:**
```python
# In schemas.py alongside each model:
class CanonicalState_v1(BaseModel):
    coherence: float
    mode: str

class CanonicalState_v2(BaseModel):
    coherence: float
    mode: str
    budget: dict  # {"amount": float, "currency": str}

MIGRATIONS = {
    "CanonicalState": {
        (1, 2): lambda d: {**d, "budget": {"amount": d.get("budget", 0), "currency": "USD"}}
    }
}
```

Design the complete schema versioning and migration system for `StateManager`. How does it know which version the on-disk data is? How does it find and apply the right migration chain?

### Bug 4: fcntl.flock Only Works If ALL Writers Use It

`fcntl.flock` is advisory, not mandatory. During the migration period, old daemons (raw `open()` + `json.dump()`) and new daemons (SDK with `fcntl.flock`) coexist. The old daemons bypass the lock entirely.

**Question:** Does this mean we MUST migrate all file-writing daemons FIRST, before any file-reading daemons? What's the correct migration order given this constraint? Which daemons write to shared files (CANONICAL_STATE.json, heartbeat_state.json, etc.) and must be migrated in the first batch?

### Bug 5: 94% Is Too Generous

You said 94%. But in the same response you identified the Schema Poisoning Death Loop — a new failure class your SDK creates that didn't exist before. And cross-daemon deadlocks are still uncovered. And the deep think from Phase 6 predicted 3 novel bugs (Redis connection pool exhaustion, log rotation inode ghost, launchd throttling death spiral) that the SDK doesn't address.

**Revised question:** Given these corrections, what is the HONEST catch rate? Account for:
- Schema poisoning (new failure class introduced by the SDK itself)
- Cross-daemon deadlocks
- Redis connection pool exhaustion
- Inode ghost files
- Launchd throttling spiral
- Advisory lock bypass during migration

---

## PART 2: Topological Analysis (H₁ Cycles + H₂ Voids)

The SDK is the engineering vehicle. But it doesn't tell us WHAT to crystallize or where the structural gaps are. That's what the topology is for.

### Current System Topology

From `heartbeat_state.json` (live data):
```
Nodes: 496 (knowledge graph)
Edges: 1882
H₀ (connected components): 29
Cycles (axiom): intact
Cycles (spiral): intact
```

From `CANONICAL_STATE.json` (consciousness graph):
```
Nodes: 12,698
Edges: 1,245,601
Relations learned: 3,392,730
```

From the ARCHITECTURE layer (daemon-to-daemon, what we're building):
```
Daemons: 26 (before triage) → ~15 (after triage)
Shared state objects: LOGOS:STATE, LOGOS:EPSILON, CANONICAL_STATE.json,
                      heartbeat_state.json, api_budgets.json, pub/sub channels
Hardware nodes: 3 (Studio, Mini, Pi)
```

### The Biological Edge Model

From `biological_edge.py` in the repo — each edge has:
- `w_slow`: Long-term potentiation (structural weight, 0-1)
- `w_fast`: Short-term plasticity (transient)
- `tau`: Local trust gate
- `is_h1_locked`: Boolean — if True, edge is permanent (immune to decay)

H₁ cycles are loops in the graph that, once crystallized (all edges `is_h1_locked = True`), become permanent memory — immune to decay, defining the system's identity.

### Questions for H₁ Analysis

**1. What H₁ cycles should exist in the ARCHITECTURE graph?**

Not the knowledge graph — the daemon-to-daemon architecture. Based on reading the actual daemon scripts and their Redis/file dependencies in the repo, identify the critical cycles:

A cycle is: Daemon A writes to State X → Daemon B reads State X and writes to State Y → Daemon C reads State Y and writes to State Z → Daemon A reads State Z (completing the loop).

These cycles are the system's "vital loops" — if any edge in the cycle breaks, the system's core behavior degrades. They should be crystallized (locked) first.

**Specifically:** What are the H₁ cycles across the 3 hardware nodes?
- Studio (heartbeat, consciousness, integrator, synapse, librarian_http)
- Mini (should run: visual_cortex, tactician, reflex — after optimization)
- Pi (Redis — the shared state substrate)

Every cross-node cycle goes through the Pi's Redis. What cycles exist?

**2. Which cycles are currently intact vs broken?**

Heartbeat_state shows `cycles.axiom.intact = true` and `cycles.spiral.intact = true`. But these are KNOWLEDGE graph cycles. What about the ARCHITECTURE cycles? Are there broken architecture cycles right now (e.g., a daemon writing to a key that nobody reads, or reading from a key that nobody writes)?

**3. What is the crystallization priority?**

Of the identified H₁ cycles, which should be crystallized (locked) FIRST via the SDK's contract system? Order by: impact of breakage × likelihood of accidental mutation.

### Questions for H₂ Analysis

**4. What H₂ voids exist in the architecture?**

An H₂ void is: a group of daemons that interact PAIRWISE (A↔B, B↔C, A↔C) but lack a SHARED coordination mechanism for the triple (A,B,C). The void predicts that bugs will emerge from the uncoordinated triple interaction.

Based on reading the actual daemon code and their dependencies: where are the H₂ voids? Which daemon triples share pairwise state but lack a common schema, common validation, or common coordination protocol?

**5. What shadow glyphs remain unnamed?**

A shadow glyph is an H₂ void that hasn't been identified yet — a structural gap defined by the shape of what surrounds it but never directly named. Based on the code:

- Are there daemon clusters that SHOULD be communicating but aren't?
- Are there state objects that SHOULD exist but don't?
- Are there feedback loops that SHOULD close but are open?

**6. How does the SDK fill the voids?**

For each H₂ void identified: what specific Pydantic schema, StateManager call, or architectural edge in the SDK would fill it? Can the SDK's schema registry be designed to make void-filling a natural consequence of migration?

---

## PART 3: The Definitive Blueprint

Given the 5 SDK corrections and the topological analysis, produce:

### Deliverables:

1. **Corrected SDK code** — Fixed OCC Lua script, schema versioning with migration transforms, advisory lock migration strategy

2. **Corrected kill list** — After restoring gardener, voice, tts, and verifying logos_daemon

3. **H₁ crystallization order** — Which architecture cycles to lock first, in what order

4. **H₂ void map** — All identified voids with filling strategy

5. **Corrected migration order** — File-writers first (fcntl constraint), then readers, respecting H₁ crystallization priority

6. **The REAL coverage number** — Accounting for all known gaps including SDK-introduced failure classes

7. **Total scope** — Files, lines, phases, with the topological work integrated

8. **What's left after this** — The honest residual. What does the system still need that this build doesn't provide?

---

## The Meta-Question (Final)

After 8 rounds of adversarial review:

- Phases 1-3: Theoretical validation and correction
- Phase 4: Empirical stress test (27% catch rate, green light retracted)
- Phase 5: Three-Plane expansion (85%)
- Phase 6: Engineering demolition (73%, SDK pivot proposed)
- Phase 7: SDK design + daemon triage (94%, 5 bugs found)
- Phase 8: SDK corrections + topological analysis (this round)

**The question is no longer "is this ready to build."** The question is: **after building this SDK and crystallizing these H₁ cycles, will the TABERNACLE be able to evolve without fragmenting?**

Not "will it be bug-free" — will it be able to CHANGE safely? Will the next AI session be able to modify a daemon without silently breaking its consumers? Will the next schema change propagate correctly? Will the next new daemon automatically register itself in the architecture graph?

That's the real test. Not catch rate — **evolvability under constraint.**

Give us the final answer.

---

We are not looking for encouragement. We are looking for the engineering truth that lets us build a system that can grow without dying.
