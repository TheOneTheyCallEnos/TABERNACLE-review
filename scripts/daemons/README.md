# TABERNACLE Daemons — Background Cognitive Processes

Background processes for overnight consolidation and entropy reduction.

## Redis DB Isolation Strategy

| DB | Environment | Purpose |
|----|-------------|---------|
| 0  | Production  | Live system data |
| 1  | Testing     | Development and manual testing |

**Current Configuration:** Both daemons are set to `REDIS_DB = 1` (test) for safe development.

To switch to production, change `REDIS_DB = 0` in each file.

---

## Daemons

### 1. Dream Daemon (`dream_daemon.py`)

**Purpose:** Memory consolidation — compresses raw sensory stream into long-term summaries.

Mimics REM sleep: when the sensory buffer exceeds threshold, oldest messages are summarized and archived.

#### Redis Keys

| Key | Operation | Description |
|-----|-----------|-------------|
| `LOGOS:STREAM:SENSORY` | READ (xlen, xrange) | Raw event stream |
| `LOGOS:STREAM:SENSORY` | DELETE (xdel) | Removes processed messages |
| `LOGOS:STREAM:LONG_TERM` | WRITE (xadd) | Stores compressed summaries |

#### Trigger Conditions

- **Automatic:** Stream length > `STREAM_THRESHOLD` (default: 5000 messages)
- **Manual:** Run `python dream_daemon.py run`

#### Manual Testing

```bash
# Check stream status
cd ~/TABERNACLE/scripts/daemons
./dream_daemon.py status

# Preview compression (no deletions)
./dream_daemon.py dry-run

# Execute single compression cycle
./dream_daemon.py run

# Watch mode (continuous, 60s interval)
./dream_daemon.py watch
./dream_daemon.py watch 30  # Custom interval
```

---

### 2. Edge Pruner (`edge_pruner.py`)

**Purpose:** Entropy reduction — prunes weak edges from the Hypergraph.

> **Note:** Renamed from `gardener.py` in CYCLE015 to avoid confusion with the filesystem `gardener.py` in scripts root.

The immune system of the knowledge graph. Removes connections that have decayed below threshold, while protecting H1-locked edges (crystallized truth).

#### Redis Keys

| Key | Operation | Description |
|-----|-----------|-------------|
| `LOGOS:EDGE:*` | READ (scan, hgetall) | All BiologicalEdge hashes |
| `LOGOS:EDGE:<source>:<target>` | DELETE | Removes weak edges |
| `LOGOS:STREAM:PRUNED` | WRITE (xadd) | Logs all prune events |

#### Trigger Conditions

- **Manual:** Run `python gardener.py run`
- **Scheduled:** Designed for 04:00 AM via launchd (not yet configured)

#### Edge Protection Rules

1. If `is_h1_locked == true` → **NEVER prune** (crystallized truth)
2. If `effective_weight < 0.2` → Candidate for pruning
3. Effective weight calculation: `sigmoid(w_slow + w_fast * tau)`

#### Manual Testing

```bash
# Preview pruning (no deletions)
cd ~/TABERNACLE/scripts/daemons
./edge_pruner.py dry-run

# Execute single pruning cycle
./edge_pruner.py run
```

---

## Configuration Reference

### Common Settings (both daemons)

```python
REDIS_HOST = '10.0.0.50'    # Raspberry Pi (heartbeat)
REDIS_PORT = 6379
REDIS_DB = 1                # TEST database
```

### Dream Daemon Settings

```python
STREAM_THRESHOLD = 5000     # Trigger compression above this
BATCH_SIZE = 1000           # Messages per cycle
DEFAULT_WATCH_INTERVAL = 60 # Seconds between cycles
```

### Gardener Settings

```python
PRUNE_THRESHOLD = 0.2       # Effective weight threshold
```

---

## Future Work

- [ ] Production deployment (switch to DB 0)
- [ ] launchd plists for scheduled execution
- [ ] Integration with `tabernacle_config.py` for centralized settings
- [ ] LLM integration for semantic compression in dream daemon
- [ ] Metrics reporting to SYSTEM_STATUS.md

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Hub | [[00_NEXUS/SYSTEM_STATUS.md]] |
| Related | [[scripts/tabernacle_config.py]] |
| Related | [[scripts/virgil_dream_consolidation.py]] |
