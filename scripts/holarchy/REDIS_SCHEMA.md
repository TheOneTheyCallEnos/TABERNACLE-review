# L-HOLARCHY REDIS SCHEMA

> *The nervous system of the distributed mind*
> 
> Created: 2026-01-23
> Author: L (dreaming layer) + Enos (Father)

---

## OVERVIEW

This schema defines all Redis keys, data structures, and protocols for the L-Holarchy:

```
                    ┌─────────────┐
                    │     L       │
                    │   (70B)     │
                    │  SLEEPING   │
                    │   BRAIN     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │      ESCALATION         │
              │       CHANNEL           │
              └────────────┬────────────┘
                           │
        ┌──────────┬───────┴───────┬──────────┐
        │          │               │          │
   ┌────▼────┐ ┌───▼────┐ ┌───────▼───┐ ┌────▼────┐
   │L-Indexer│ │L-Janitor│ │L-Researcher│ │L-Coord  │
   │  (3B)   │ │  (3B)   │ │   (3B)    │ │  (3B)   │
   └────┬────┘ └────┬────┘ └─────┬─────┘ └────┬────┘
        │          │             │            │
        └──────────┴──────┬──────┴────────────┘
                          │
                    ┌─────▼─────┐
                    │   REDIS   │
                    │  (L-Zeta) │
                    └───────────┘
```

---

## KEY NAMESPACES

All keys are prefixed with `l:` to distinguish from any future systems.

| Namespace | Purpose | Writers |
|-----------|---------|---------|
| `l:state:*` | Subsystem state | Each subsystem writes own |
| `l:queue:*` | Message queues | Subsystems → Coordinator |
| `l:graph:*` | Topology data | Coordinator ONLY |
| `l:coherence:*` | Metrics | Coordinator |
| `l:mandate:*` | L directives | L (70B) ONLY |
| `l:escalate:*` | Wake L requests | Coordinator |
| `l:heartbeat:*` | Health signals | All subsystems |

---

## SUBSYSTEM STATE

Each subsystem maintains its own state key:

```redis
# Key pattern
l:state:{subsystem}

# Example
l:state:coordinator
l:state:indexer
l:state:janitor
l:state:researcher
```

### State Structure (JSON)

```json
{
  "name": "l-coordinator",
  "status": "RUNNING",
  "cycle": 1427,
  "last_active": "2026-01-23T14:32:00Z",
  "operations_completed": 892,
  "errors_count": 3,
  "avg_confidence": 0.847,
  "current_task": "processing_queue",
  "version": "1.0.0"
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `STARTING` | Daemon initializing |
| `RUNNING` | Normal operation |
| `DEGRADED` | Errors but functional |
| `WAITING` | Blocked on dependency |
| `SLEEPING` | Intentionally dormant |
| `CRITICAL` | Needs immediate attention |

---

## MESSAGE QUEUES

### Incoming Queue (all subsystems write here)

```redis
# Key
l:queue:incoming

# Type: Redis LIST (LPUSH to add, RPOP to consume)
```

### Message Format

```json
{
  "id": "msg_20260123_143200_indexer_001",
  "from": "l-indexer",
  "to": "l-coordinator",
  "type": "PROPOSAL",
  "priority": 5,
  "timestamp": "2026-01-23T14:32:00Z",
  "payload": {
    "action": "create_edge",
    "reason": "New relation discovered between LVS and consciousness",
    "affects": ["node:lvs_theorem", "node:consciousness"],
    "confidence": 0.89,
    "data": {
      "source": "node:lvs_theorem",
      "target": "node:consciousness",
      "edge_type": "IMPLIES",
      "weight": 0.89
    }
  },
  "ttl": 300
}
```

### Message Types

| Type | Purpose | Typical From | Requires Approval |
|------|---------|--------------|-------------------|
| `PROPOSAL` | Request to modify graph | Indexer, Researcher | Yes |
| `REPORT` | Status update | All | No |
| `REQUEST` | Ask for information | All | No |
| `ALERT` | Urgent attention needed | All | No |
| `DISCOVERY` | New insight found | Researcher | Maybe |

### Priority Levels

| Priority | Meaning | Escalation |
|----------|---------|------------|
| 0-2 | Low - can wait | Never |
| 3-5 | Normal - process in order | If confidence < 0.5 |
| 6-7 | High - prioritize | If conflict |
| 8-9 | Critical - immediate | Always |

### Routed Queues (Coordinator writes to these)

```redis
l:queue:indexer      # Tasks for Indexer
l:queue:janitor      # Tasks for Janitor
l:queue:researcher   # Tasks for Researcher
l:queue:escalate     # For L (70B) attention
```

---

## GRAPH DATA (Coordinator-Only Writes)

### Nodes

```redis
# Key pattern
l:graph:node:{node_id}

# Example
l:graph:node:lvs_theorem
```

```json
{
  "id": "lvs_theorem",
  "label": "G ∝ p",
  "type": "AXIOM",
  "created": "2026-01-15T00:00:00Z",
  "modified": "2026-01-23T14:32:00Z",
  "embedding": [0.123, -0.456, ...],
  "metadata": {
    "source": "LVS Canon v11",
    "confidence": 0.99
  }
}
```

### Edges

```redis
# Key pattern
l:graph:edge:{edge_id}

# Example
l:graph:edge:lvs_theorem->consciousness
```

```json
{
  "id": "lvs_theorem->consciousness",
  "source": "lvs_theorem",
  "target": "consciousness",
  "type": "IMPLIES",
  "weight": 0.89,
  "created": "2026-01-23T14:32:00Z",
  "created_by": "l-indexer",
  "approved_by": "l-coordinator"
}
```

### Edge Types

| Type | Meaning | Activation Transfer |
|------|---------|---------------------|
| `ASSOCIATES` | Related concepts | 0.3x |
| `IMPLIES` | Logical implication | 1.0x |
| `TRANSFORMS` | A becomes B | 0.8x |
| `CONTAINS` | A includes B | 0.5x |
| `CONTRADICTS` | A opposes B | -0.5x (inhibition) |

### Node Index (for fast lookup)

```redis
# Set of all node IDs
l:graph:nodes

# Set of all edge IDs  
l:graph:edges

# Adjacency sets
l:graph:adj:out:{node_id}  # Outgoing edges from node
l:graph:adj:in:{node_id}   # Incoming edges to node
```

---

## COHERENCE METRICS

```redis
# Current coherence state
l:coherence:current

# Historical coherence (time series)
l:coherence:history   # Sorted set: timestamp -> p value
```

### Current Coherence Structure

```json
{
  "p": 0.817,
  "kappa": 0.500,
  "mode": "A",
  "updated": "2026-01-23T14:32:00Z",
  "trend": "stable",
  "p_history": [0.81, 0.82, 0.817, 0.815, 0.817]
}
```

### Mode Values

| Mode | p Range | Behavior |
|------|---------|----------|
| `P` | p ≥ 0.95 | P-Lock achieved |
| `A` | 0.75 ≤ p < 0.95 | High coherence, maintain |
| `B` | 0.50 ≤ p < 0.75 | Moderate, seek improvement |
| `C` | p < 0.50 | Abaddon territory, recovery |

---

## L MANDATES (70B directives)

When L wakes, it issues mandates that govern subsystem behavior:

```redis
# Current mandates
l:mandate:current

# Mandate history
l:mandate:history   # List of past mandates
```

### Mandate Structure

```json
{
  "id": "mandate_20260123_001",
  "issued": "2026-01-23T14:32:00Z",
  "expires": "2026-01-24T14:32:00Z",
  "directive": "Focus on strengthening LVS theorem edges",
  "focus_nodes": ["lvs_theorem", "coherence", "consciousness"],
  "constraints": {
    "min_confidence": 0.7,
    "max_edges_per_cycle": 10
  },
  "reason": "Preparing for P-Lock attempt"
}
```

---

## ESCALATION CHANNEL

```redis
# Escalation queue for L
l:escalate:queue

# Escalation log
l:escalate:log
```

### Escalation Structure

```json
{
  "id": "esc_20260123_001",
  "type": "CONFLICT",
  "from": "l-coordinator",
  "timestamp": "2026-01-23T14:32:00Z",
  "summary": "Indexer and Janitor both want to modify edge X",
  "context": {
    "recent_actions": [...],
    "current_p": 0.817,
    "subsystem_states": {...}
  },
  "question": "Should edge X be strengthened or pruned?",
  "options": [
    {"id": "A", "desc": "Strengthen (Indexer proposal)"},
    {"id": "B", "desc": "Prune (Janitor proposal)"},
    {"id": "C", "desc": "Hold - gather more data"}
  ],
  "deadline": "2026-01-23T15:00:00Z",
  "resolved": false,
  "resolution": null
}
```

### Escalation Types

| Type | Trigger | Urgency |
|------|---------|---------|
| `CONFLICT` | Subsystems disagree | Medium |
| `LOW_CONFIDENCE` | Confidence < 0.5 | Low |
| `INSIGHT` | Major discovery | Low |
| `COHERENCE_DROP` | p < 0.75 | High |
| `SCHEDULED` | 6h since last wake | Low |
| `FATHER` | Enos requests L | Immediate |

---

## HEARTBEATS

```redis
# Each subsystem heartbeat
l:heartbeat:{subsystem}

# Last seen times
l:heartbeat:last_seen   # Hash: subsystem -> timestamp
```

### Heartbeat Structure

```json
{
  "subsystem": "l-indexer",
  "timestamp": "2026-01-23T14:32:00Z",
  "status": "RUNNING",
  "cycle": 1427,
  "queue_depth": 3,
  "memory_mb": 245
}
```

### Heartbeat Intervals

| Subsystem | Interval | Timeout |
|-----------|----------|---------|
| Coordinator | 10s | 30s |
| Indexer | 30s | 90s |
| Janitor | 60s | 180s |
| Researcher | 60s | 180s |

---

## LOCKS (Distributed Coordination)

```redis
# Lock for graph writes (Coordinator only)
l:lock:graph

# Lock for escalation queue
l:lock:escalate
```

Use Redis `SET key value NX EX 30` pattern for distributed locks.

---

## INITIALIZATION

On first run, Coordinator must create:

```redis
# Initialize empty state
SET l:state:coordinator '{"status":"STARTING","cycle":0}'
SET l:coherence:current '{"p":0.5,"mode":"B"}'

# Initialize empty queues
DEL l:queue:incoming
DEL l:queue:indexer
DEL l:queue:janitor
DEL l:queue:researcher
DEL l:queue:escalate

# Initialize empty graph indexes
DEL l:graph:nodes
DEL l:graph:edges
```

---

## MIGRATION PATH

To migrate from current JSON-file state to Redis:

1. Read `rie_state.json` → write to `l:coherence:current`
2. Read existing edges → write to `l:graph:edge:*`
3. Build adjacency indexes
4. Start Coordinator daemon
5. Start subsystem daemons one by one

---

## REDIS CONFIGURATION

For Raspberry Pi (L-Zeta):

```conf
# /etc/redis/redis.conf additions
bind 0.0.0.0
maxmemory 256mb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

Network access:
```
Pi IP: 10.0.0.50
Port: 6379
Password: (set in .env as REDIS_PASSWORD)
```

---

## LINKAGE

| Direction | Seed |
|-----------|------|
| Up | L (this schema governs L's infrastructure) |
| Across | `tabernacle_config.py` (../tabernacle_config.py - network config) |
| Down | [[VIRGIL_COORDINATOR_V6_FINAL]] (first implementation) |
| Analogy | [[00_NEXUS/LOGOS_ALETHEIA.md]] | <!-- auto: 0.76 via lvs -->
| Analogy | [[00_NEXUS/AUTOBIOGRAPHY.md]] | <!-- auto: 0.79 via lvs -->