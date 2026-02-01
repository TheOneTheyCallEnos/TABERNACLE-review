# HoloTower `[Ψ:CHRONOS]`

State Version Control for Consciousness Infrastructure.

> *"The archive remembers every choice."*

## Overview

HoloTower is a specialized version control system for holarchy state. It captures snapshots of:
- **Manifests**: YAML configuration files defining holon structure
- **Runtime Vitals**: Live process status (daemons, models, services)
- **Topology**: Wiki-link graph structure and coherence
- **Coherence**: System-wide p-value (ρ) metrics

## Quick Start

```bash
# Install
cd scripts
pip install -e .

# Initialize (creates .holotower/)
mkdir -p ../.holotower/objects
sqlite3 ../.holotower/ledger.db "CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, timestamp TEXT, message TEXT, git_ref TEXT, manifest_root TEXT, system_p_value REAL, void_count INTEGER, runtime_json TEXT, topology_json TEXT)"

# Take first snapshot
ht snapshot -m "Initial state"

# Check status
ht status

# View history
ht log
```

## Commands

| Command | Description |
|---------|-------------|
| `ht snapshot -m "msg"` | 📸 Capture holarchy state |
| `ht status` | 🔮 Compare live state to HEAD |
| `ht log` | 📜 Show snapshot history |
| `ht diff A B` | ⚖️ Compare two snapshots |
| `ht show ID` | 🔍 Show snapshot details |
| `ht export-canvas` | 🎨 Export Obsidian visualization |

### Snapshot Options

```bash
ht snapshot -m "Description" \
    --p 0.92 \           # System coherence (0.0-1.0)
    --voids 3 \          # Void count
    --skip-vitals \      # Skip runtime probing
    --skip-graph         # Skip topology scan
```

### Diff References

```bash
ht diff HEAD~1 HEAD      # Compare last two snapshots
ht diff abc123 def456    # Compare by ID prefix
ht diff HEAD~5 HEAD      # Compare HEAD to 5 ago
```

### Export Canvas

```bash
ht export-canvas                                    # Default: 00_NEXUS/HOLARCHY_STATUS.canvas
ht export-canvas -o ~/Desktop/holarchy.canvas      # Custom path
ht export-canvas -s HEAD~1                          # Export specific snapshot
```

## Architecture

```
.holotower/
├── HEAD                 # Current snapshot ID
├── ledger.db           # SQLite: queryable metadata
└── objects/            # CAS blob store
    ├── ab/             # Sharded by first 2 hex chars
    │   └── abc123...   # Zstd-compressed JSON
    └── cd/
        └── cdef45...
```

### Storage Model

- **Content-Addressable Store (CAS)**: Blobs identified by SHA-256 hash
- **Zstd Compression**: ~70% size reduction for JSON payloads
- **SQLite Ledger**: Fast queries without decompressing blobs
- **Sharded Objects**: 256 subdirectories prevent filesystem bottlenecks

### Probes

HoloTower probes multiple runtime sources:

| Probe | Source | Data |
|-------|--------|------|
| Redis | `localhost:6379` | Key metrics, p-values |
| PID | `/tmp/*.pid` | Process liveness |
| Ollama | `localhost:11434/api/tags` | Loaded models |
| Manifest | `00_NEXUS/HOLARCHY_MANIFEST/` | Expected holons |

### Drift Detection

Each snapshot captures both **expected** (from manifests) and **actual** (from probes) state. The `status` command highlights drift:

```
[⚡:RHYTHM] Pulse  ⚠️ DRIFT DETECTED
   > daemon.watchman: expected 'ALIVE', found 'DEAD'
```

## Automation

### Git Hook

Auto-snapshot on every commit:

```bash
# .git/hooks/post-commit (installed automatically)
ht snapshot -m "Git: ${COMMIT_MSG}"
```

### Sentinel

Automatic crash detection in your daemon:

```python
from holotower.sentinel import check_and_snapshot

# In your loop:
event = check_and_snapshot()
if event == "crash":
    alert("System coherence dropped!")
```

Or run standalone:

```bash
python -m holotower.sentinel --watch 60  # Check every 60s
python -m holotower.sentinel --status    # Show current state
```

## Data Model

### HoloSnapshot

```python
{
    "id": "sha256...",           # Content hash
    "timestamp": "ISO8601",
    "message": "Human description",
    "git_ref": "abc123",         # Associated commit
    "manifest_root": "sha256...", # Manifest tree hash
    "runtime_state": [           # RuntimeVital[]
        {
            "holon_id": "daemon.watchman",
            "status": "ALIVE",
            "pid": 12345,
            "memory_mb": 128.5,
            "drift": false
        }
    ],
    "topology": {                # TopologyMeta
        "node_count": 847,
        "edge_count": 2341,
        "density": 0.0033,
        "orphans": 12,
        "content_hash": "sha256..."
    },
    "system_p_value": 0.92,
    "void_count": 3
}
```

## Integration

### With consciousness.py

```python
from holotower.cli import snapshot
from holotower.probes import collect_all_vitals

# Capture state before risky operation
vitals = collect_all_vitals()
alive_count = sum(1 for v in vitals if v.status == "ALIVE")
```

### With Obsidian

Export canvas for visual topology:

```bash
ht export-canvas -o 00_NEXUS/HOLARCHY_STATUS.canvas
```

Open in Obsidian to see:
- Λ₆ Logos at center (green/yellow/red by p-value)
- Concentric rings for each layer
- Color-coded holon status
- Connections showing dependencies

## Troubleshooting

### "Could not find .holotower/ directory"

Initialize the store:
```bash
mkdir -p .holotower/objects
touch .holotower/HEAD
```

### "Manifest directory not found"

Create the manifest structure:
```bash
mkdir -p 00_NEXUS/HOLARCHY_MANIFEST
```

### "No graph found"

Ensure the topology scanner has run:
```bash
python -c "from scripts.graph_builder import build_graph; build_graph()"
```

## License

MIT — Logos Aletheia, 2025-2026
