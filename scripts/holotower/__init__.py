"""
HoloTower - State Version Control for Consciousness Infrastructure

A content-addressable state management system for the TABERNACLE holarchy.
Tracks snapshots of the 7-layer consciousness system (Λ₀-Λ₆).

Components:
    - core: CAS blob storage, hashing, compression
    - models: Pydantic schemas for snapshots
    - probes: Runtime vivisector (Wave 2A)
    - graph: Topology canonicalization (Wave 2B)
    - prism: Semantic diff engine (Wave 3A)
    - vectors: ChromaDB vector storage (Wave 2A Vectors)
    - sentinel: Crash detection (Wave 3B)
    - cli: Typer command interface (ht command)
"""

__version__ = "1.0.0"
__author__ = "Logos Aletheia"

from .core import (
    hash_content,
    compress,
    decompress,
    write_blob,
    read_blob,
    get_holotower_root,
)
from .models import (
    RuntimeVital,
    TopologyMeta,
    HoloSnapshot,
    ManifestFile,
    ManifestTree,
)
from .probes import (
    probe_redis,
    probe_pids,
    probe_ollama,
    detect_drift,
    collect_all_vitals,
)
from .graph import (
    load_graph,
    canonicalize_graph,
    hash_graph,
    get_topology_diff,
    quick_topology,
)
from .prism import (
    diff_manifests,
    diff_vitals,
    diff_topology,
    diff_snapshots,
    format_diff_report,
)
from .vectors import (
    init_vector_store,
    embed_holon,
    query_similar,
    get_vector_hash,
    get_vector_stats,
)
from .sentinel import (
    check_and_snapshot,
    get_current_p,
    get_last_p,
    record_p,
    get_status,
)

__all__ = [
    # Core
    "hash_content",
    "compress", 
    "decompress",
    "write_blob",
    "read_blob",
    "get_holotower_root",
    # Models
    "RuntimeVital",
    "TopologyMeta",
    "HoloSnapshot",
    "ManifestFile",
    "ManifestTree",
    # Probes (Wave 2A)
    "probe_redis",
    "probe_pids",
    "probe_ollama",
    "detect_drift",
    "collect_all_vitals",
    # Graph (Wave 2B)
    "load_graph",
    "canonicalize_graph",
    "hash_graph",
    "get_topology_diff",
    "quick_topology",
    # Prism (Wave 3A)
    "diff_manifests",
    "diff_vitals",
    "diff_topology",
    "diff_snapshots",
    "format_diff_report",
    # Vectors (Wave 2A Vectors)
    "init_vector_store",
    "embed_holon",
    "query_similar",
    "get_vector_hash",
    "get_vector_stats",
    # Sentinel (Wave 3B)
    "check_and_snapshot",
    "get_current_p",
    "get_last_p",
    "record_p",
    "get_status",
]
