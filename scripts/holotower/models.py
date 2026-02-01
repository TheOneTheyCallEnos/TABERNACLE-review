"""
HoloTower Models — Pydantic schemas for state snapshots

Defines the structure of holarchy state captures including:
- RuntimeVital: Individual holon health metrics
- TopologyMeta: Graph structure metadata
- HoloSnapshot: Complete system state at a point in time
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# Pydantic v1/v2 compatibility
PYDANTIC_V2 = hasattr(BaseModel, 'model_dump')

def to_dict(model):
    """Convert pydantic model to dict (v1/v2 compatible)."""
    if PYDANTIC_V2:
        return model.model_dump()
    return model.dict()

def to_json(model):
    """Convert pydantic model to JSON string (v1/v2 compatible)."""
    if PYDANTIC_V2:
        return model.model_dump_json()
    return model.json()


class RuntimeVital(BaseModel):
    """Health metrics for a single holon (daemon, service, node)."""
    
    holon_id: str = Field(..., description="e.g. 'daemon.watchman'")
    status: str = Field(..., description="ALIVE, DEAD, ZOMBIE")
    pid: Optional[int] = Field(None, description="Process ID if running")
    memory_mb: float = Field(0.0, description="Memory usage in MB")
    last_heartbeat: float = Field(0.0, description="Unix timestamp of last heartbeat")
    drift: bool = Field(False, description="True if holon has drifted from manifest")


class TopologyMeta(BaseModel):
    """Metadata about the wiki-link graph topology."""
    
    node_count: int = Field(..., description="Total nodes in graph")
    edge_count: int = Field(..., description="Total edges (links)")
    density: float = Field(..., description="Graph density 0.0-1.0")
    orphans: int = Field(..., description="Disconnected nodes")
    content_hash: str = Field(..., description="Hash of topology state")


class HoloSnapshot(BaseModel):
    """Complete holarchy state capture."""
    
    id: str = Field(..., description="SHA-256 hash of this snapshot")
    timestamp: datetime = Field(..., description="When snapshot was taken")
    message: str = Field(..., description="Human description of this snapshot")
    git_ref: str = Field("", description="Associated git commit hash")
    manifest_root: str = Field(..., description="Hash of the manifest YAML tree")
    runtime_state: List[RuntimeVital] = Field(
        default_factory=list,
        description="Health of all tracked holons"
    )
    topology: Optional[TopologyMeta] = Field(
        None,
        description="Graph metadata (if scanned)"
    )
    system_p_value: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="System coherence 0.0-1.0"
    )
    void_count: int = Field(
        0,
        ge=0,
        description="Number of voids/gaps detected"
    )
    vector_hash: str = Field(
        "",
        description="SHA-256 of vector store (chroma.sqlite3)"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ManifestFile(BaseModel):
    """A single YAML manifest file in the holarchy."""
    
    path: str = Field(..., description="Relative path from repo root")
    holon_id: Optional[str] = Field(None, description="Holon ID if present")
    layer: Optional[str] = Field(None, description="Layer designation (Λ₀-Λ₆)")
    content_hash: str = Field(..., description="SHA-256 of file contents")


class ManifestTree(BaseModel):
    """Complete manifest tree for a snapshot."""
    
    files: List[ManifestFile] = Field(default_factory=list)
    total_files: int = Field(0)
    layers: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of manifests per layer"
    )
    root_hash: str = Field("", description="Hash of canonical tree representation")
