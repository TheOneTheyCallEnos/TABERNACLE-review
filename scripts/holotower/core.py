"""
HoloTower Core - Content-Addressable Storage primitives

Implements:
    - SHA-256 hashing for content addressing
    - Zstd compression for efficient storage
    - Sharded blob storage (2-char prefix directories)
    - Blob read/write operations
    - Audit trail logging
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import zstandard as zstd


# Compression context (reusable for efficiency)
_COMPRESSOR = zstd.ZstdCompressor(level=3)
_DECOMPRESSOR = zstd.ZstdDecompressor()


def get_holotower_root() -> Path:
    """
    Find the holotower/ directory.
    
    Searches from current working directory upward, then falls back
    to the TABERNACLE root location.
    
    Returns:
        Path to holotower/ directory
        
    Raises:
        FileNotFoundError: If holotower/ cannot be found
    """
    # First, check common locations
    candidates = [
        Path.cwd() / "holotower",
        Path(__file__).parent.parent.parent / "holotower",  # scripts/holotower/core.py -> repo root
    ]
    
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    
    # Walk up from cwd
    current = Path.cwd()
    while current != current.parent:
        holotower_path = current / "holotower"
        if holotower_path.is_dir():
            return holotower_path
        current = current.parent
    
    raise FileNotFoundError(
        "Could not find holotower/ directory. "
        "Run 'holotower init' to create one."
    )


def hash_content(data: bytes) -> str:
    """
    Compute SHA-256 hash of content.
    
    Args:
        data: Raw bytes to hash
        
    Returns:
        Lowercase hexadecimal SHA-256 hash string (64 chars)
    """
    return hashlib.sha256(data).hexdigest()


def compress(data: bytes) -> bytes:
    """
    Compress data using Zstandard.
    
    Args:
        data: Raw bytes to compress
        
    Returns:
        Zstd-compressed bytes
    """
    return _COMPRESSOR.compress(data)


def decompress(data: bytes) -> bytes:
    """
    Decompress Zstandard-compressed data.
    
    Args:
        data: Zstd-compressed bytes
        
    Returns:
        Original uncompressed bytes
    """
    return _DECOMPRESSOR.decompress(data)


def write_blob(content: bytes, root: Optional[Path] = None) -> str:
    """
    Write content to the Content-Addressable Store.
    
    Content is hashed, compressed, and stored in a sharded directory
    structure: objects/<first 2 chars of hash>/<full hash>
    
    Args:
        content: Raw bytes to store
        root: Optional path to holotower/ (auto-detected if None)
        
    Returns:
        SHA-256 hash of the original content (serves as blob ID)
    """
    if root is None:
        root = get_holotower_root()
    
    # Hash original content
    sha = hash_content(content)
    
    # Shard by first 2 characters
    shard_dir = root / "objects" / sha[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    blob_path = shard_dir / sha
    
    # Only write if not already present (CAS is immutable)
    if not blob_path.exists():
        compressed = compress(content)
        blob_path.write_bytes(compressed)
    
    return sha


def read_blob(sha: str, root: Optional[Path] = None) -> bytes:
    """
    Read content from the Content-Addressable Store.
    
    Args:
        sha: SHA-256 hash of the content to retrieve
        root: Optional path to holotower/ (auto-detected if None)
        
    Returns:
        Original uncompressed content bytes
        
    Raises:
        FileNotFoundError: If blob does not exist
        ValueError: If content hash doesn't match (corruption detected)
    """
    if root is None:
        root = get_holotower_root()
    
    # Locate blob in sharded structure
    blob_path = root / "objects" / sha[:2] / sha
    
    if not blob_path.exists():
        raise FileNotFoundError(f"Blob not found: {sha}")
    
    # Decompress
    compressed = blob_path.read_bytes()
    content = decompress(compressed)
    
    # Verify integrity
    actual_hash = hash_content(content)
    if actual_hash != sha:
        raise ValueError(
            f"Content integrity check failed. "
            f"Expected {sha}, got {actual_hash}"
        )
    
    return content


# =============================================================================
# AUDIT TRAIL
# =============================================================================

def get_ledger_path() -> Path:
    """Get path to ledger.db."""
    return get_holotower_root() / "ledger.db"


def ensure_audit_schema() -> None:
    """
    Ensure audit_trail table exists in ledger.db.
    
    Creates the table if it doesn't exist. Safe to call multiple times.
    """
    ledger = get_ledger_path()
    conn = sqlite3.connect(str(ledger))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                target_holon TEXT,
                hash_before TEXT,
                hash_after TEXT,
                metadata JSON
            )
        """)
        # Index for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
            ON audit_trail(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_actor 
            ON audit_trail(actor)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_action 
            ON audit_trail(action)
        """)
        conn.commit()
    finally:
        conn.close()


def log_audit(
    actor: str,
    action: str,
    target: Optional[str] = None,
    hash_before: Optional[str] = None,
    hash_after: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Log an action to the audit trail.
    
    Args:
        actor: Who/what performed the action. Examples:
               - "Enos" (human)
               - "Daemon:Vacuum" (automated vacuum process)
               - "Daemon:Enzyme" (digestive enzyme)
               - "Agent:Virgil" (AI agent)
               - "System:HoloTower" (system process)
        action: What was done. Examples:
                - "INGEST" (vacuum moved file to inbox)
                - "SOLIDIFY" (enzyme archived file)
                - "MITOSIS" (enzyme split file)
                - "SNAPSHOT" (holotower captured state)
                - "MODIFY" (file content changed)
        target: The holon or file affected (optional)
        hash_before: Content hash before change (optional)
        hash_after: Content hash after change (optional)
        metadata: Additional structured data (optional, stored as JSON)
        
    Returns:
        Audit entry ID (UUID)
    """
    ensure_audit_schema()
    
    audit_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata) if metadata else None
    
    ledger = get_ledger_path()
    conn = sqlite3.connect(str(ledger))
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_trail 
            (id, timestamp, actor, action, target_holon, hash_before, hash_after, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            audit_id,
            timestamp,
            actor,
            action,
            target,
            hash_before,
            hash_after,
            metadata_json
        ))
        conn.commit()
    finally:
        conn.close()
    
    return audit_id


def get_audit_trail(
    limit: int = 100,
    actor: Optional[str] = None,
    action: Optional[str] = None,
    target: Optional[str] = None,
    since: Optional[datetime] = None
) -> List[Dict]:
    """
    Query audit trail with optional filters.
    
    Args:
        limit: Maximum entries to return (default 100)
        actor: Filter by actor (exact match)
        action: Filter by action type (exact match)
        target: Filter by target holon (substring match)
        since: Only entries after this timestamp
        
    Returns:
        List of audit entries as dicts, newest first
    """
    ensure_audit_schema()
    
    ledger = get_ledger_path()
    conn = sqlite3.connect(str(ledger))
    try:
        cursor = conn.cursor()
        
        # Build query dynamically
        query = "SELECT id, timestamp, actor, action, target_holon, hash_before, hash_after, metadata FROM audit_trail WHERE 1=1"
        params = []
        
        if actor:
            query += " AND actor = ?"
            params.append(actor)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        if target:
            query += " AND target_holon LIKE ?"
            params.append(f"%{target}%")
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            entry = {
                "id": row[0],
                "timestamp": row[1],
                "actor": row[2],
                "action": row[3],
                "target_holon": row[4],
                "hash_before": row[5],
                "hash_after": row[6],
                "metadata": json.loads(row[7]) if row[7] else None
            }
            results.append(entry)
        
        return results
    finally:
        conn.close()
