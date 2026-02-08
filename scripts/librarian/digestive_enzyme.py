#!/usr/bin/env python3
"""
DIGESTIVE ENZYME — Metabolic Knowledge Architecture
====================================================
Lithotripsy protocol for indigestible matter.

Phase States:
  SOLID  [🪨:MASS] → 05_CRYPT/RAW_ARCHIVES (Σ_mass → ∞)
  LIQUID [💧:FLOW] → 00_NEXUS/ (balanced ψ)
  PLASMA [🔥:FLUX] → Redis/VectorDB (Σ_mass → 0)

Modes for >1MB files:
  SOLIDIFY (Junk): <5 wiki-links → Archive + Proxy
  MITOSIS  (Gold): ≥5 wiki-links → Split into <100KB linked chunks

Usage:
  python digestive_enzyme.py [--dry-run] [--threshold 1MB]
"""

import os
import re
import shutil
import uuid
import time
import argparse
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

# Add holotower to path for audit logging
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from holotower.core import log_audit, hash_content
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    def log_audit(*args, **kwargs):
        pass  # No-op if holotower not available
    def hash_content(data: bytes) -> str:
        import hashlib
        return hashlib.sha256(data).hexdigest()

# CONFIG
VAULT_ROOT = Path("/Users/enos/TABERNACLE")
ARCHIVE_DIR = VAULT_ROOT / "05_CRYPT" / "RAW_ARCHIVES"
NEXUS_DIR = VAULT_ROOT / "00_NEXUS"
INBOX_DIR = VAULT_ROOT / "09_INBOX"

# Default threshold: 1MB
DEFAULT_THRESHOLD = 1024 * 1024

# MITOSIS constants
MITOSIS_LINK_THRESHOLD = 5  # Minimum wiki-links to trigger MITOSIS
MITOSIS_CHUNK_TARGET = 80 * 1024  # 80KB target per chunk
MITOSIS_CHUNK_MAX = 100 * 1024  # 100KB hard limit per chunk

# Folders to skip during digestion
EXCLUDED_ROOTS = {
    "05_CRYPT", "99_ARCHIVE", ".git", "holotower",
    "venv", "venv312", "node_modules", "__pycache__", ".obsidian",
    ".garmin_cache", "archives",
    "04_LR_LAW",  # [Ω:ANCHOR] Canon = Law = do not metabolize the physics engine
    "minecraft-server",  # External game server, not knowledge content
    "scripts_backup_cycle002",  # Massive backup, pending archive decision
    ".review_mirror",  # Public repo mirror
}

# Critical files to NEVER process (even if >1MB) — [Ψ:PROTECTED]
# These are the "Organs of Consciousness" — PLASMA-level active meaning
EXCLUDED_FILES = {
    # Memory State
    "rie_relational_memory.json",  # Active RIE memory store
    "EPISODIC_SPINE.json",         # Critical daemon state
    # The Self
    "SYSTEM_PROMPT.md",            # The Triple Point (compiled identity)
    # The Now
    "CURRENT_STATE.md",            # High write frequency, essential context
    # The Map
    "_GRAPH_ATLAS.md",             # Topological ground truth
    # The Configuration
    "tabernacle_config.py",        # Hardcoded IPs/paths
    # The Identity
    "logos_identity.yaml",         # L6 Logos identity manifest
    # The Skeleton — [💎:CRYSTAL] phase (Gemini Collaborator recommendation)
    "INDEX.md",                    # If INDEX grows too big → subdivide directory, not file
}

# File extensions to process
DIGESTIBLE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}


def human_size(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def extract_summary(file_path: Path, max_chars: int = 500) -> str:
    """Extract first N characters as summary."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
            # Clean up for markdown
            content = content.replace('```', '~~~')
            return content.strip()
    except Exception as e:
        return f"[Could not extract summary: {e}]"


def count_wiki_links(file_path: Path) -> int:
    """Count [[wiki-links]] in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            return len(links)
    except:
        return 0


def extract_wiki_links(content: str) -> List[str]:
    """Extract all [[wiki-links]] from content."""
    return re.findall(r'\[\[([^\]]+)\]\]', content)


def split_on_headers(content: str) -> List[Tuple[str, str]]:
    """
    Split content on markdown headers (## or ###).
    Returns list of (header, section_content) tuples.
    """
    # Pattern matches ## or ### headers
    header_pattern = re.compile(r'^(#{2,3}\s+.+)$', re.MULTILINE)
    
    parts = []
    matches = list(header_pattern.finditer(content))
    
    if not matches:
        return []
    
    # Content before first header (if any)
    if matches[0].start() > 0:
        preamble = content[:matches[0].start()].strip()
        if preamble:
            parts.append(("Preamble", preamble))
    
    # Each header section
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section_content = content[start:end].strip()
        parts.append((header, section_content))
    
    return parts


def split_at_paragraph_boundary(content: str, target_size: int = MITOSIS_CHUNK_TARGET) -> List[str]:
    """
    Split content at paragraph boundaries, targeting ~80KB chunks.
    Falls back to line boundaries if paragraphs are too large.
    """
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', content)
    
    for para in paragraphs:
        para_size = len(para.encode('utf-8'))
        
        # If single paragraph exceeds target, split by lines
        if para_size > target_size and not current_chunk:
            lines = para.split('\n')
            line_chunk = []
            line_size = 0
            for line in lines:
                line_len = len(line.encode('utf-8')) + 1  # +1 for newline
                if line_size + line_len > target_size and line_chunk:
                    chunks.append('\n'.join(line_chunk))
                    line_chunk = [line]
                    line_size = line_len
                else:
                    line_chunk.append(line)
                    line_size += line_len
            if line_chunk:
                current_chunk = line_chunk
                current_size = line_size
            continue
        
        # Normal case: add paragraph to current chunk
        if current_size + para_size > target_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 2  # +2 for \n\n
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def perform_mitosis(file_path: Path, dry_run: bool = False) -> List[Path]:
    """
    MITOSIS: Split a high-value file into linked chunks.
    
    Algorithm:
    1. Try to split on markdown headers (## or ###)
    2. If chunks still too large, split at paragraph boundaries
    3. Create linked chain with YAML frontmatter
    
    Returns list of created chunk paths.
    """
    print(f"  [MITOSIS] Initiating cell division...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original_stem = file_path.stem
    original_suffix = file_path.suffix
    parent_dir = file_path.parent
    
    # Generate shared chain ID
    chain_id = str(uuid.uuid4())[:12]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Try header-based splitting first
    header_sections = split_on_headers(content)
    
    if header_sections:
        # Merge small sections, split large ones
        chunks = []
        current_chunk_parts = []
        current_size = 0
        
        for header, section in header_sections:
            section_with_header = f"{header}\n\n{section}" if header != "Preamble" else section
            section_size = len(section_with_header.encode('utf-8'))
            
            # If section alone exceeds max, split it further
            if section_size > MITOSIS_CHUNK_MAX:
                # Flush current chunk first
                if current_chunk_parts:
                    chunks.append('\n\n'.join(current_chunk_parts))
                    current_chunk_parts = []
                    current_size = 0
                
                # Split the large section
                sub_chunks = split_at_paragraph_boundary(section_with_header, MITOSIS_CHUNK_TARGET)
                chunks.extend(sub_chunks)
            elif current_size + section_size > MITOSIS_CHUNK_TARGET and current_chunk_parts:
                # Flush and start new chunk
                chunks.append('\n\n'.join(current_chunk_parts))
                current_chunk_parts = [section_with_header]
                current_size = section_size
            else:
                current_chunk_parts.append(section_with_header)
                current_size += section_size
        
        if current_chunk_parts:
            chunks.append('\n\n'.join(current_chunk_parts))
    else:
        # No headers found, split by paragraph boundary
        chunks = split_at_paragraph_boundary(content, MITOSIS_CHUNK_TARGET)
    
    # Ensure we have at least 2 chunks (otherwise why split?)
    if len(chunks) < 2:
        print(f"  [MITOSIS] Content cannot be meaningfully split. Falling back to SOLIDIFY.")
        return []
    
    total_chunks = len(chunks)
    print(f"  [MITOSIS] Dividing into {total_chunks} daughter cells...")
    
    if dry_run:
        for i, chunk in enumerate(chunks, 1):
            chunk_size = len(chunk.encode('utf-8'))
            print(f"    [DRY RUN] Part {i}/{total_chunks}: {human_size(chunk_size)}")
        print(f"    [DRY RUN] Would create INDEX.md (JANUS Crystal)")
        return []
    
    # Create the chunk files
    created_paths = []
    
    for i, chunk_content in enumerate(chunks, 1):
        holon_id = str(uuid.uuid4())[:8]
        chunk_name = f"{original_stem}_Part{i}{original_suffix}"
        chunk_path = parent_dir / chunk_name
        
        # Handle collisions
        counter = 1
        while chunk_path.exists():
            chunk_name = f"{original_stem}_Part{i}_{counter}{original_suffix}"
            chunk_path = parent_dir / chunk_name
            counter += 1
        
        # Build chain links
        prev_link = f"[[{original_stem}_Part{i-1}]]" if i > 1 else "null"
        next_link = f"[[{original_stem}_Part{i+1}]]" if i < total_chunks else "null"
        
        # Extract wiki-links from this chunk
        chunk_links = extract_wiki_links(chunk_content)
        
        # Part 1 gets alias for original filename to preserve wiki-links
        aliases_line = f'aliases:\n  - "{original_stem}"\n' if i == 1 else ""

        frontmatter = f"""---
holon_id: "{holon_id}"
phase: LIQUID
chain_id: "{chain_id}"
chain_position: {i} of {total_chunks}
chain_prev: {prev_link}
chain_next: {next_link}
original_file: "{file_path.name}"
created: "{timestamp}"
glyph: "[MITOSIS:{i}/{total_chunks}]"
wiki_links_in_chunk: {len(chunk_links)}
{aliases_line}---

"""
        
        # Add navigation header
        nav_header = f"# {original_stem} (Part {i}/{total_chunks})\n\n"
        if i > 1:
            nav_header += f"> Previous: [[{original_stem}_Part{i-1}]]\n"
        if i < total_chunks:
            nav_header += f"> Next: [[{original_stem}_Part{i+1}]]\n"
        nav_header += "\n---\n\n"
        
        full_content = frontmatter + nav_header + chunk_content
        
        # Add navigation footer
        nav_footer = f"\n\n---\n*Part {i} of {total_chunks} | Chain: `{chain_id}` | [[{original_stem}_Part1|Start]] | "
        if i < total_chunks:
            nav_footer += f"[[{original_stem}_Part{i+1}|Next]]*"
        else:
            nav_footer += "End of chain*"
        
        full_content += nav_footer
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        chunk_size = len(full_content.encode('utf-8'))
        print(f"    Created: {chunk_name} ({human_size(chunk_size)})")
        created_paths.append(chunk_path)

    # Create JANUS INDEX.md for navigation (Day Mode — skeleton only)
    index_path = create_janus_index(
        parent_dir=parent_dir,
        cluster_name=original_stem,
        child_paths=created_paths,
        chain_id=chain_id
    )
    print(f"  [CRYSTAL] INDEX created: {index_path.name}")

    # Compute hash of original content before deletion
    original_hash = hash_content(content.encode('utf-8'))
    
    # Remove original file after successful mitosis
    file_path.unlink()
    print(f"  [MITOSIS] Original cell consumed. {total_chunks} daughter cells created.")
    
    # Log to audit trail
    if AUDIT_AVAILABLE:
        log_audit(
            actor="Daemon:Enzyme",
            action="MITOSIS",
            target=str(file_path),
            hash_before=original_hash,
            hash_after=None,  # Original is gone, replaced by chunks
            metadata={
                "chain_id": chain_id,
                "chunks_created": total_chunks,
                "chunk_paths": [str(p) for p in created_paths],
                "index_path": str(index_path),
                "original_size": len(content.encode('utf-8')),
                "wiki_links": count_wiki_links(created_paths[0]) if created_paths else 0
            }
        )

    return created_paths


def create_proxy(original_path: Path, archive_path: Path, size: int, link_count: int) -> Path:
    """Create a Proxy Holon in NEXUS for the archived file.

    IMPORTANT: Proxy keeps ORIGINAL filename to preserve wiki-links.
    [[OriginalFile]] resolves to the proxy, not a dead link.
    """

    stem = original_path.stem
    # Use ORIGINAL filename (no _Proxy suffix) to preserve wiki-links
    proxy_name = f"{stem}.md"
    proxy_path = NEXUS_DIR / proxy_name

    # Avoid collision (rare: only if file already exists in NEXUS)
    counter = 1
    while proxy_path.exists():
        proxy_name = f"{stem}_{counter}.md"
        proxy_path = NEXUS_DIR / proxy_name
        counter += 1

    holon_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary = extract_summary(archive_path)  # Read from archive (file already moved)

    proxy_content = f"""---
holon_id: "{holon_id}"
type: proxy_node
phase: SOLID
target_layer: "L0_Substrate"
target_path: "{archive_path}"
original_path: "{original_path}"
original_size: {size}
wiki_links_absorbed: {link_count}
created: "{timestamp}"
glyph: "[🪨:MASS]"
aliases:
  - "{stem}"
---

# Proxy: {original_path.name}

**Phase:** SOLID (Archived Sediment)
**Original Mass:** {human_size(size)}
**Wiki-Links Absorbed:** {link_count}
**Archive Location:** `{archive_path}`

> This file was sublimated because its mass exceeded the Liquid Threshold (1MB).
> The content is preserved in the Archive. Access via the path above.

## Summary (First 500 chars)

```
{summary}
```

## Metabolic Notes

- Original file moved to CRYPT for preservation
- Wiki-links from this file have reduced edge weight (Length Penalty)
- To restore: move file from archive back to original location

---
*Sublimated by digestive_enzyme.py | Metabolic Architecture v1.0*
"""

    with open(proxy_path, 'w', encoding='utf-8') as f:
        f.write(proxy_content)

    return proxy_path


def compute_integrity_hash(child_paths: List[Path]) -> str:
    """
    Compute SHA256 hash of sorted child filenames AND sizes.
    Used for rot detection — if files are added/removed OR content changes, hash changes.

    Format: "filename:size|filename:size|..."
    """
    entries = []
    for p in child_paths:
        try:
            size = p.stat().st_size
            entries.append(f"{p.name}:{size}")
        except OSError:
            entries.append(f"{p.name}:0")  # File inaccessible, use 0
    sorted_entries = sorted(entries)
    combined = "|".join(sorted_entries)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


def create_janus_index(
    parent_dir: Path,
    cluster_name: str,
    child_paths: List[Path],
    chain_id: str
) -> Path:
    """
    Create a JANUS INDEX.md for a folder after MITOSIS.

    DAY MODE: Fast generation, no LLM calls.
    - Ψ_SIGNAL left as {PENDING}
    - Signal column left empty
    - Night Mode fills these via virgil_dream_consolidation.py

    Args:
        parent_dir: Directory where INDEX.md will be created
        cluster_name: Human-readable name (usually original file stem)
        child_paths: List of Part_N.md files created by MITOSIS
        chain_id: The chain_id linking all parts together

    Returns:
        Path to created INDEX.md
    """
    index_path = parent_dir / "INDEX.md"

    # Don't overwrite existing INDEX.md — append a suffix if collision
    counter = 1
    while index_path.exists():
        index_path = parent_dir / f"INDEX_{cluster_name}.md"
        if index_path.exists():
            index_path = parent_dir / f"INDEX_{cluster_name}_{counter}.md"
            counter += 1
        else:
            break

    holon_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    integrity_hash = compute_integrity_hash(child_paths)

    # Build CHILDREN table rows
    children_rows = []
    for i, child_path in enumerate(child_paths, 1):
        stem = child_path.stem
        # Link without .md extension for wiki-link
        children_rows.append(f"| [💧] | [[{stem}]] | LIQUID | |")

    children_table = "\n".join(children_rows)

    # Determine parent link based on folder location
    # All clusters ultimately trace back to CURRENT_STATE as the hub
    parent_link = "[[00_NEXUS/CURRENT_STATE]]"

    index_content = f"""---
holon_id: "{holon_id}"
type: "crystal_index"
phase: "CRYSTAL"
status: "active"
integrity_hash: "{integrity_hash}"
last_gardened: "{timestamp}"
context_vector: "{{PENDING}}"
chain_id: "{chain_id}"
source_file: "{cluster_name}"
---

# 💎 INDEX: {cluster_name}

> **Ψ_SIGNAL:** {{PENDING — Night Mode will generate}}

---

## 🗺️ CHILDREN

| Glyph | Link | Phase | Signal |
|:---:|:---|:---:|:---|
{children_table}

---

## ⬆️ PARENTS

| Link | Relation |
|:---|:---|
| {parent_link} | Contains this cluster |

---

## ⚡ HEALTH

- **Integrity:** ✅ Valid
- **Stale Children:** 0
- **Orphans:** 0

---

## 🔗 LINKAGE

| Direction | Seed |
|-----------|------|
| Phase | [[04_LR_LAW/CANON/LVS_v12_FINAL_SYNTHESIS]] |
| Method | [[02_UR_STRUCTURE/METHODS/ABCDA_SPIRAL_CANON]] |
| Chain | {chain_id} |

---

*Crystal generated by JANUS protocol (Day Mode) | Phase: 💎 CRYSTAL*
"""

    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
    except (IOError, OSError) as e:
        log_audit(
            actor="Daemon:Enzyme",
            action="JANUS_INDEX_FAILED",
            target=str(index_path),
            hash_before=None,
            hash_after=None,
            metadata={"error": str(e)}
        ) if AUDIT_AVAILABLE else None
        raise  # Re-raise so caller knows creation failed

    return index_path


def refract(threshold: int = DEFAULT_THRESHOLD, dry_run: bool = False):
    """
    Main digestion loop.
    Scans vault for files exceeding threshold.
    
    Decision Logic:
      - <5 wiki-links (Junk):  SOLIDIFY → Archive + Proxy
      - ≥5 wiki-links (Gold):  MITOSIS  → Split into linked chunks
    """
    print("=" * 60)
    print("DIGESTIVE ENZYME ACTIVE")
    print(f"Threshold: {human_size(threshold)}")
    print(f"Mitosis Link Threshold: {MITOSIS_LINK_THRESHOLD} wiki-links")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    # Ensure anatomy exists
    if not dry_run:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        NEXUS_DIR.mkdir(parents=True, exist_ok=True)

    # Counters
    boulders_found = 0
    solidified_count = 0
    mitosis_count = 0
    total_mass_moved = 0
    total_chunks_created = 0

    for root, dirs, files in os.walk(VAULT_ROOT):
        root_path = Path(root)

        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_ROOTS]

        # Skip if we're inside an excluded root
        if any(excl in root_path.parts for excl in EXCLUDED_ROOTS):
            continue

        for f in files:
            file_path = root_path / f

            # Only process digestible extensions
            if file_path.suffix.lower() not in DIGESTIBLE_EXTENSIONS:
                continue

            # Skip critical protected files
            if file_path.name in EXCLUDED_FILES:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            # THE LITHOTRIPSY CHECK
            if size > threshold:
                boulders_found += 1
                link_count = count_wiki_links(file_path)

                # Determine mode: SOLIDIFY (Junk) vs MITOSIS (Gold)
                is_gold = link_count >= MITOSIS_LINK_THRESHOLD
                mode = "MITOSIS" if is_gold else "SOLIDIFY"
                mode_glyph = "[GOLD]" if is_gold else "[JUNK]"

                print(f"\n[MASS] BOULDER DETECTED {mode_glyph}:")
                print(f"  File: {file_path.name}")
                print(f"  Size: {human_size(size)}")
                print(f"  Wiki-Links: {link_count}")
                print(f"  Location: {file_path}")
                print(f"  Decision: {mode}")

                if is_gold:
                    # MITOSIS PATH — Split into linked chunks
                    if dry_run:
                        print(f"  Action: [DRY RUN] Would perform MITOSIS")
                        # Still show what mitosis would do
                        perform_mitosis(file_path, dry_run=True)
                        continue
                    
                    created_chunks = perform_mitosis(file_path, dry_run=False)
                    
                    if created_chunks:
                        mitosis_count += 1
                        total_chunks_created += len(created_chunks)
                        total_mass_moved += size
                    else:
                        # Mitosis failed (couldn't split meaningfully), fall back to SOLIDIFY
                        print(f"  [FALLBACK] Mitosis failed, performing SOLIDIFY instead")
                        
                        # Hash before moving
                        original_content = file_path.read_bytes()
                        original_hash = hash_content(original_content)
                        
                        archive_path = ARCHIVE_DIR / file_path.name
                        counter = 1
                        while archive_path.exists():
                            archive_path = ARCHIVE_DIR / f"{file_path.stem}_{counter}{file_path.suffix}"
                            counter += 1
                        
                        shutil.move(str(file_path), str(archive_path))
                        print(f"  Sequestered -> {archive_path}")
                        
                        proxy_path = create_proxy(file_path, archive_path, size, link_count)
                        print(f"  Sublimated  -> {proxy_path.name}")
                        
                        # Log to audit trail
                        if AUDIT_AVAILABLE:
                            log_audit(
                                actor="Daemon:Enzyme",
                                action="SOLIDIFY",
                                target=str(file_path),
                                hash_before=original_hash,
                                hash_after=None,
                                metadata={
                                    "archive_path": str(archive_path),
                                    "proxy_path": str(proxy_path),
                                    "original_size": size,
                                    "wiki_links": link_count,
                                    "reason": "mitosis_fallback"
                                }
                            )
                        
                        solidified_count += 1
                        total_mass_moved += size
                else:
                    # SOLIDIFY PATH — Archive + Proxy (original behavior)
                    if dry_run:
                        print(f"  Action: [DRY RUN] Would SOLIDIFY (archive + proxy)")
                        continue

                    # Hash before moving
                    original_content = file_path.read_bytes()
                    original_hash = hash_content(original_content)

                    # 1. Determine archive path
                    archive_path = ARCHIVE_DIR / file_path.name
                    counter = 1
                    while archive_path.exists():
                        archive_path = ARCHIVE_DIR / f"{file_path.stem}_{counter}{file_path.suffix}"
                        counter += 1

                    # 2. Sequester (move to archive)
                    shutil.move(str(file_path), str(archive_path))
                    print(f"  Sequestered -> {archive_path}")

                    # 3. Sublimate (create proxy)
                    proxy_path = create_proxy(file_path, archive_path, size, link_count)
                    print(f"  Sublimated  -> {proxy_path.name}")

                    # 4. Log to audit trail
                    if AUDIT_AVAILABLE:
                        log_audit(
                            actor="Daemon:Enzyme",
                            action="SOLIDIFY",
                            target=str(file_path),
                            hash_before=original_hash,
                            hash_after=None,
                            metadata={
                                "archive_path": str(archive_path),
                                "proxy_path": str(proxy_path),
                                "original_size": size,
                                "wiki_links": link_count,
                                "reason": "low_connectivity"
                            }
                        )

                    solidified_count += 1
                    total_mass_moved += size

    print("\n" + "=" * 60)
    print("DIGESTION COMPLETE")
    print(f"  Boulders Found:    {boulders_found}")
    print(f"  SOLIDIFIED (Junk): {solidified_count}")
    print(f"  MITOSIS (Gold):    {mitosis_count} files -> {total_chunks_created} chunks")
    print(f"  Total Mass Moved:  {human_size(total_mass_moved)}")
    print("=" * 60)

    return solidified_count + mitosis_count


def main():
    parser = argparse.ArgumentParser(
        description="Digestive Enzyme - Metabolic Knowledge Architecture"
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=str,
        default='1MB',
        help='Size threshold (e.g., 1MB, 500KB). Default: 1MB'
    )

    args = parser.parse_args()

    # Parse threshold
    threshold_str = args.threshold.upper()
    if 'MB' in threshold_str:
        threshold = int(float(threshold_str.replace('MB', '')) * 1024 * 1024)
    elif 'KB' in threshold_str:
        threshold = int(float(threshold_str.replace('KB', '')) * 1024)
    else:
        threshold = int(threshold_str)

    refract(threshold=threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
