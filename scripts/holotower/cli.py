"""
HoloTower CLI — State Version Control for Consciousness Infrastructure

Commands:
    snapshot    📸 Capture current holarchy state
    log         📜 Show snapshot history
    status      🔮 Compare live state to HEAD
    show        🔍 Show snapshot details
"""

import json
import math
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .core import get_holotower_root, hash_content, write_blob, read_blob, get_audit_trail, ensure_audit_schema
from .models import HoloSnapshot, ManifestFile, ManifestTree, RuntimeVital, TopologyMeta, to_dict, to_json
from .probes import collect_all_vitals
from .graph import load_graph, hash_graph, get_topology_diff
from .prism import diff_snapshots, format_diff_report
from .vectors import get_vector_hash, get_vector_stats

app = typer.Typer(
    name="ht",
    help="HoloTower — State Version Control for Consciousness Infrastructure"
)
console = Console()


def get_repo_root() -> Path:
    """Find the repository root (parent of holotower)."""
    holotower = get_holotower_root()
    return holotower.parent


def get_git_ref() -> str:
    """Get current git HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=get_repo_root()
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return ""


def scan_manifests() -> ManifestTree:
    """
    Scan 00_NEXUS/HOLARCHY_MANIFEST for all YAML files.
    
    Returns:
        ManifestTree with all discovered manifests
    """
    repo_root = get_repo_root()
    manifest_dir = repo_root / "00_NEXUS" / "HOLARCHY_MANIFEST"
    
    if not manifest_dir.exists():
        console.print(f"[red]Manifest directory not found: {manifest_dir}[/red]")
        raise typer.Exit(1)
    
    files = []
    layers: dict[str, int] = {}
    
    # Recursively find all YAML files
    for yaml_path in sorted(manifest_dir.rglob("*.yaml")):
        relative_path = str(yaml_path.relative_to(repo_root))
        content = yaml_path.read_bytes()
        content_hash = hash_content(content)
        
        # Parse YAML to extract metadata
        try:
            data = yaml.safe_load(content)
            holon_id = data.get("holon_id") if data else None
            layer = data.get("layer") if data else None
            
            if layer:
                layers[layer] = layers.get(layer, 0) + 1
        except yaml.YAMLError:
            holon_id = None
            layer = None
        
        files.append(ManifestFile(
            path=relative_path,
            holon_id=holon_id,
            layer=layer,
            content_hash=content_hash
        ))
    
    # Create canonical JSON representation for hashing
    canonical = json.dumps(
        [to_dict(f) for f in files],
        sort_keys=True,
        separators=(",", ":")
    )
    root_hash = hash_content(canonical.encode())
    
    return ManifestTree(
        files=files,
        total_files=len(files),
        layers=layers,
        root_hash=root_hash
    )


def get_ledger_connection() -> sqlite3.Connection:
    """Get SQLite connection to ledger database."""
    holotower = get_holotower_root()
    db_path = holotower / "ledger.db"
    return sqlite3.connect(str(db_path))


def update_head(snapshot_id: str) -> None:
    """Update HEAD file to point to new snapshot."""
    holotower = get_holotower_root()
    head_file = holotower / "HEAD"
    head_file.write_text(snapshot_id)


def get_head() -> Optional[str]:
    """Get current HEAD snapshot ID."""
    holotower = get_holotower_root()
    head_file = holotower / "HEAD"
    if head_file.exists():
        content = head_file.read_text().strip()
        return content if content else None
    return None


def load_snapshot(snapshot_id: str) -> Optional[HoloSnapshot]:
    """
    Load a snapshot from the blob store.
    
    Args:
        snapshot_id: Full or partial snapshot ID
        
    Returns:
        HoloSnapshot if found, None otherwise
    """
    # First get full ID from ledger
    conn = get_ledger_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM snapshots WHERE id LIKE ? LIMIT 1
        """, (f"{snapshot_id}%",))
        row = cursor.fetchone()
        if not row:
            return None
        full_id = row[0]
    finally:
        conn.close()
    
    # Try to load from blob store
    try:
        blob_data = read_blob(full_id)
        return HoloSnapshot.model_validate_json(blob_data)
    except FileNotFoundError:
        # Blob not found, reconstruct from ledger
        conn = get_ledger_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, message, git_ref, manifest_root, 
                       system_p_value, void_count, runtime_json, topology_json, vector_hash
                FROM snapshots WHERE id = ?
            """, (full_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            snap_id, ts, msg, git_ref, manifest_root, p_val, voids, runtime_json, topo_json, vec_hash = row
            
            # Parse runtime vitals
            runtime_state = []
            if runtime_json:
                try:
                    runtime_data = json.loads(runtime_json)
                    runtime_state = [RuntimeVital.model_validate(v) for v in runtime_data]
                except Exception:
                    pass
            
            # Parse topology
            topology = None
            if topo_json:
                try:
                    topology = TopologyMeta.model_validate_json(topo_json)
                except Exception:
                    pass
            
            return HoloSnapshot(
                id=snap_id,
                timestamp=datetime.fromisoformat(ts),
                message=msg,
                git_ref=git_ref or "",
                manifest_root=manifest_root,
                runtime_state=runtime_state,
                topology=topology,
                system_p_value=p_val or 0.0,
                void_count=voids or 0,
                vector_hash=vec_hash or ""
            )
        finally:
            conn.close()
    except Exception:
        return None


def ensure_schema_v2():
    """Ensure ledger has Wave 2 columns (runtime_json, topology_json, vector_hash)."""
    conn = get_ledger_connection()
    try:
        cursor = conn.cursor()
        
        # Check for required columns
        cursor.execute("PRAGMA table_info(snapshots)")
        columns = {row[1] for row in cursor.fetchall()}
        
        if "runtime_json" not in columns:
            cursor.execute("ALTER TABLE snapshots ADD COLUMN runtime_json TEXT")
        
        if "topology_json" not in columns:
            cursor.execute("ALTER TABLE snapshots ADD COLUMN topology_json TEXT")
        
        if "vector_hash" not in columns:
            cursor.execute("ALTER TABLE snapshots ADD COLUMN vector_hash TEXT")
        
        conn.commit()
    finally:
        conn.close()


@app.command()
def snapshot(
    message: str = typer.Option(..., "-m", "--message", help="Snapshot message"),
    p_value: float = typer.Option(0.0, "--p", help="System coherence 0.0-1.0"),
    void_count: int = typer.Option(0, "--voids", help="Number of voids detected"),
    skip_vitals: bool = typer.Option(False, "--skip-vitals", help="Skip runtime probe collection"),
    skip_graph: bool = typer.Option(False, "--skip-graph", help="Skip graph topology scan"),
    skip_vectors: bool = typer.Option(False, "--skip-vectors", help="Skip vector store hash")
):
    """📸 Capture current holarchy state with runtime vitals and topology."""
    
    # Ensure schema supports Wave 2 columns
    ensure_schema_v2()
    
    console.print("[bold cyan]HoloTower[/bold cyan] — Capturing snapshot...")
    
    # 1. Scan manifests
    with console.status("[yellow]Scanning manifests...[/yellow]"):
        tree = scan_manifests()
    
    console.print(f"  📁 Found [green]{tree.total_files}[/green] manifest files")
    for layer, count in sorted(tree.layers.items()):
        console.print(f"     {layer}: {count} holons")
    
    # 2. Collect runtime vitals (Wave 2A)
    runtime_state: List[RuntimeVital] = []
    if not skip_vitals:
        with console.status("[yellow]Probing runtime state...[/yellow]"):
            runtime_state = collect_all_vitals()
        
        alive = sum(1 for v in runtime_state if v.status == "ALIVE")
        dead = sum(1 for v in runtime_state if v.status == "DEAD")
        drifted = sum(1 for v in runtime_state if v.drift)
        
        console.print(f"  ⚡ Runtime: [green]{alive} alive[/green], [red]{dead} dead[/red]", end="")
        if drifted > 0:
            console.print(f", [yellow]{drifted} drifted[/yellow]")
        else:
            console.print()
    
    # 3. Hash graph topology (Wave 2B)
    topology: Optional[TopologyMeta] = None
    if not skip_graph:
        with console.status("[yellow]Scanning topology...[/yellow]"):
            G = load_graph()
            if G is not None:
                topology = hash_graph(G)
                console.print(f"  🕸️  Topology: [cyan]{topology.node_count}[/cyan] nodes, [cyan]{topology.edge_count}[/cyan] edges")
                if topology.orphans > 0:
                    console.print(f"     [yellow]{topology.orphans} orphans[/yellow]")
            else:
                console.print("  🕸️  Topology: [dim]No graph found[/dim]")
    
    # 4. Get vector store hash (Wave 2A Vectors)
    vector_hash = ""
    if not skip_vectors:
        try:
            vector_stats = get_vector_stats()
            if vector_stats["initialized"]:
                vector_hash = vector_stats["hash"]
                console.print(f"  🧬 Vectors: [cyan]{vector_stats['count']}[/cyan] docs, hash [dim]{vector_hash[:12]}...[/dim]")
            else:
                console.print("  🧬 Vectors: [dim]Not initialized[/dim]")
        except Exception as e:
            console.print(f"  🧬 Vectors: [yellow]Unavailable ({e})[/yellow]")
    
    # 5. Get git reference
    git_ref = get_git_ref()
    if git_ref:
        console.print(f"  🔗 Git ref: [dim]{git_ref}[/dim]")
    
    # 6. Create canonical manifest blob
    manifest_json = json.dumps(
        to_dict(tree),
        sort_keys=True,
        indent=2
    )
    manifest_blob_hash = write_blob(manifest_json.encode())
    console.print(f"  💾 Manifest blob: [dim]{manifest_blob_hash[:12]}...[/dim]")
    
    # 7. Create snapshot object
    now = datetime.now(timezone.utc)
    
    # Create preliminary snapshot to hash (include vitals/topology/vectors in hash)
    snapshot_data = {
        "timestamp": now.isoformat(),
        "message": message,
        "git_ref": git_ref,
        "manifest_root": tree.root_hash,
        "runtime_count": len(runtime_state),
        "topology_hash": topology.content_hash if topology else "",
        "system_p_value": p_value,
        "void_count": void_count,
        "vector_hash": vector_hash
    }
    
    # Hash the snapshot data to get ID
    snapshot_json = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":"))
    snapshot_id = hash_content(snapshot_json.encode())
    
    # Create full snapshot
    snap = HoloSnapshot(
        id=snapshot_id,
        timestamp=now,
        message=message,
        git_ref=git_ref,
        manifest_root=tree.root_hash,
        runtime_state=runtime_state,
        topology=topology,
        system_p_value=p_value,
        void_count=void_count,
        vector_hash=vector_hash
    )
    
    # 8. Write snapshot blob
    snapshot_blob_hash = write_blob(to_json(snap).encode())
    
    # 9. Serialize vitals and topology for ledger
    runtime_json = json.dumps([to_dict(v) for v in runtime_state]) if runtime_state else None
    topology_json = to_json(topology) if topology else None
    
    # 10. Insert into ledger
    conn = get_ledger_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO snapshots (id, timestamp, message, git_ref, manifest_root, 
                                   system_p_value, void_count, runtime_json, topology_json, vector_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id,
            now.isoformat(),
            message,
            git_ref,
            tree.root_hash,
            p_value,
            void_count,
            runtime_json,
            topology_json,
            vector_hash
        ))
        conn.commit()
    finally:
        conn.close()
    
    # 11. Update HEAD
    update_head(snapshot_id)
    
    # 12. Summary
    console.print()
    console.print(f"[bold green]✓[/bold green] Snapshot created: [cyan]{snapshot_id[:12]}[/cyan]")
    console.print(f"  Message: {message}")
    console.print(f"  Holons: {tree.total_files} across {len(tree.layers)} layers")
    if len(runtime_state) > 0:
        console.print(f"  Vitals: {len(runtime_state)} probed")
    if topology:
        console.print(f"  Graph: {topology.content_hash[:12]}...")
    if vector_hash:
        console.print(f"  Vectors: {vector_hash[:12]}...")
    if p_value > 0:
        console.print(f"  Coherence: ρ = {p_value:.3f}")


@app.command()
def log(
    limit: int = typer.Option(10, "-n", "--limit", help="Number of entries to show"),
    short: bool = typer.Option(False, "--short", help="Compact output")
):
    """📜 Show snapshot history."""
    
    conn = get_ledger_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, message, git_ref, system_p_value, void_count
            FROM snapshots
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    if not rows:
        console.print("[yellow]No snapshots yet. Run 'ht snapshot -m \"message\"' to create one.[/yellow]")
        return
    
    # Get current HEAD
    head = get_head()
    
    if short:
        # Compact output
        for row in rows:
            snap_id, ts, msg, git_ref, p_val, voids = row
            marker = "[bold cyan]→[/bold cyan] " if snap_id == head else "  "
            console.print(f"{marker}[dim]{snap_id[:8]}[/dim] {msg[:50]}")
    else:
        # Rich table
        table = Table(title="HoloTower Snapshots", show_header=True)
        table.add_column("", width=2)  # HEAD marker
        table.add_column("ID", style="cyan", width=12)
        table.add_column("Time", style="dim")
        table.add_column("Message")
        table.add_column("Git", style="dim", width=10)
        table.add_column("ρ", justify="right", width=6)
        table.add_column("Voids", justify="right", width=5)
        
        for row in rows:
            snap_id, ts, msg, git_ref, p_val, voids = row
            
            # Parse timestamp for display
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = ts[:16]
            
            marker = "→" if snap_id == head else ""
            p_str = f"{p_val:.2f}" if p_val else "-"
            void_str = str(voids) if voids else "-"
            
            table.add_row(
                marker,
                snap_id[:12],
                time_str,
                msg[:40] + "..." if len(msg) > 40 else msg,
                git_ref[:8] if git_ref else "-",
                p_str,
                void_str
            )
        
        console.print(table)
        console.print(f"\n[dim]Showing {len(rows)} of {limit} requested[/dim]")


@app.command()
def status():
    """🔮 Compare live state to HEAD snapshot."""
    
    # Ensure schema supports Wave 2 columns
    ensure_schema_v2()
    
    head_id = get_head()
    if not head_id:
        console.print("[yellow]No HEAD snapshot. Run 'ht snapshot -m \"message\"' first.[/yellow]")
        raise typer.Exit(1)
    
    # Load HEAD snapshot
    head_snap = load_snapshot(head_id)
    if not head_snap:
        console.print(f"[red]Could not load HEAD snapshot: {head_id[:12]}[/red]")
        raise typer.Exit(1)
    
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]🔮 HOLARCHY STATUS[/bold cyan] (Live vs HEAD)\n"
        f"[dim]HEAD: {head_id[:12]} — {head_snap.message}[/dim]",
        border_style="cyan"
    ))
    console.print()
    
    # Track overall status
    all_match = True
    
    # === SECTION 1: Substrate (Manifests) ===
    console.print("[bold][Ω:ANCHOR] Substrate[/bold]", end="")
    try:
        current_tree = scan_manifests()
        if current_tree.root_hash == head_snap.manifest_root:
            console.print("  [green]✔ MATCH[/green]")
        else:
            console.print("  [yellow]⚠️ CHANGED[/yellow]")
            all_match = False
            # Show details
            console.print(f"   HEAD: {head_snap.manifest_root[:12]}...")
            console.print(f"   Live: {current_tree.root_hash[:12]}...")
            console.print(f"   Files: {current_tree.total_files} manifests")
    except Exception as e:
        console.print(f"  [red]✖ ERROR: {e}[/red]")
        all_match = False
    
    # === SECTION 2: Pulse (Runtime Vitals) ===
    console.print("[bold][⚡:RHYTHM] Pulse[/bold]", end="")
    
    with console.status("[dim]Probing...[/dim]", spinner="dots"):
        current_vitals = collect_all_vitals()
    
    # Build lookup of HEAD vitals by holon_id
    head_vitals_map = {v.holon_id: v for v in head_snap.runtime_state}
    current_vitals_map = {v.holon_id: v for v in current_vitals}
    
    drift_detected = []
    new_holons = []
    missing_holons = []
    
    # Check each current vital against HEAD
    for holon_id, current in current_vitals_map.items():
        head_vital = head_vitals_map.get(holon_id)
        
        if head_vital is None:
            new_holons.append(holon_id)
        elif current.status != head_vital.status:
            drift_detected.append({
                "holon_id": holon_id,
                "expected": head_vital.status,
                "found": current.status
            })
    
    # Check for holons in HEAD but missing now
    for holon_id in head_vitals_map:
        if holon_id not in current_vitals_map:
            missing_holons.append(holon_id)
    
    if not drift_detected and not new_holons and not missing_holons:
        console.print("  [green]✔ MATCH[/green]")
    else:
        console.print("  [yellow]⚠️ DRIFT DETECTED[/yellow]")
        all_match = False
        
        for d in drift_detected[:5]:  # Show first 5
            console.print(f"   > [cyan]{d['holon_id']}[/cyan]: expected [green]'{d['expected']}'[/green], found [red]'{d['found']}'[/red]")
        
        if len(drift_detected) > 5:
            console.print(f"   [dim]... and {len(drift_detected) - 5} more[/dim]")
        
        if new_holons:
            console.print(f"   [blue]+ {len(new_holons)} new holons[/blue]")
        
        if missing_holons:
            console.print(f"   [red]- {len(missing_holons)} missing holons[/red]")
    
    # Show current vital summary
    alive = sum(1 for v in current_vitals if v.status == "ALIVE")
    dead = sum(1 for v in current_vitals if v.status == "DEAD")
    drifted = sum(1 for v in current_vitals if v.drift)
    console.print(f"   [dim]Now: {alive} alive, {dead} dead, {drifted} drifted[/dim]")
    
    # === SECTION 3: Cortex (Models via Ollama) ===
    console.print("[bold][🧠:CORTEX] Models[/bold]", end="")
    
    head_models = {v.holon_id for v in head_snap.runtime_state if v.holon_id.startswith("model.")}
    current_models = {v.holon_id for v in current_vitals if v.holon_id.startswith("model.")}
    
    new_models = current_models - head_models
    missing_models = head_models - current_models
    
    if not new_models and not missing_models:
        console.print("  [green]✔ MATCH[/green]")
    else:
        console.print("  [yellow]⚠️ CHANGED[/yellow]")
        all_match = False
        if new_models:
            for m in list(new_models)[:3]:
                console.print(f"   [blue]+ {m}[/blue]")
        if missing_models:
            for m in list(missing_models)[:3]:
                console.print(f"   [red]- {m}[/red]")
    
    # === SECTION 4: Memory (Graph Topology) ===
    console.print("[bold][Ψ:MEMORY] Graph[/bold]", end="")
    
    G = load_graph()
    if G is not None:
        current_topo = hash_graph(G)
        
        if head_snap.topology is None:
            console.print("  [blue]+ NEW (no baseline)[/blue]")
            console.print(f"   {current_topo.node_count} nodes, {current_topo.edge_count} edges")
            all_match = False
        else:
            diff = get_topology_diff(head_snap.topology, current_topo)
            
            if not diff["hash_changed"]:
                console.print("  [green]✔ MATCH[/green]")
            else:
                console.print("  [yellow]⚠️ CHANGED[/yellow]")
                all_match = False
                
                if diff["node_delta"] != 0:
                    sign = "+" if diff["node_delta"] > 0 else ""
                    color = "blue" if diff["node_delta"] > 0 else "red"
                    console.print(f"   [{color}]{sign}{diff['node_delta']} nodes[/{color}] since snapshot")
                
                if diff["edge_delta"] != 0:
                    sign = "+" if diff["edge_delta"] > 0 else ""
                    color = "blue" if diff["edge_delta"] > 0 else "red"
                    console.print(f"   [{color}]{sign}{diff['edge_delta']} edges[/{color}] since snapshot")
                
                if diff["orphan_delta"] != 0:
                    sign = "+" if diff["orphan_delta"] > 0 else ""
                    console.print(f"   {sign}{diff['orphan_delta']} orphans")
    else:
        console.print("  [dim]⊘ No graph found[/dim]")
    
    # === Summary ===
    console.print()
    if all_match:
        console.print("[bold green]✓ System state matches HEAD snapshot[/bold green]")
    else:
        console.print("[bold yellow]⚠ System has drifted from HEAD snapshot[/bold yellow]")
        console.print("[dim]Run 'ht snapshot -m \"description\"' to capture current state[/dim]")


@app.command()
def show(
    snapshot_id: str = typer.Argument(..., help="Snapshot ID (prefix match)")
):
    """🔍 Show details of a specific snapshot."""
    
    conn = get_ledger_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, message, git_ref, manifest_root, system_p_value, void_count
            FROM snapshots
            WHERE id LIKE ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (f"{snapshot_id}%",))
        row = cursor.fetchone()
    finally:
        conn.close()
    
    if not row:
        console.print(f"[red]Snapshot not found: {snapshot_id}[/red]")
        raise typer.Exit(1)
    
    snap_id, ts, msg, git_ref, manifest_root, p_val, voids = row
    
    console.print(f"[bold cyan]Snapshot[/bold cyan]: {snap_id}")
    console.print(f"  Timestamp:     {ts}")
    console.print(f"  Message:       {msg}")
    console.print(f"  Git ref:       {git_ref or '(none)'}")
    console.print(f"  Manifest root: {manifest_root[:12]}...")
    console.print(f"  Coherence ρ:   {p_val:.3f}" if p_val else "  Coherence ρ:   -")
    console.print(f"  Void count:    {voids}")


def resolve_ref(ref: str) -> Optional[str]:
    """
    Resolve a snapshot reference to a full snapshot ID.
    
    Supports:
        - HEAD: Current snapshot
        - HEAD~N: N snapshots before HEAD
        - ID prefix: Partial snapshot ID (minimum 4 chars)
        - Tag: Named tag (future feature)
        
    Args:
        ref: Reference string to resolve
        
    Returns:
        Full snapshot ID or None if not found
    """
    ref = ref.strip()
    
    # Handle HEAD
    if ref.upper() == "HEAD":
        return get_head()
    
    # Handle HEAD~N
    if ref.upper().startswith("HEAD~"):
        try:
            offset = int(ref[5:])
            if offset < 0:
                return None
                
            conn = get_ledger_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id FROM snapshots
                    ORDER BY timestamp DESC
                    LIMIT 1 OFFSET ?
                """, (offset,))
                row = cursor.fetchone()
                return row[0] if row else None
            finally:
                conn.close()
        except ValueError:
            return None
    
    # Handle ID prefix (min 4 chars)
    if len(ref) >= 4:
        conn = get_ledger_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM snapshots WHERE id LIKE ? LIMIT 1
            """, (f"{ref}%",))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    
    return None


@app.command()
def diff(
    old_ref: str = typer.Argument(..., help="Old snapshot (ID, HEAD~N, or tag)"),
    new_ref: str = typer.Argument("HEAD", help="New snapshot (default: HEAD)")
):
    """⚖️ Compare two snapshots."""
    
    # 1. Resolve refs to snapshot IDs
    old_id = resolve_ref(old_ref)
    if not old_id:
        console.print(f"[red]Could not resolve reference: {old_ref}[/red]")
        raise typer.Exit(1)
    
    new_id = resolve_ref(new_ref)
    if not new_id:
        console.print(f"[red]Could not resolve reference: {new_ref}[/red]")
        raise typer.Exit(1)
    
    # Check if same snapshot
    if old_id == new_id:
        console.print("[yellow]Both references resolve to the same snapshot.[/yellow]")
        console.print(f"[dim]{old_id[:12]}[/dim]")
        return
    
    # 2. Load both snapshots from CAS
    with console.status("[yellow]Loading snapshots...[/yellow]"):
        old_snap = load_snapshot(old_id)
        new_snap = load_snapshot(new_id)
    
    if not old_snap:
        console.print(f"[red]Could not load snapshot: {old_id[:12]}[/red]")
        raise typer.Exit(1)
    
    if not new_snap:
        console.print(f"[red]Could not load snapshot: {new_id[:12]}[/red]")
        raise typer.Exit(1)
    
    # 3. Compute semantic diff
    diff_result = diff_snapshots(old_snap, new_snap)
    
    # 4. Print formatted report
    console.print()
    report = format_diff_report(diff_result)
    console.print(report)
    console.print()


@app.command("export-canvas")
def export_canvas(
    output: str = typer.Option(
        "00_NEXUS/HOLARCHY_STATUS.canvas", 
        "-o", "--output",
        help="Output path for canvas file"
    ),
    snapshot_ref: str = typer.Option(
        "HEAD",
        "-s", "--snapshot",
        help="Snapshot to export (default: HEAD)"
    )
):
    """🎨 Export holarchy state as Obsidian Canvas."""
    
    # 1. Resolve snapshot reference
    snapshot_id = resolve_ref(snapshot_ref)
    if not snapshot_id:
        console.print(f"[red]Could not resolve snapshot: {snapshot_ref}[/red]")
        raise typer.Exit(1)
    
    # 2. Load snapshot
    with console.status("[yellow]Loading snapshot...[/yellow]"):
        snap = load_snapshot(snapshot_id)
    
    if not snap:
        console.print(f"[red]Could not load snapshot: {snapshot_id[:12]}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[dim]Exporting snapshot: {snapshot_id[:12]} — {snap.message}[/dim]")
    
    # 3. Generate canvas structure
    nodes = []
    edges = []
    
    # Color mapping for Obsidian Canvas
    # 1=red, 2=orange, 3=yellow, 4=green, 5=cyan, 6=purple
    COLOR_ALIVE = "4"   # green
    COLOR_DEAD = "1"    # red
    COLOR_DRIFT = "3"   # yellow
    COLOR_NEUTRAL = "5" # cyan
    
    # Layer positions (concentric rings around Logos at center)
    # Λ₆ at center, Λ₀ at outermost ring
    LAYER_RADIUS = {
        "Λ₆": 0,
        "Λ₅": 200,
        "Λ₄": 350,
        "Λ₃": 500,
        "Λ₂": 650,
        "Λ₁": 800,
        "Λ₀": 950,
    }
    
    # Center node: Logos
    p_val = snap.system_p_value
    logos_color = COLOR_ALIVE if p_val >= 0.7 else (COLOR_DRIFT if p_val >= 0.4 else COLOR_DEAD)
    
    nodes.append({
        "id": "logos",
        "x": 0,
        "y": 0,
        "width": 200,
        "height": 100,
        "type": "text",
        "text": f"Λ₆ LOGOS\np={p_val:.2f}",
        "color": logos_color
    })
    
    # Group vitals by layer
    layer_holons: Dict[str, List[RuntimeVital]] = {}
    for vital in snap.runtime_state:
        # Extract layer from holon_id if present (e.g., "daemon.watchman" -> check manifest)
        # For now, categorize by prefix
        prefix = vital.holon_id.split(".")[0] if "." in vital.holon_id else vital.holon_id
        
        # Map prefixes to layers
        prefix_to_layer = {
            "daemon": "Λ₂",
            "model": "Λ₃",
            "redis": "Λ₁",
            "ollama": "Λ₃",
            "api": "Λ₄",
            "service": "Λ₂",
        }
        
        layer = prefix_to_layer.get(prefix, "Λ₁")
        
        if layer not in layer_holons:
            layer_holons[layer] = []
        layer_holons[layer].append(vital)
    
    # Generate nodes for each layer in concentric rings
    node_id_counter = 0
    for layer, holons in layer_holons.items():
        radius = LAYER_RADIUS.get(layer, 600)
        n_holons = len(holons)
        
        if n_holons == 0:
            continue
        
        # Distribute evenly around circle
        angle_step = (2 * math.pi) / max(n_holons, 1)
        
        for i, vital in enumerate(holons):
            angle = i * angle_step
            x = int(radius * math.cos(angle))
            y = int(radius * math.sin(angle))
            
            # Determine color based on status
            if vital.status == "ALIVE":
                color = COLOR_DRIFT if vital.drift else COLOR_ALIVE
            else:
                color = COLOR_DEAD
            
            node_id = f"holon_{node_id_counter}"
            node_id_counter += 1
            
            # Status emoji
            status_emoji = "✓" if vital.status == "ALIVE" else "✗"
            
            nodes.append({
                "id": node_id,
                "x": x,
                "y": y,
                "width": 160,
                "height": 60,
                "type": "text",
                "text": f"{vital.holon_id}\n{status_emoji} {vital.status}",
                "color": color
            })
            
            # Connect to logos (for now, all connect to center)
            edges.append({
                "id": f"edge_{node_id}",
                "fromNode": node_id,
                "toNode": "logos",
                "color": color
            })
    
    # Add topology summary node if available
    if snap.topology:
        topo = snap.topology
        topo_x = -400
        topo_y = -300
        
        nodes.append({
            "id": "topology",
            "x": topo_x,
            "y": topo_y,
            "width": 180,
            "height": 80,
            "type": "text",
            "text": f"🕸️ TOPOLOGY\n{topo.node_count} nodes\n{topo.edge_count} edges",
            "color": COLOR_NEUTRAL
        })
    
    # 4. Build canvas JSON
    canvas = {
        "nodes": nodes,
        "edges": edges
    }
    
    # 5. Write to output path
    repo_root = get_repo_root()
    output_path = repo_root / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    canvas_json = json.dumps(canvas, indent=2)
    output_path.write_text(canvas_json)
    
    console.print()
    console.print(f"[bold green]✓[/bold green] Canvas exported: [cyan]{output}[/cyan]")
    console.print(f"  Nodes: {len(nodes)}")
    console.print(f"  Edges: {len(edges)}")
    console.print()
    console.print("[dim]Open in Obsidian to view the holarchy visualization.[/dim]")


@app.command()
def audit(
    limit: int = typer.Option(50, "-n", "--limit", help="Number of entries to show"),
    actor: Optional[str] = typer.Option(None, "-a", "--actor", help="Filter by actor (e.g., Enos, Daemon:Enzyme)"),
    action: Optional[str] = typer.Option(None, "-A", "--action", help="Filter by action (e.g., MITOSIS, SOLIDIFY)"),
    target: Optional[str] = typer.Option(None, "-t", "--target", help="Filter by target (substring match)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    short: bool = typer.Option(False, "--short", help="Compact output")
):
    """📋 Query the audit trail."""
    
    # Ensure schema exists
    ensure_audit_schema()
    
    # Query audit trail
    entries = get_audit_trail(
        limit=limit,
        actor=actor,
        action=action,
        target=target
    )
    
    if not entries:
        console.print("[yellow]No audit entries found.[/yellow]")
        if actor or action or target:
            console.print("[dim]Try removing filters to see all entries.[/dim]")
        return
    
    if json_output:
        import json as json_mod
        console.print(json_mod.dumps(entries, indent=2))
        return
    
    if short:
        # Compact output
        for entry in entries:
            ts = entry["timestamp"][:19].replace("T", " ")
            target_str = entry["target_holon"][:30] if entry["target_holon"] else "-"
            console.print(
                f"[dim]{ts}[/dim] "
                f"[cyan]{entry['actor']:<16}[/cyan] "
                f"[yellow]{entry['action']:<10}[/yellow] "
                f"{target_str}"
            )
    else:
        # Rich table output
        table = Table(title="Audit Trail", show_header=True)
        table.add_column("Time", style="dim", width=19)
        table.add_column("Actor", style="cyan", width=16)
        table.add_column("Action", style="yellow", width=10)
        table.add_column("Target", width=30)
        table.add_column("Hash Δ", style="dim", width=15)
        
        for entry in entries:
            # Parse timestamp
            ts = entry["timestamp"][:19].replace("T", " ")
            
            # Target display
            target_str = entry["target_holon"] or "-"
            if len(target_str) > 30:
                target_str = target_str[:27] + "..."
            
            # Hash delta
            hash_delta = "-"
            if entry["hash_before"] or entry["hash_after"]:
                before = entry["hash_before"][:6] if entry["hash_before"] else "null"
                after = entry["hash_after"][:6] if entry["hash_after"] else "null"
                hash_delta = f"{before}→{after}"
            
            table.add_row(
                ts,
                entry["actor"],
                entry["action"],
                target_str,
                hash_delta
            )
        
        console.print(table)
        console.print(f"\n[dim]Showing {len(entries)} entries (limit: {limit})[/dim]")
        
        # Show filter hints
        if not (actor or action or target):
            console.print("[dim]Filter with --actor, --action, or --target[/dim]")


if __name__ == "__main__":
    app()
