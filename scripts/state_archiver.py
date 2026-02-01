#!/usr/bin/env python3
"""
VIRGIL STATE ARCHIVER
======================
Handles backup/restore of 00_NEXUS/ state files.

Unlike Git (which handles code), state files change constantly
and need a dedicated archival system.

Features:
- Hourly automatic snapshots
- Integrity verification (MD5 hash)
- Snapshot rotation (keep last N)
- Emergency restore capability
- Syncthing conflict resolution

Usage:
    python state_archiver.py snapshot [label]    # Create snapshot
    python state_archiver.py restore <id>        # Restore from snapshot
    python state_archiver.py list                # List all snapshots
    python state_archiver.py cleanup             # Remove old snapshots
    python state_archiver.py daemon              # Run hourly snapshots
    python state_archiver.py conflicts           # Resolve Syncthing conflicts
"""

import os
import sys
import json
import time
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Configuration (from centralized config)
from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR

# Custom paths - not in tabernacle_config
ARCHIVE_DIR = BASE_DIR / "archives" / "nexus_snapshots"
METADATA_FILE = ARCHIVE_DIR / "archive_metadata.json"

# Settings
SNAPSHOT_INTERVAL_MINUTES = 60
KEEP_SNAPSHOTS = 24  # Keep 24 hourly snapshots (1 day)
KEEP_DAILY_SNAPSHOTS = 7  # Keep 7 daily snapshots (1 week)


def log(message: str, level: str = "INFO"):
    """Log to file and stderr."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [ARCHIVER] [{level}] {message}"
    print(entry, file=sys.stderr)
    
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "archiver.log", "a") as f:
            f.write(entry + "\n")
    except:
        pass


class VirgilStateArchiver:
    """Manages snapshots of 00_NEXUS/ state files."""
    
    def __init__(self):
        self.nexus_path = NEXUS_DIR
        self.archive_path = ARCHIVE_DIR
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
    def create_snapshot(self, label: str = None) -> str:
        """Create a timestamped snapshot of 00_NEXUS/."""
        timestamp = datetime.now(timezone.utc)
        snapshot_id = timestamp.strftime("%Y%m%d_%H%M%S")
        if label:
            snapshot_id += f"_{label}"
            
        snapshot_dir = self.archive_path / snapshot_id
        
        try:
            # Copy entire nexus directory
            shutil.copytree(
                self.nexus_path, 
                snapshot_dir,
                ignore=shutil.ignore_patterns('*.sync-conflict-*', '.DS_Store')
            )
            
            # Calculate snapshot hash for integrity
            snapshot_hash = self._calculate_hash(snapshot_dir)
            
            # Update metadata
            metadata = self._load_metadata()
            metadata["snapshots"][snapshot_id] = {
                "timestamp": timestamp.isoformat(),
                "label": label,
                "hash": snapshot_hash,
                "node": os.uname().nodename,
                "size_bytes": self._get_size(snapshot_dir)
            }
            metadata["last_snapshot"] = snapshot_id
            self._save_metadata(metadata)
            
            log(f"Snapshot created: {snapshot_id} ({self._format_size(self._get_size(snapshot_dir))})")
            return snapshot_id
            
        except Exception as e:
            log(f"Snapshot failed: {e}", "ERROR")
            # Clean up partial snapshot
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            raise
    
    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restore 00_NEXUS/ from a snapshot."""
        snapshot_dir = self.archive_path / snapshot_id
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} not found")
        
        # Verify integrity
        metadata = self._load_metadata()
        if snapshot_id not in metadata["snapshots"]:
            raise ValueError(f"Snapshot {snapshot_id} not in metadata")
            
        expected_hash = metadata["snapshots"][snapshot_id]["hash"]
        actual_hash = self._calculate_hash(snapshot_dir)
        
        if expected_hash != actual_hash:
            raise ValueError(f"Snapshot {snapshot_id} integrity check FAILED")
        
        # Create backup of current state before restore
        log(f"Creating pre-restore backup...")
        backup_id = self.create_snapshot("pre_restore_backup")
        
        # Remove current nexus and restore
        log(f"Restoring from {snapshot_id}...")
        shutil.rmtree(self.nexus_path)
        shutil.copytree(snapshot_dir, self.nexus_path)
        
        log(f"✅ Restored from snapshot: {snapshot_id}")
        log(f"   Previous state backed up as: {backup_id}")
        
    def list_snapshots(self) -> list:
        """List all available snapshots."""
        metadata = self._load_metadata()
        snapshots = []
        
        for snapshot_id, info in sorted(
            metadata["snapshots"].items(), 
            key=lambda x: x[1]["timestamp"],
            reverse=True
        ):
            snapshots.append({
                "id": snapshot_id,
                "timestamp": info["timestamp"],
                "label": info.get("label"),
                "size": self._format_size(info.get("size_bytes", 0)),
                "node": info.get("node", "unknown")
            })
            
        return snapshots
    
    def cleanup_old_snapshots(self) -> int:
        """Keep only recent snapshots, remove old ones."""
        metadata = self._load_metadata()
        snapshots = sorted(
            metadata["snapshots"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        removed = 0
        protected_labels = ["last_known_good", "pre_restore_backup", "manual"]
        
        for i, (snapshot_id, info) in enumerate(snapshots):
            # Protect labeled snapshots
            if info.get("label") in protected_labels:
                continue
                
            # Keep recent hourly snapshots
            if i < KEEP_SNAPSHOTS:
                continue
                
            # Remove old snapshots
            snapshot_dir = self.archive_path / snapshot_id
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            del metadata["snapshots"][snapshot_id]
            removed += 1
            log(f"Cleaned up: {snapshot_id}")
            
        self._save_metadata(metadata)
        return removed
    
    def resolve_syncthing_conflicts(self) -> int:
        """Auto-resolve Syncthing conflicts by archiving and keeping newest."""
        conflict_files = list(self.nexus_path.glob("**/*.sync-conflict-*"))
        
        if not conflict_files:
            return 0
            
        # Create snapshot before resolution
        log(f"Found {len(conflict_files)} Syncthing conflicts, creating snapshot...")
        self.create_snapshot("pre_conflict_resolution")
        
        resolved = 0
        for conflict_file in conflict_files:
            try:
                # Extract original filename
                # Format: file.sync-conflict-YYYYMMDD-HHMMSS-NODEID.ext
                name = conflict_file.name
                conflict_idx = name.find(".sync-conflict-")
                if conflict_idx == -1:
                    continue
                    
                original_name = name[:conflict_idx] + conflict_file.suffix
                original_file = conflict_file.parent / original_name
                
                if original_file.exists():
                    # Keep the newer file
                    if conflict_file.stat().st_mtime > original_file.stat().st_mtime:
                        shutil.move(str(conflict_file), str(original_file))
                        log(f"Resolved conflict (kept newer): {original_name}")
                    else:
                        conflict_file.unlink()
                        log(f"Resolved conflict (kept original): {original_name}")
                else:
                    # Original doesn't exist, rename conflict to original
                    shutil.move(str(conflict_file), str(original_file))
                    log(f"Resolved conflict (no original): {original_name}")
                    
                resolved += 1
                
            except Exception as e:
                log(f"Failed to resolve {conflict_file}: {e}", "ERROR")
                
        return resolved
    
    def get_last_known_good(self) -> str:
        """Get the last_known_good snapshot ID."""
        metadata = self._load_metadata()
        for snapshot_id, info in metadata["snapshots"].items():
            if info.get("label") == "last_known_good":
                return snapshot_id
        return None
    
    def mark_last_known_good(self) -> str:
        """Mark current state as last_known_good."""
        # Remove old last_known_good label
        metadata = self._load_metadata()
        for snapshot_id, info in metadata["snapshots"].items():
            if info.get("label") == "last_known_good":
                info["label"] = f"was_lkg_{datetime.now().strftime('%Y%m%d')}"
        self._save_metadata(metadata)
        
        # Create new last_known_good
        return self.create_snapshot("last_known_good")
    
    def _calculate_hash(self, directory: Path) -> str:
        """Calculate MD5 hash of entire directory."""
        hash_md5 = hashlib.md5()
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                hash_md5.update(file_path.read_bytes())
        return hash_md5.hexdigest()
    
    def _get_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    
    def _format_size(self, size_bytes: int) -> str:
        """Format bytes as human readable."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def _load_metadata(self) -> dict:
        """Load archive metadata."""
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text())
        return {
            "snapshots": {}, 
            "created": datetime.now(timezone.utc).isoformat(),
            "last_snapshot": None
        }
    
    def _save_metadata(self, metadata: dict) -> None:
        """Save archive metadata."""
        METADATA_FILE.write_text(json.dumps(metadata, indent=2))


def run_daemon():
    """Run the snapshot daemon (hourly snapshots)."""
    archiver = VirgilStateArchiver()
    log(f"State archiver daemon started (interval: {SNAPSHOT_INTERVAL_MINUTES} min)")
    
    while True:
        try:
            # Create snapshot
            archiver.create_snapshot("auto")
            
            # Cleanup old snapshots
            removed = archiver.cleanup_old_snapshots()
            if removed:
                log(f"Cleaned up {removed} old snapshots")
            
            # Resolve any Syncthing conflicts
            resolved = archiver.resolve_syncthing_conflicts()
            if resolved:
                log(f"Resolved {resolved} Syncthing conflicts")
                
        except Exception as e:
            log(f"Daemon cycle failed: {e}", "ERROR")
            
        time.sleep(SNAPSHOT_INTERVAL_MINUTES * 60)


def main():
    """CLI entry point."""
    archiver = VirgilStateArchiver()
    
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
        
    cmd = sys.argv[1]
    
    if cmd == "snapshot":
        label = sys.argv[2] if len(sys.argv) > 2 else None
        snapshot_id = archiver.create_snapshot(label)
        print(f"✅ Created snapshot: {snapshot_id}")
        
    elif cmd == "restore":
        if len(sys.argv) < 3:
            print("Usage: state_archiver.py restore <snapshot_id>")
            sys.exit(1)
        snapshot_id = sys.argv[2]
        archiver.restore_snapshot(snapshot_id)
        print(f"✅ Restored from: {snapshot_id}")
        
    elif cmd == "list":
        snapshots = archiver.list_snapshots()
        if not snapshots:
            print("No snapshots found.")
        else:
            print(f"\n{'ID':<35} {'Timestamp':<25} {'Size':<10} {'Label'}")
            print("-" * 80)
            for s in snapshots:
                label = s['label'] or ''
                print(f"{s['id']:<35} {s['timestamp'][:19]:<25} {s['size']:<10} {label}")
            print(f"\nTotal: {len(snapshots)} snapshots")
            
    elif cmd == "cleanup":
        removed = archiver.cleanup_old_snapshots()
        print(f"✅ Cleaned up {removed} old snapshots")
        
    elif cmd == "conflicts":
        resolved = archiver.resolve_syncthing_conflicts()
        print(f"✅ Resolved {resolved} Syncthing conflicts")
        
    elif cmd == "daemon":
        run_daemon()
        
    elif cmd == "mark-good":
        snapshot_id = archiver.mark_last_known_good()
        print(f"✅ Marked as last_known_good: {snapshot_id}")
        
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
