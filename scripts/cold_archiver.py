#!/usr/bin/env python3
"""
VIRGIL COLD ARCHIVER - The Vault Keeper
========================================
Manages cold storage on the Seagate external drive.

Functions:
- Archive old logs (weekly rotation)
- Create daily NEXUS snapshots
- Create weekly full backups (Borg)
- Enforce retention policies
- Mount verification and safety checks

Schedule: Daily at 4:00 AM via launchd

Author: Virgil
Date: 2026-01-17
"""

import os
import sys
import json
import shutil
import tarfile
import hashlib
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

# ============================================================================
# CONFIGURATION (using centralized config)
# ============================================================================

from tabernacle_config import BASE_DIR, NEXUS_DIR, LOG_DIR, SCRIPTS_DIR

TABERNACLE = BASE_DIR  # Alias for backwards compatibility
NEXUS = NEXUS_DIR
LOGS_DIR = LOG_DIR

# Cold storage paths
SEAGATE_VOLUME = Path("/Volumes/Seagate Portable Drive")
VAULT = SEAGATE_VOLUME / "TABERNACLE_VAULT"
VAULT_RAW = VAULT / "RAW_ARCHIVES"
VAULT_LOGS = VAULT / "LOG_ARCHIVE"
VAULT_SNAPSHOTS = VAULT / "NEXUS_SNAPSHOTS"
VAULT_BACKUPS = VAULT / "FULL_BACKUPS"
VAULT_BORG = VAULT / "BORG_REPO"

# Retention policies
RETENTION_SNAPSHOTS_DAYS = 30      # Keep daily snapshots for 30 days
RETENTION_BACKUPS_WEEKS = 52       # Keep weekly backups for 1 year
RETENTION_LOGS_MONTHS = 24         # Keep monthly log archives for 2 years

# State file
STATE_FILE = NEXUS / "cold_archiver_state.json"

# Logging
LOG_FILE = LOGS_DIR / "cold_archiver.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [COLD] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state() -> Dict:
    """Load archiver state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load state: {e}")
    return {
        "last_snapshot": None,
        "last_backup": None,
        "last_log_archive": None,
        "last_cleanup": None,
        "stats": {
            "total_snapshots": 0,
            "total_backups": 0,
            "total_archived_mb": 0
        }
    }


def save_state(state: Dict) -> None:
    """Save archiver state to disk."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save state: {e}")


# ============================================================================
# SAFETY CHECKS
# ============================================================================

def is_seagate_mounted() -> bool:
    """Check if Seagate drive is mounted."""
    return SEAGATE_VOLUME.exists() and SEAGATE_VOLUME.is_dir()


def ensure_vault_structure() -> bool:
    """Ensure vault directory structure exists."""
    if not is_seagate_mounted():
        log.error("Seagate drive not mounted!")
        return False

    for vault_dir in [VAULT, VAULT_LOGS, VAULT_SNAPSHOTS, VAULT_BACKUPS, VAULT_BORG]:
        vault_dir.mkdir(parents=True, exist_ok=True)

    return True


def get_vault_stats() -> Dict:
    """Get storage stats for the vault."""
    if not is_seagate_mounted():
        return {"error": "Not mounted"}

    try:
        stat = os.statvfs(SEAGATE_VOLUME)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        used = total - free

        return {
            "total_gb": round(total / (1024**3), 1),
            "used_gb": round(used / (1024**3), 1),
            "free_gb": round(free / (1024**3), 1),
            "percent_used": round((used / total) * 100, 1)
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# ARCHIVING FUNCTIONS
# ============================================================================

def create_nexus_snapshot() -> Optional[Path]:
    """Create a compressed snapshot of 00_NEXUS."""
    if not ensure_vault_structure():
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    snapshot_name = f"nexus_{timestamp}.tar.gz"
    snapshot_path = VAULT_SNAPSHOTS / snapshot_name

    try:
        log.info(f"Creating NEXUS snapshot: {snapshot_name}")

        with tarfile.open(snapshot_path, "w:gz") as tar:
            # Exclude large/temporary files
            def exclude_filter(tarinfo):
                name = tarinfo.name
                # Exclude cache directories and large files
                if any(x in name for x in ['.garmin_cache', '__pycache__', '.git']):
                    return None
                # Exclude files larger than 10MB
                if tarinfo.size > 10 * 1024 * 1024:
                    log.debug(f"Skipping large file: {name} ({tarinfo.size / 1024 / 1024:.1f}MB)")
                    return None
                return tarinfo

            tar.add(NEXUS, arcname="00_NEXUS", filter=exclude_filter)

        size_mb = snapshot_path.stat().st_size / (1024 * 1024)
        log.info(f"Snapshot created: {snapshot_name} ({size_mb:.1f}MB)")

        return snapshot_path

    except Exception as e:
        log.error(f"Failed to create snapshot: {e}")
        return None


def archive_old_logs() -> Optional[Path]:
    """Archive logs older than 7 days into monthly tarballs."""
    if not ensure_vault_structure():
        return None

    now = datetime.now()
    cutoff = now - timedelta(days=7)

    # Group old logs by month
    logs_by_month = {}

    for log_file in LOGS_DIR.glob("*.log"):
        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff:
                month_key = mtime.strftime("%Y-%m")
                if month_key not in logs_by_month:
                    logs_by_month[month_key] = []
                logs_by_month[month_key].append(log_file)
        except Exception:
            continue

    archived_files = []

    for month, files in logs_by_month.items():
        archive_name = f"logs_{month}.tar.gz"
        archive_path = VAULT_LOGS / archive_name

        try:
            # Append to existing archive or create new
            mode = "a:gz" if archive_path.exists() else "w:gz"

            with tarfile.open(archive_path, mode) as tar:
                for log_file in files:
                    tar.add(log_file, arcname=log_file.name)
                    archived_files.append(log_file)

            log.info(f"Archived {len(files)} logs to {archive_name}")

        except Exception as e:
            log.error(f"Failed to archive logs for {month}: {e}")

    # Remove archived logs from hot storage
    for log_file in archived_files:
        try:
            log_file.unlink()
            log.debug(f"Removed archived log: {log_file.name}")
        except Exception as e:
            log.warning(f"Failed to remove {log_file.name}: {e}")

    return VAULT_LOGS if archived_files else None


def create_full_backup() -> Optional[Path]:
    """Create a full compressed backup of TABERNACLE using rsync + tar."""
    if not ensure_vault_structure():
        return None

    timestamp = datetime.now().strftime("%Y-W%W")
    backup_name = f"tabernacle_{timestamp}.tar.gz"
    backup_path = VAULT_BACKUPS / backup_name

    # Skip if this week's backup already exists
    if backup_path.exists():
        log.info(f"Weekly backup already exists: {backup_name}")
        return backup_path

    try:
        log.info(f"Creating full backup: {backup_name}")

        # Create temporary staging directory
        staging = Path("/tmp/tabernacle_backup_staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()

        # rsync to staging (excluding large/unnecessary files)
        exclude_patterns = [
            '--exclude=.git',
            '--exclude=__pycache__',
            '--exclude=*.pyc',
            '--exclude=.DS_Store',
            '--exclude=venv',
            '--exclude=node_modules',
            '--exclude=RAW_ARCHIVES.migrated',
            '--exclude=.garmin_cache',
            '--exclude=.garmin_tokens',
        ]

        rsync_cmd = ['rsync', '-a', '--delete'] + exclude_patterns + [
            str(TABERNACLE) + '/',
            str(staging) + '/'
        ]

        subprocess.run(rsync_cmd, check=True, capture_output=True)

        # Create tarball
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(staging, arcname="TABERNACLE")

        # Cleanup staging
        shutil.rmtree(staging)

        size_mb = backup_path.stat().st_size / (1024 * 1024)
        log.info(f"Full backup created: {backup_name} ({size_mb:.1f}MB)")

        return backup_path

    except Exception as e:
        log.error(f"Failed to create full backup: {e}")
        # Cleanup on failure
        if staging.exists():
            shutil.rmtree(staging)
        return None


# ============================================================================
# RETENTION ENFORCEMENT
# ============================================================================

def cleanup_old_snapshots() -> int:
    """Remove snapshots older than retention period."""
    if not is_seagate_mounted():
        return 0

    cutoff = datetime.now() - timedelta(days=RETENTION_SNAPSHOTS_DAYS)
    removed = 0

    for snapshot in VAULT_SNAPSHOTS.glob("nexus_*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(snapshot.stat().st_mtime)
            if mtime < cutoff:
                snapshot.unlink()
                log.info(f"Removed old snapshot: {snapshot.name}")
                removed += 1
        except Exception as e:
            log.warning(f"Failed to remove {snapshot.name}: {e}")

    return removed


def cleanup_old_backups() -> int:
    """Remove backups older than retention period."""
    if not is_seagate_mounted():
        return 0

    cutoff = datetime.now() - timedelta(weeks=RETENTION_BACKUPS_WEEKS)
    removed = 0

    for backup in VAULT_BACKUPS.glob("tabernacle_*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(backup.stat().st_mtime)
            if mtime < cutoff:
                backup.unlink()
                log.info(f"Removed old backup: {backup.name}")
                removed += 1
        except Exception as e:
            log.warning(f"Failed to remove {backup.name}: {e}")

    return removed


def cleanup_old_logs() -> int:
    """Remove log archives older than retention period."""
    if not is_seagate_mounted():
        return 0

    cutoff = datetime.now() - timedelta(days=RETENTION_LOGS_MONTHS * 30)
    removed = 0

    for log_archive in VAULT_LOGS.glob("logs_*.tar.gz"):
        try:
            mtime = datetime.fromtimestamp(log_archive.stat().st_mtime)
            if mtime < cutoff:
                log_archive.unlink()
                log.info(f"Removed old log archive: {log_archive.name}")
                removed += 1
        except Exception as e:
            log.warning(f"Failed to remove {log_archive.name}: {e}")

    return removed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_daily() -> Dict:
    """Run daily archival tasks."""
    log.info("=" * 50)
    log.info("COLD ARCHIVER - Daily Run")
    log.info("=" * 50)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mounted": False,
        "snapshot": None,
        "logs_archived": False,
        "cleanup": {"snapshots": 0, "backups": 0, "logs": 0},
        "errors": []
    }

    # Check mount
    if not is_seagate_mounted():
        results["errors"].append("Seagate drive not mounted")
        log.error("Seagate drive not mounted - aborting")
        return results

    results["mounted"] = True

    # Daily snapshot
    snapshot = create_nexus_snapshot()
    if snapshot:
        results["snapshot"] = str(snapshot)
    else:
        results["errors"].append("Failed to create snapshot")

    # Archive old logs
    if archive_old_logs():
        results["logs_archived"] = True

    # Weekly backup (only on Sundays)
    if datetime.now().weekday() == 6:  # Sunday
        backup = create_full_backup()
        if backup:
            results["backup"] = str(backup)

    # Cleanup
    results["cleanup"]["snapshots"] = cleanup_old_snapshots()
    results["cleanup"]["backups"] = cleanup_old_backups()
    results["cleanup"]["logs"] = cleanup_old_logs()

    # Update state
    state = load_state()
    state["last_snapshot"] = results["snapshot"]
    state["last_cleanup"] = datetime.now().isoformat()
    state["stats"]["total_snapshots"] += 1 if results["snapshot"] else 0
    save_state(state)

    # Summary
    vault_stats = get_vault_stats()
    log.info("-" * 50)
    log.info("SUMMARY:")
    log.info(f"  Snapshot: {'OK' if results['snapshot'] else 'FAILED'}")
    log.info(f"  Logs archived: {results['logs_archived']}")
    log.info(f"  Cleanup: {sum(results['cleanup'].values())} items removed")
    log.info(f"  Vault usage: {vault_stats.get('used_gb', '?')}GB / {vault_stats.get('total_gb', '?')}GB")
    log.info("=" * 50)

    return results


def run_status() -> None:
    """Print current archiver status."""
    print("\n" + "=" * 50)
    print("COLD ARCHIVER STATUS")
    print("=" * 50)

    # Mount status
    mounted = is_seagate_mounted()
    print(f"\nSeagate Mounted: {'YES' if mounted else 'NO'}")

    if mounted:
        stats = get_vault_stats()
        print(f"Vault Storage: {stats.get('used_gb', '?')}GB used / {stats.get('total_gb', '?')}GB total ({stats.get('percent_used', '?')}%)")

        # Count archives
        snapshots = list(VAULT_SNAPSHOTS.glob("nexus_*.tar.gz"))
        backups = list(VAULT_BACKUPS.glob("tabernacle_*.tar.gz"))
        logs = list(VAULT_LOGS.glob("logs_*.tar.gz"))

        print(f"\nArchives:")
        print(f"  Snapshots: {len(snapshots)} (retain {RETENTION_SNAPSHOTS_DAYS} days)")
        print(f"  Full backups: {len(backups)} (retain {RETENTION_BACKUPS_WEEKS} weeks)")
        print(f"  Log archives: {len(logs)} (retain {RETENTION_LOGS_MONTHS} months)")

        if snapshots:
            latest = max(snapshots, key=lambda p: p.stat().st_mtime)
            print(f"\nLatest snapshot: {latest.name}")

        if backups:
            latest = max(backups, key=lambda p: p.stat().st_mtime)
            print(f"Latest backup: {latest.name}")

    # State
    state = load_state()
    print(f"\nLast snapshot: {state.get('last_snapshot', 'Never')}")
    print(f"Last cleanup: {state.get('last_cleanup', 'Never')}")
    print(f"Total snapshots created: {state.get('stats', {}).get('total_snapshots', 0)}")

    print("\n" + "=" * 50)


def run_restore(target: str, destination: str) -> None:
    """Restore from a backup."""
    target_path = Path(target)
    dest_path = Path(destination)

    if not target_path.exists():
        # Try to find in vault
        for search_dir in [VAULT_SNAPSHOTS, VAULT_BACKUPS]:
            found = list(search_dir.glob(f"*{target}*"))
            if found:
                target_path = found[0]
                break

    if not target_path.exists():
        log.error(f"Archive not found: {target}")
        return

    if not dest_path.exists():
        dest_path.mkdir(parents=True)

    log.info(f"Restoring {target_path.name} to {dest_path}")

    try:
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall(dest_path)
        log.info("Restore complete!")
    except Exception as e:
        log.error(f"Restore failed: {e}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Virgil Cold Archiver - Vault Keeper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  daily     Run daily archival tasks (snapshot, log archive, cleanup)
  status    Show current archiver status
  snapshot  Create a NEXUS snapshot now
  backup    Create a full backup now
  restore   Restore from an archive
  cleanup   Run cleanup only

Examples:
  python3 cold_archiver.py daily
  python3 cold_archiver.py status
  python3 cold_archiver.py restore nexus_2026-01-17 /tmp/restore
        """
    )

    parser.add_argument("command", choices=["daily", "status", "snapshot", "backup", "restore", "cleanup"],
                       help="Command to run")
    parser.add_argument("args", nargs="*", help="Additional arguments for restore")

    args = parser.parse_args()

    if args.command == "daily":
        run_daily()
    elif args.command == "status":
        run_status()
    elif args.command == "snapshot":
        snapshot = create_nexus_snapshot()
        if snapshot:
            print(f"Created: {snapshot}")
    elif args.command == "backup":
        backup = create_full_backup()
        if backup:
            print(f"Created: {backup}")
    elif args.command == "restore":
        if len(args.args) < 2:
            print("Usage: cold_archiver.py restore <archive> <destination>")
            sys.exit(1)
        run_restore(args.args[0], args.args[1])
    elif args.command == "cleanup":
        s = cleanup_old_snapshots()
        b = cleanup_old_backups()
        l = cleanup_old_logs()
        print(f"Cleaned up: {s} snapshots, {b} backups, {l} log archives")


if __name__ == "__main__":
    main()
