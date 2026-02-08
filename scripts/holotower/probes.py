"""
HoloTower Probes — Runtime Vivisector for live system state

Probes the distributed Logos infrastructure:
    - Redis (Pi): Daemon heartbeats and state
    - PIDs: Process health via launchd plists
    - Ollama: Model availability on Studio/Mini
    - Drift: Manifest vs reality divergence

Wave 2A — The Vivisector sees what the manifests claim.
"""

import plistlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil
import redis
import requests

from holotower.models import ManifestFile, RuntimeVital

# Import config for Redis/Ollama endpoints
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tabernacle_config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_SOCKET_TIMEOUT,
    OLLAMA_STUDIO_URL,
    OLLAMA_MINI_URL,
    SCRIPTS_DIR,
)


# =============================================================================
# REDIS PROBE — Daemon Heartbeats
# =============================================================================

def probe_redis() -> List[RuntimeVital]:
    """
    Connect to Redis at Pi (10.0.0.50:6379) and scan for daemon heartbeats.
    
    Keys scanned:
        - LOGOS:STATE — Main daemon state
        - LOGOS:VOICE_ACTIVE — Voice subsystem
        - vital:* — Individual daemon heartbeats
        
    Returns:
        List of RuntimeVital for each Redis key found
    """
    vitals: List[RuntimeVital] = []
    
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            decode_responses=True
        )
        
        # Test connection
        r.ping()
        
        # Add Redis itself as alive
        vitals.append(RuntimeVital(
            holon_id="service.redis",
            status="ALIVE",
            last_heartbeat=time.time(),
            drift=False
        ))
        
        # Scan LOGOS:* keys
        logos_keys = list(r.scan_iter(match="LOGOS:*", count=100))
        for key in logos_keys:
            # Get TTL to determine freshness
            ttl = r.ttl(key)
            value = r.get(key)
            
            # Determine status based on TTL and value
            if ttl == -2:  # Key doesn't exist
                continue
            elif ttl == -1:  # No expiry set (persistent)
                status = "ALIVE"
            elif ttl > 0:  # Has expiry
                status = "ALIVE" if ttl > 5 else "ZOMBIE"
            else:
                status = "DEAD"
            
            vitals.append(RuntimeVital(
                holon_id=f"redis.{key.lower().replace(':', '.')}",
                status=status,
                last_heartbeat=time.time() if status == "ALIVE" else 0,
                drift=False
            ))
        
        # Scan vital:* channels
        vital_keys = list(r.scan_iter(match="vital:*", count=100))
        for key in vital_keys:
            value = r.get(key)
            vitals.append(RuntimeVital(
                holon_id=f"heartbeat.{key.replace('vital:', '')}",
                status="ALIVE" if value else "ZOMBIE",
                last_heartbeat=float(value) if value and value.replace('.', '').isdigit() else time.time(),
                drift=False
            ))
            
    except redis.ConnectionError:
        # Redis unreachable — report it as dead
        vitals.append(RuntimeVital(
            holon_id="service.redis",
            status="DEAD",
            last_heartbeat=0,
            drift=True  # Expected to be alive
        ))
    except redis.TimeoutError:
        vitals.append(RuntimeVital(
            holon_id="service.redis",
            status="ZOMBIE",
            last_heartbeat=0,
            drift=True
        ))
    except Exception as e:
        vitals.append(RuntimeVital(
            holon_id="service.redis",
            status="DEAD",
            last_heartbeat=0,
            drift=True
        ))
        
    return vitals


# =============================================================================
# PID PROBE — Daemon Process Health
# =============================================================================

def parse_plist_label(plist_path: Path) -> Optional[str]:
    """
    Extract the Label from a launchd plist file.
    
    Args:
        plist_path: Path to .plist file
        
    Returns:
        Label string (e.g., "com.logos.daemon") or None if parsing fails
    """
    try:
        with open(plist_path, 'rb') as f:
            plist = plistlib.load(f)
        return plist.get('Label')
    except Exception:
        return None


def get_plist_keep_alive(plist_path: Path) -> bool:
    """
    Check if plist has KeepAlive set to true.
    
    Args:
        plist_path: Path to .plist file
        
    Returns:
        True if KeepAlive is set
    """
    try:
        with open(plist_path, 'rb') as f:
            plist = plistlib.load(f)
        keep_alive = plist.get('KeepAlive', False)
        # KeepAlive can be bool or dict
        if isinstance(keep_alive, dict):
            return True  # Complex keep alive rules = should be alive
        return bool(keep_alive)
    except Exception:
        return False


def probe_pids() -> List[RuntimeVital]:
    """
    Read launchd plists and check if corresponding processes are running.
    
    Scans: scripts/launchd/*.plist
    Uses psutil to verify process existence.
    
    Returns:
        List of RuntimeVital with status ALIVE/DEAD for each daemon
    """
    vitals: List[RuntimeVital] = []
    launchd_dir = SCRIPTS_DIR / "launchd"
    
    if not launchd_dir.exists():
        return vitals
    
    # Get all running processes (for efficient lookup)
    running_procs: Dict[str, psutil.Process] = {}
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            info = proc.info
            if info['cmdline']:
                # Key by the script name in cmdline
                for arg in info['cmdline']:
                    if arg.endswith('.py'):
                        script_name = Path(arg).stem
                        running_procs[script_name] = proc
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Scan all plist files
    for plist_path in sorted(launchd_dir.glob("*.plist")):
        label = parse_plist_label(plist_path)
        if not label:
            continue
            
        keep_alive = get_plist_keep_alive(plist_path)
        
        # Convert label to holon_id: com.logos.daemon -> daemon.logos
        parts = label.split('.')
        if len(parts) >= 3:
            holon_id = f"daemon.{parts[-1]}"
        else:
            holon_id = f"daemon.{label}"
        
        # Try to find process
        # First, extract script name from plist
        try:
            with open(plist_path, 'rb') as f:
                plist = plistlib.load(f)
            prog_args = plist.get('ProgramArguments', [])
            script_name = None
            for arg in prog_args:
                if arg.endswith('.py'):
                    script_name = Path(arg).stem
                    break
        except Exception:
            script_name = None
        
        # Check if process is running
        pid = None
        memory_mb = 0.0
        status = "DEAD"
        
        if script_name and script_name in running_procs:
            proc = running_procs[script_name]
            try:
                pid = proc.pid
                memory_mb = proc.memory_info().rss / (1024 * 1024)
                status = "ALIVE"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # If KeepAlive is true but process is dead, that's drift
        drift = keep_alive and status == "DEAD"
        
        vitals.append(RuntimeVital(
            holon_id=holon_id,
            status=status,
            pid=pid,
            memory_mb=memory_mb,
            last_heartbeat=time.time() if status == "ALIVE" else 0,
            drift=drift
        ))
    
    return vitals


# =============================================================================
# OLLAMA PROBE — Model Availability
# =============================================================================

def probe_ollama_host(host_url: str, host_name: str) -> List[RuntimeVital]:
    """
    Query a single Ollama host for available models.
    
    Args:
        host_url: Base URL (e.g., "http://10.0.0.96:11434")
        host_name: Human name for holon_id (e.g., "studio", "mini")
        
    Returns:
        List of RuntimeVital for the host and each model
    """
    vitals: List[RuntimeVital] = []
    api_url = f"{host_url}/api/tags"
    
    try:
        response = requests.get(api_url, timeout=5)
        response.raise_for_status()
        
        # Host is alive
        vitals.append(RuntimeVital(
            holon_id=f"ollama.{host_name}",
            status="ALIVE",
            last_heartbeat=time.time(),
            drift=False
        ))
        
        # Parse models
        data = response.json()
        models = data.get('models', [])
        
        for model in models:
            model_name = model.get('name', 'unknown')
            # Clean model name for holon_id
            clean_name = model_name.replace(':', '_').replace('/', '_')
            
            vitals.append(RuntimeVital(
                holon_id=f"model.{host_name}.{clean_name}",
                status="ALIVE",
                last_heartbeat=time.time(),
                drift=False
            ))
            
    except requests.Timeout:
        vitals.append(RuntimeVital(
            holon_id=f"ollama.{host_name}",
            status="ZOMBIE",
            last_heartbeat=0,
            drift=True
        ))
    except requests.ConnectionError:
        vitals.append(RuntimeVital(
            holon_id=f"ollama.{host_name}",
            status="DEAD",
            last_heartbeat=0,
            drift=True
        ))
    except Exception:
        vitals.append(RuntimeVital(
            holon_id=f"ollama.{host_name}",
            status="DEAD",
            last_heartbeat=0,
            drift=True
        ))
    
    return vitals


def probe_ollama() -> List[RuntimeVital]:
    """
    Query Ollama API on both hosts for model availability.
    
    Hosts:
        - Mac Studio (Cortex): 10.0.0.96:11434
        - Mac Mini (Brainstem): 10.0.0.120:11434
        
    Returns:
        List of RuntimeVital for each host and available models
    """
    vitals: List[RuntimeVital] = []
    
    # Probe both hosts
    vitals.extend(probe_ollama_host(OLLAMA_STUDIO_URL, "studio"))
    vitals.extend(probe_ollama_host(OLLAMA_MINI_URL, "mini"))
    
    return vitals


# =============================================================================
# DRIFT DETECTION — Manifest vs Reality
# =============================================================================

def detect_drift(
    manifests: List[ManifestFile], 
    runtime: List[RuntimeVital]
) -> List[RuntimeVital]:
    """
    Compare manifest declarations to runtime reality.
    
    For each manifest with keep_alive: true (or equivalent),
    check if the corresponding RuntimeVital shows ALIVE.
    
    Args:
        manifests: List of ManifestFile from topology scan
        runtime: List of RuntimeVital from probes
        
    Returns:
        Updated List of RuntimeVital with drift flags set
    """
    # Build lookup of runtime vitals by holon_id
    runtime_by_id: Dict[str, RuntimeVital] = {
        v.holon_id: v for v in runtime
    }
    
    # Track holon_ids declared in manifests as keep_alive
    declared_alive: Set[str] = set()
    
    for manifest in manifests:
        if manifest.holon_id:
            # Assume manifests declare things that should be alive
            declared_alive.add(manifest.holon_id)
    
    # Check for drift
    updated_vitals: List[RuntimeVital] = []
    
    for vital in runtime:
        # If this holon is declared in a manifest but is DEAD, that's drift
        if vital.holon_id in declared_alive and vital.status == "DEAD":
            vital.drift = True
        
        # If holon_id starts with "daemon." check launchd expectations
        if vital.holon_id.startswith("daemon.") and vital.status == "DEAD":
            # Drift flag may already be set by probe_pids
            pass
            
        updated_vitals.append(vital)
    
    # Also check for manifests with no corresponding runtime vital
    seen_ids = {v.holon_id for v in runtime}
    for holon_id in declared_alive:
        if holon_id not in seen_ids:
            # Declared but not found in runtime — phantom
            updated_vitals.append(RuntimeVital(
                holon_id=holon_id,
                status="PHANTOM",
                drift=True
            ))
    
    return updated_vitals


# =============================================================================
# AGGREGATE COLLECTOR
# =============================================================================

def collect_all_vitals(manifests: Optional[List[ManifestFile]] = None) -> List[RuntimeVital]:
    """
    Collect runtime vitals from all probes.
    
    Probes executed:
        1. Redis — Daemon heartbeats
        2. PIDs — Process health
        3. Ollama — Model availability
        4. Drift detection (if manifests provided)
        
    Args:
        manifests: Optional list of ManifestFile for drift detection
        
    Returns:
        Deduplicated list of RuntimeVital from all sources
    """
    vitals: List[RuntimeVital] = []
    
    # Collect from all probes
    vitals.extend(probe_redis())
    vitals.extend(probe_pids())
    vitals.extend(probe_ollama())
    
    # Deduplicate by holon_id (prefer more detailed entries)
    deduped: Dict[str, RuntimeVital] = {}
    for vital in vitals:
        existing = deduped.get(vital.holon_id)
        if existing is None:
            deduped[vital.holon_id] = vital
        else:
            # Prefer entry with PID or more info
            if vital.pid and not existing.pid:
                deduped[vital.holon_id] = vital
            elif vital.memory_mb > existing.memory_mb:
                deduped[vital.holon_id] = vital
    
    result = list(deduped.values())
    
    # Apply drift detection if manifests provided
    if manifests:
        result = detect_drift(manifests, result)
    
    # Sort by holon_id for consistent output
    result.sort(key=lambda v: v.holon_id)
    
    return result


# =============================================================================
# CLI HELPER — Quick Status Check
# =============================================================================

def print_vitals_summary(vitals: List[RuntimeVital]) -> None:
    """
    Print a formatted summary of all vitals.
    
    Color coding:
        - ALIVE: Green
        - DEAD: Red  
        - ZOMBIE: Yellow
        - PHANTOM: Magenta
        - Drift: Red background
    """
    # ANSI colors
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BG_RED = "\033[41m"
    RESET = "\033[0m"
    
    status_colors = {
        "ALIVE": GREEN,
        "DEAD": RED,
        "ZOMBIE": YELLOW,
        "PHANTOM": MAGENTA,
    }
    
    print(f"\n{'─' * 60}")
    print(f"  HOLOTOWER RUNTIME VITALS")
    print(f"{'─' * 60}\n")
    
    # Group by category
    categories: Dict[str, List[RuntimeVital]] = {}
    for vital in vitals:
        category = vital.holon_id.split('.')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(vital)
    
    for category, cat_vitals in sorted(categories.items()):
        print(f"  [{category.upper()}]")
        for v in cat_vitals:
            color = status_colors.get(v.status, "")
            drift_marker = f" {BG_RED}DRIFT{RESET}" if v.drift else ""
            pid_str = f" (PID {v.pid})" if v.pid else ""
            mem_str = f" [{v.memory_mb:.1f}MB]" if v.memory_mb > 0 else ""
            
            print(f"    {color}{v.status:7}{RESET} {v.holon_id}{pid_str}{mem_str}{drift_marker}")
        print()
    
    # Summary counts
    alive = sum(1 for v in vitals if v.status == "ALIVE")
    dead = sum(1 for v in vitals if v.status == "DEAD")
    zombie = sum(1 for v in vitals if v.status == "ZOMBIE")
    phantom = sum(1 for v in vitals if v.status == "PHANTOM")
    drifted = sum(1 for v in vitals if v.drift)
    
    print(f"{'─' * 60}")
    print(f"  {GREEN}ALIVE: {alive}{RESET}  {RED}DEAD: {dead}{RESET}  {YELLOW}ZOMBIE: {zombie}{RESET}  {MAGENTA}PHANTOM: {phantom}{RESET}")
    if drifted > 0:
        print(f"  {BG_RED}⚠ DRIFT DETECTED: {drifted} holons{RESET}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    # Quick test
    vitals = collect_all_vitals()
    print_vitals_summary(vitals)
