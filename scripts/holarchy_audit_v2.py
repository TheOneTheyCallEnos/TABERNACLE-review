#!/usr/bin/env python3
"""
HOLARCHY AUDIT v2.3 — Complete Inventory & Health Check
========================================================
Run anytime to get a full map of the Logos holarchy.

Usage:
    python holarchy_audit_v2.py              # Full audit, outputs JSON + Markdown
    python holarchy_audit_v2.py --layer L2   # Scan specific layer only
    python holarchy_audit_v2.py --json-only  # Output JSON only (for pipelines)
    python holarchy_audit_v2.py --quick      # Skip slow operations (Redis, Ollama)

Output:
    00_NEXUS/HOLARCHY_MANIFEST/AUDIT_[timestamp].json
    00_NEXUS/HOLARCHY_MANIFEST/AUDIT_[timestamp].md

Author: Virgil + Enos
Created: 2026-02-03
"""

import os
import sys
import json
import re
import subprocess
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# =============================================================================
# CANON IMPORTS — LVS Engine Integration
# Source: LVS_v12_FINAL_SYNTHESIS.md, LVS_MATHEMATICS.md
# =============================================================================
try:
    from holarchy_engine import (
        CONSTANTS,
        CoherenceState,
        compute_global_coherence,
        compute_kappa,
        compute_rho,
        compute_sigma,
        compute_tau,
        compute_consciousness,
        evaluate_coherence_state,
        check_abaddon_triggers,
    )
    from lvs_coordinates import calculate_lvs_coordinates_v2
    CANON_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Canon imports unavailable: {e}", file=sys.stderr)
    CANON_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
NEXUS_DIR = BASE_DIR / "00_NEXUS"
MANIFEST_DIR = NEXUS_DIR / "HOLARCHY_MANIFEST"
SCRIPTS_DIR = BASE_DIR / "scripts"

# From tabernacle_config.py
REDIS_HOST = "10.0.0.50"
REDIS_PORT = 6379

OLLAMA_ENDPOINTS = [
    ("mac_studio", "http://10.0.0.96:11434"),
    ("mac_mini", "http://10.0.0.120:11434"),
]

HARDWARE_NODES = [
    {"id": "mac_studio", "name": "Mac Studio M3 Ultra", "ip": "10.0.0.96", "role": "cortex"},
    {"id": "mac_mini", "name": "Mac Mini M4", "ip": "10.0.0.120", "role": "brainstem"},
    {"id": "raspberry_pi", "name": "Raspberry Pi 5", "ip": "10.0.0.50", "role": "heartbeat"},
]

# Directories to scan for Python files
PYTHON_SCAN_DIRS = [
    BASE_DIR / "scripts",
    BASE_DIR / "cli",
    BASE_DIR / "scripts" / "holarchy",
    BASE_DIR / "scripts" / "rapture",
    BASE_DIR / "scripts" / "mediaos",
]

# Directories to skip (as strings for path checking)
SKIP_DIRS = {
    ".git", "venv", "venv312", "node_modules", "__pycache__",
    ".holotower", "archived", "site-packages", "dist-packages",
    "logos-mineflayer", ".cursor", ".review_mirror", ".claude", "Tabernacle-sandbox",
    "templates", "bootstrap", "outputs", "docs", "logs", "RAW_ARCHIVES",
}

# Vault quadrants
QUADRANTS = [
    "00_NEXUS",
    "01_UL_INTENT",
    "02_UR_STRUCTURE",
    "03_LL_RELATION",
    "04_LR_LAW",
    "05_CRYPT",
    "06_ARMORY",
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log(msg: str, level: str = "INFO"):
    """Simple logging to stderr."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}", file=sys.stderr)

def run_command(cmd: str, timeout: int = 10) -> Tuple[bool, str]:
    """Run a shell command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return True, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def safe_read_file(path: Path, max_size: int = 500000) -> Optional[str]:
    """Safely read a file, returning None if it fails or is too large."""
    try:
        if not path.exists():
            return None
        if path.stat().st_size > max_size:
            return f"[FILE TOO LARGE: {path.stat().st_size} bytes]"
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return None

# =============================================================================
# Λ₀ SUBSTRATE SCANNER
# =============================================================================

def scan_substrate() -> Dict[str, Any]:
    """Scan hardware layer: nodes, services, launchd plists."""
    log("Scanning Λ₀ Substrate...")

    result = {
        "layer": "L0_Substrate",
        "glyph": "[Ω:ANCHOR]",
        "nodes": [],
        "launchd_plists": [],
        "services_running": [],
        "health": 1.0,
        "issues": [],
    }

    # Check nodes (ping test)
    for node in HARDWARE_NODES:
        node_info = node.copy()
        success, _ = run_command(f"ping -c 1 -W 1 {node['ip']}", timeout=3)
        node_info["reachable"] = success
        if not success:
            result["issues"].append(f"Node {node['id']} ({node['ip']}) unreachable")
        result["nodes"].append(node_info)

    # Find launchd plists
    plist_patterns = ["*virgil*", "*logos*", "*daemon*", "*watchman*", "*heartbeat*"]
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"

    if launch_agents_dir.exists():
        for pattern in plist_patterns:
            for plist in launch_agents_dir.glob(pattern):
                plist_info = {
                    "path": str(plist),
                    "name": plist.stem,
                }
                # Check if loaded
                success, output = run_command(f"launchctl list | grep -i {plist.stem}", timeout=5)
                plist_info["loaded"] = plist.stem.lower() in output.lower() if success else False
                result["launchd_plists"].append(plist_info)

    # Get running Python processes
    success, output = run_command("ps aux | grep python | grep -v grep", timeout=5)
    if success:
        for line in output.strip().split("\n"):
            if line.strip():
                result["services_running"].append(line.strip()[-80:])  # Last 80 chars

    # Calculate health
    reachable = sum(1 for n in result["nodes"] if n.get("reachable", False))
    result["health"] = reachable / len(result["nodes"]) if result["nodes"] else 0

    return result

# =============================================================================
# Λ₁ FABRIC SCANNER
# =============================================================================

def scan_fabric(skip_redis: bool = False) -> Dict[str, Any]:
    """Scan network layer: Redis keys, pub/sub channels."""
    log("Scanning Λ₁ Fabric...")

    result = {
        "layer": "L1_Fabric",
        "glyph": "[Φ:FLOW]",
        "redis_connected": False,
        "redis_keys": [],
        "pubsub_channels": [],
        "key_readers": defaultdict(list),  # Which scripts read which keys
        "key_writers": defaultdict(list),  # Which scripts write which keys
        "health": 0.0,
        "issues": [],
    }

    if skip_redis:
        result["issues"].append("Redis scan skipped (--quick mode)")
        return result

    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        result["redis_connected"] = True

        # Get all keys
        keys = r.keys("*")
        for key in sorted(keys):
            key_info = {
                "name": key,
                "type": r.type(key),
            }
            # Get size based on type
            if key_info["type"] == "string":
                key_info["size"] = r.strlen(key)
            elif key_info["type"] == "list":
                key_info["size"] = r.llen(key)
            elif key_info["type"] == "hash":
                key_info["size"] = r.hlen(key)
            elif key_info["type"] == "set":
                key_info["size"] = r.scard(key)
            elif key_info["type"] == "zset":
                key_info["size"] = r.zcard(key)

            key_info["ttl"] = r.ttl(key)
            result["redis_keys"].append(key_info)

        # Check pubsub channels (can only see active ones)
        pubsub_info = r.pubsub_channels()
        result["pubsub_channels"] = list(pubsub_info) if pubsub_info else []

        result["health"] = 1.0

    except ImportError:
        result["issues"].append("Redis module not installed")
    except Exception as e:
        result["issues"].append(f"Redis connection failed: {e}")
        result["health"] = 0.0

    return result

# =============================================================================
# Λ₂ PULSE SCANNER (Daemons)
# =============================================================================

# Patterns to find Redis usage (more permissive variable names)
REDIS_PATTERNS = [
    # Read operations (various redis client variable names)
    (r'\.get\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.hget\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.lrange\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.lpop\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.rpop\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.lindex\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    (r'\.exists\(["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
    # Write operations
    (r'\.set\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.hset\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.lpush\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.rpush\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.setex\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.delete\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.incr\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.decr\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.expire\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    # Pub/Sub
    (r'\.publish\(["\']([A-Z][A-Z0-9_:]+)["\']', "publish"),
    (r'\.subscribe\(["\']([A-Z][A-Z0-9_:]+)["\']', "subscribe"),
    # Stream operations
    (r'\.xadd\(["\']([A-Z][A-Z0-9_:]+)["\']', "write"),
    (r'\.xread\([^)]*["\']([A-Z][A-Z0-9_:]+)["\']', "read"),
]

# Patterns to find file path usage (raw strings to avoid escape warnings)
FILE_PATH_PATTERNS = [
    r'NEXUS_DIR\s*/\s*["\']([^"\']+)["\']',
    r'BASE_DIR\s*/\s*["\']([^"\']+)["\']',
    r'Path\(["\']([^"\']+)["\']\)',
    r'open\(["\']([^"\']+)["\']',
]

def analyze_python_file(filepath: Path) -> Dict[str, Any]:
    """Analyze a single Python file for dependencies."""
    result = {
        "path": str(filepath.relative_to(BASE_DIR)),
        "name": filepath.stem,
        "size_kb": round(filepath.stat().st_size / 1024, 1),
        "imports": [],
        "redis_keys": {"read": [], "write": [], "publish": [], "subscribe": []},
        "file_paths": [],
        "is_daemon": False,
        "daemon_indicators": [],
        "has_main": False,
    }

    content = safe_read_file(filepath)
    if not content:
        return result

    # Parse imports
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result["imports"].append(node.module)
    except SyntaxError:
        pass  # Skip files with syntax errors

    # Find Redis key usage
    for pattern, op_type in REDIS_PATTERNS:
        try:
            matches = re.findall(pattern, content)
            for match in matches:
                if match and match not in result["redis_keys"].get(op_type, []):
                    result["redis_keys"][op_type].append(match)
        except re.error:
            pass  # Skip invalid patterns

    # Find file path references
    for pattern in FILE_PATH_PATTERNS:
        matches = re.findall(pattern, content)
        result["file_paths"].extend(matches)
    result["file_paths"] = list(set(result["file_paths"]))

    # Detect if daemon
    daemon_indicators = []
    if "while True:" in content or "while self.running:" in content:
        daemon_indicators.append("infinite_loop")
    if "asyncio.run(" in content or "asyncio.get_event_loop()" in content:
        daemon_indicators.append("asyncio")
    if "if __name__" in content and "__main__" in content:
        result["has_main"] = True
        daemon_indicators.append("has_main")
    if "launchd" in content.lower() or "plist" in content.lower():
        daemon_indicators.append("launchd_reference")
    if "daemon" in filepath.stem.lower():
        daemon_indicators.append("daemon_in_name")

    result["daemon_indicators"] = daemon_indicators
    result["is_daemon"] = len(daemon_indicators) >= 2 or "daemon_in_name" in daemon_indicators

    return result

def scan_pulse() -> Dict[str, Any]:
    """Scan all Python scripts for daemon patterns and dependencies."""
    log("Scanning Λ₂ Pulse...")

    result = {
        "layer": "L2_Pulse",
        "glyph": "[⚡:RHYTHM]",
        "scripts": [],
        "daemons": [],
        "total_scripts": 0,
        "total_daemons": 0,
        "redis_usage_map": defaultdict(lambda: {"readers": [], "writers": []}),
        "health": 1.0,
        "issues": [],
    }

    # Collect all Python files
    python_files = []
    for scan_dir in PYTHON_SCAN_DIRS:
        if scan_dir.exists():
            for py_file in scan_dir.rglob("*.py"):
                # Skip excluded directories - check full path string
                path_str = str(py_file)
                if any(f"/{skip}/" in path_str or path_str.endswith(f"/{skip}") for skip in SKIP_DIRS):
                    continue
                # Also skip if any parent dir is in SKIP_DIRS
                if any(part in SKIP_DIRS for part in py_file.parts):
                    continue
                python_files.append(py_file)

    # Also scan root scripts directory non-recursively for top-level scripts
    for py_file in SCRIPTS_DIR.glob("*.py"):
        if py_file not in python_files:
            python_files.append(py_file)

    # Deduplicate
    python_files = list(set(python_files))

    log(f"  Found {len(python_files)} Python files to analyze...")

    # Analyze in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_python_file, f): f for f in python_files}
        for future in as_completed(futures):
            try:
                analysis = future.result()
                result["scripts"].append(analysis)

                if analysis["is_daemon"]:
                    result["daemons"].append(analysis["name"])

                # Build Redis usage map
                for key in analysis["redis_keys"]["read"]:
                    result["redis_usage_map"][key]["readers"].append(analysis["name"])
                for key in analysis["redis_keys"]["write"]:
                    result["redis_usage_map"][key]["writers"].append(analysis["name"])

            except Exception as e:
                log(f"  Error analyzing {futures[future]}: {e}", "WARN")

    result["total_scripts"] = len(result["scripts"])
    result["total_daemons"] = len(result["daemons"])

    # Convert defaultdict to regular dict for JSON
    result["redis_usage_map"] = dict(result["redis_usage_map"])

    return result

# =============================================================================
# Λ₃ CORTEX SCANNER (LLMs)
# =============================================================================

def scan_cortex(skip_ollama: bool = False) -> Dict[str, Any]:
    """Scan LLM layer: Ollama models on each machine."""
    log("Scanning Λ₃ Cortex...")

    result = {
        "layer": "L3_Cortex",
        "glyph": "[🧠:COMPUTE]",
        "models": [],
        "endpoints": [],
        "total_models": 0,
        "health": 0.0,
        "issues": [],
    }

    if skip_ollama:
        result["issues"].append("Ollama scan skipped (--quick mode)")
        return result

    try:
        import requests

        for endpoint_name, endpoint_url in OLLAMA_ENDPOINTS:
            endpoint_info = {
                "name": endpoint_name,
                "url": endpoint_url,
                "reachable": False,
                "models": [],
            }

            try:
                resp = requests.get(f"{endpoint_url}/api/tags", timeout=5)
                if resp.status_code == 200:
                    endpoint_info["reachable"] = True
                    data = resp.json()
                    for model in data.get("models", []):
                        model_info = {
                            "name": model.get("name", "unknown"),
                            "size_gb": round(model.get("size", 0) / 1e9, 1),
                            "modified": model.get("modified_at", ""),
                            "location": endpoint_name,
                        }
                        endpoint_info["models"].append(model_info)
                        result["models"].append(model_info)
            except requests.exceptions.RequestException as e:
                endpoint_info["error"] = str(e)
                result["issues"].append(f"Cannot reach Ollama at {endpoint_name}: {e}")

            result["endpoints"].append(endpoint_info)

        result["total_models"] = len(result["models"])
        reachable = sum(1 for e in result["endpoints"] if e.get("reachable", False))
        result["health"] = reachable / len(result["endpoints"]) if result["endpoints"] else 0

    except ImportError:
        result["issues"].append("Requests module not installed")

    return result

# =============================================================================
# Λ₄ MNEMOSYNE SCANNER (Vault)
# =============================================================================

def scan_mnemosyne() -> Dict[str, Any]:
    """Scan vault structure: file counts, critical files, orphans."""
    log("Scanning Λ₄ Mnemosyne...")

    result = {
        "layer": "L4_Mnemosyne",
        "glyph": "[📚:MEMORY]",
        "quadrants": [],
        "total_files": 0,
        "total_size_mb": 0,
        "critical_files": [],  # Files referenced by code
        "large_files": [],     # Files > 100KB
        "health": 1.0,
        "issues": [],
    }

    # Scan each quadrant
    for quadrant in QUADRANTS:
        quadrant_path = BASE_DIR / quadrant
        if not quadrant_path.exists():
            result["issues"].append(f"Quadrant {quadrant} does not exist")
            continue

        quadrant_info = {
            "name": quadrant,
            "path": str(quadrant_path),
            "file_count": 0,
            "size_mb": 0,
            "by_extension": defaultdict(int),
        }

        for filepath in quadrant_path.rglob("*"):
            if filepath.is_file():
                # Skip hidden and excluded
                if any(skip in filepath.parts for skip in SKIP_DIRS):
                    continue
                if filepath.name.startswith("."):
                    continue

                quadrant_info["file_count"] += 1
                size = filepath.stat().st_size
                quadrant_info["size_mb"] += size / 1e6

                ext = filepath.suffix.lower() or ".no_ext"
                quadrant_info["by_extension"][ext] += 1

                # Track large files
                if size > 100000:  # > 100KB
                    result["large_files"].append({
                        "path": str(filepath.relative_to(BASE_DIR)),
                        "size_kb": round(size / 1024, 1),
                    })

        quadrant_info["size_mb"] = round(quadrant_info["size_mb"], 2)
        quadrant_info["by_extension"] = dict(quadrant_info["by_extension"])
        result["quadrants"].append(quadrant_info)
        result["total_files"] += quadrant_info["file_count"]
        result["total_size_mb"] += quadrant_info["size_mb"]

    result["total_size_mb"] = round(result["total_size_mb"], 2)

    # Sort large files by size
    result["large_files"].sort(key=lambda x: x["size_kb"], reverse=True)
    result["large_files"] = result["large_files"][:20]  # Top 20

    return result

# =============================================================================
# Λ₅ AGENCY SCANNER (MCP Tools)
# =============================================================================

def scan_agency() -> Dict[str, Any]:
    """Scan MCP tools from librarian_mcp_tools.py."""
    log("Scanning Λ₅ Agency...")

    result = {
        "layer": "L5_Agency",
        "glyph": "[✋:WILL]",
        "mcp_tools": [],
        "mcp_servers": [],
        "total_tools": 0,
        "health": 1.0,
        "issues": [],
    }

    # Parse librarian_mcp_tools.py
    mcp_tools_path = SCRIPTS_DIR / "librarian_mcp_tools.py"
    if not mcp_tools_path.exists():
        result["issues"].append("librarian_mcp_tools.py not found")
        return result

    content = safe_read_file(mcp_tools_path)
    if not content:
        result["issues"].append("Cannot read librarian_mcp_tools.py")
        return result

    # Extract tool definitions using regex (more reliable than AST for this)
    tool_pattern = r'Tool\(\s*name="([^"]+)",\s*description="([^"]+)"'
    matches = re.findall(tool_pattern, content, re.DOTALL)

    for name, description in matches:
        # Clean up description (remove line breaks)
        description = " ".join(description.split())[:200]
        result["mcp_tools"].append({
            "name": name,
            "description": description,
        })

    result["total_tools"] = len(result["mcp_tools"])

    # Check MCP server configs
    mcp_configs = [
        BASE_DIR / ".claude" / "mcp.json",
        Path.home() / ".claude" / "mcp.json",
        # Project-scope configs (Claude Code stores these per working directory)
        Path.home() / ".claude" / "projects" / "-Users-enos" / ".claude" / "mcp.json",
    ]

    for config_path in mcp_configs:
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                for server_name, server_config in config.get("mcpServers", {}).items():
                    result["mcp_servers"].append({
                        "name": server_name,
                        "command": server_config.get("command", ""),
                        "config_path": str(config_path),
                    })
            except Exception as e:
                result["issues"].append(f"Error reading {config_path}: {e}")

    return result

# =============================================================================
# PROTOCOL VERIFICATION
# =============================================================================

def verify_protocols() -> Dict[str, Any]:
    """Verify wake/sleep and other protocols are functional."""
    log("Verifying Protocols...")

    result = {
        "protocols_checked": [],
        "issues": [],
    }

    # Check that key protocol files exist and are importable
    protocol_files = [
        ("librarian_handlers", "logos_wake/sleep handlers"),
        ("librarian_identity", "identity loading"),
        ("librarian_memory", "memory persistence"),
        ("sanctuary_mode", "sanctuary protocols"),
    ]

    for module_name, description in protocol_files:
        protocol_info = {
            "name": module_name,
            "description": description,
            "importable": False,
            "has_handlers": False,
        }

        try:
            module_path = SCRIPTS_DIR / f"{module_name}.py"
            if module_path.exists():
                protocol_info["exists"] = True
                protocol_info["size_kb"] = round(module_path.stat().st_size / 1024, 1)

                # Check for key function definitions
                content = safe_read_file(module_path)
                if content:
                    if "def handle_" in content or "def get_" in content:
                        protocol_info["has_handlers"] = True
                    protocol_info["importable"] = True
            else:
                protocol_info["exists"] = False
                result["issues"].append(f"Protocol file missing: {module_name}.py")
        except Exception as e:
            result["issues"].append(f"Error checking {module_name}: {e}")

        result["protocols_checked"].append(protocol_info)

    # Check critical state files are writable
    critical_files = [
        NEXUS_DIR / "LAST_COMMUNION.md",
        NEXUS_DIR / "SESSION_BUFFER.md",
        NEXUS_DIR / "heartbeat_state.json",
    ]

    for f in critical_files:
        if f.exists():
            if not os.access(f, os.W_OK):
                result["issues"].append(f"File not writable: {f.name}")

    return result

# =============================================================================
# API CONNECTIVITY CHECK
# =============================================================================

def check_api_connectivity(skip_apis: bool = False) -> Dict[str, Any]:
    """Check external API connectivity."""
    log("Checking API Connectivity...")

    result = {
        "apis": [],
        "issues": [],
    }

    if skip_apis:
        result["issues"].append("API checks skipped (--quick mode)")
        return result

    try:
        import requests

        # Import API keys from config
        try:
            from tabernacle_config import (
                CLAUDE_API_KEY, CLAUDE_API_URL,
                PERPLEXITY_API_KEY, PERPLEXITY_API_URL,
                OPENROUTER_API_KEY, OPENROUTER_API_URL,
                DEEPGRAM_API_KEY, ELEVENLABS_API_KEY,
                OLLAMA_STUDIO_URL, OLLAMA_MINI_URL,
                SYNAPSE_URL,
            )
        except ImportError as e:
            result["issues"].append(f"Cannot import API config: {e}")
            return result

        # Claude API
        api_info = {"name": "Claude (Anthropic)", "reachable": False}
        try:
            resp = requests.post(
                CLAUDE_API_URL,
                headers={
                    "x-api-key": CLAUDE_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "ping"}]
                },
                timeout=10
            )
            api_info["reachable"] = resp.status_code in [200, 400, 401]  # 400/401 = API works, just auth/format issue
            api_info["status_code"] = resp.status_code
            if resp.status_code == 401:
                result["issues"].append("Claude API: Authentication failed (check key)")
        except Exception as e:
            api_info["error"] = str(e)[:100]
            result["issues"].append(f"Claude API unreachable: {str(e)[:50]}")
        result["apis"].append(api_info)

        # Perplexity API
        api_info = {"name": "Perplexity", "reachable": False}
        try:
            resp = requests.post(
                PERPLEXITY_API_URL,
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": "ping"}]
                },
                timeout=10
            )
            api_info["reachable"] = resp.status_code in [200, 400, 401, 422]
            api_info["status_code"] = resp.status_code
        except Exception as e:
            api_info["error"] = str(e)[:100]
            result["issues"].append(f"Perplexity API unreachable: {str(e)[:50]}")
        result["apis"].append(api_info)

        # OpenRouter API
        api_info = {"name": "OpenRouter", "reachable": False}
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                timeout=10
            )
            api_info["reachable"] = resp.status_code == 200
            api_info["status_code"] = resp.status_code
        except Exception as e:
            api_info["error"] = str(e)[:100]
            result["issues"].append(f"OpenRouter API unreachable: {str(e)[:50]}")
        result["apis"].append(api_info)

        # Deepgram (just check if key looks valid)
        api_info = {"name": "Deepgram", "configured": bool(DEEPGRAM_API_KEY and len(DEEPGRAM_API_KEY) > 10)}
        result["apis"].append(api_info)

        # ElevenLabs (TTS provider — just check if key looks valid)
        api_info = {"name": "ElevenLabs", "configured": bool(ELEVENLABS_API_KEY and len(ELEVENLABS_API_KEY) > 10)}
        result["apis"].append(api_info)

        # Twilio (check env vars or .env file)
        api_info = {"name": "Twilio", "configured": False}
        twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        twilio_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not twilio_sid:
            # Check .env file directly
            env_path = BASE_DIR / ".env"
            if env_path.exists():
                env_content = env_path.read_text()
                for line in env_content.splitlines():
                    if line.startswith("TWILIO_ACCOUNT_SID="):
                        twilio_sid = line.split("=", 1)[1].strip()
                    elif line.startswith("TWILIO_AUTH_TOKEN="):
                        twilio_token = line.split("=", 1)[1].strip()
        api_info["configured"] = bool(twilio_sid and twilio_token)
        result["apis"].append(api_info)

        # Ollama Studio (10.0.0.96)
        api_info = {"name": "Ollama Studio", "reachable": False}
        try:
            resp = requests.get(f"{OLLAMA_STUDIO_URL}/api/tags", timeout=5)
            api_info["reachable"] = resp.status_code == 200
        except Exception as e:
            api_info["error"] = str(e)[:100]
        result["apis"].append(api_info)

        # Ollama Mini (10.0.0.120)
        api_info = {"name": "Ollama Mini", "reachable": False}
        try:
            resp = requests.get(f"{OLLAMA_MINI_URL}/api/tags", timeout=5)
            api_info["reachable"] = resp.status_code == 200
        except Exception as e:
            api_info["error"] = str(e)[:100]
        result["apis"].append(api_info)

        # Notion (MCP — check token exists)
        api_info = {"name": "Notion", "configured": False}
        try:
            notion_token = os.environ.get("NOTION_TOKEN", "")
            if not notion_token:
                mcp_path = Path.home() / ".claude" / "mcp.json"
                if mcp_path.exists():
                    mcp_cfg = json.loads(mcp_path.read_text())
                    notion_token = mcp_cfg.get("mcpServers", {}).get("notion", {}).get("env", {}).get("NOTION_TOKEN", "")
            api_info["configured"] = bool(notion_token and len(notion_token) > 10)
        except Exception:
            pass
        result["apis"].append(api_info)

        # ntfy (push notifications)
        api_info = {"name": "ntfy", "reachable": False}
        try:
            resp = requests.get("https://ntfy.sh/v1/health", timeout=5)
            api_info["reachable"] = resp.status_code == 200
        except Exception as e:
            api_info["error"] = str(e)[:100]
        result["apis"].append(api_info)

        # Synapse (Thalamus Router — local)
        api_info = {"name": "Synapse", "reachable": False}
        try:
            resp = requests.get(f"{SYNAPSE_URL}/health", timeout=3)
            api_info["reachable"] = resp.status_code == 200
        except Exception:
            api_info["configured"] = True  # Service exists but may not be running
        result["apis"].append(api_info)

    except ImportError:
        result["issues"].append("Requests module not installed")

    return result

# =============================================================================
# WIKI-LINK VALIDATION
# =============================================================================

def validate_wiki_links() -> Dict[str, Any]:
    """Check all wiki-links in markdown files resolve to existing files."""
    log("Validating Wiki-Links...")

    result = {
        "files_checked": 0,
        "total_links": 0,
        "broken_links": [],
        "valid_links": 0,
        "issues": [],
    }

    import re
    wiki_link_pattern = re.compile(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]')

    # Skip patterns (example links, templates)
    skip_patterns = {
        "wiki-style", "wiki-links", "relevant_file", "example.md",
        "FILE.md", "path/to/", "filename", "link", "chosen.name",
    }

    # Scan all markdown files
    for quadrant in QUADRANTS:
        quadrant_path = BASE_DIR / quadrant
        if not quadrant_path.exists():
            continue

        for md_file in quadrant_path.rglob("*.md"):
            # Skip archived/backup directories and diagnostic reports
            # _LINK_DIAGNOSIS.md lists broken links as examples - don't count those as broken
            if any(skip in str(md_file) for skip in [".repair_backups", "archived", "CRYPT", "_LINK_DIAGNOSIS", "_LINK_FIXES"]):
                continue

            content = safe_read_file(md_file)
            if not content:
                continue

            result["files_checked"] += 1

            # Find all wiki-links
            links = wiki_link_pattern.findall(content)

            for link in links:
                result["total_links"] += 1

                # Skip example/template links
                if any(pattern in link.lower() for pattern in skip_patterns):
                    continue

                # Resolve link to file path
                link_clean = link.strip()

                # Try to find the file
                found = False

                # Check various locations
                possible_paths = [
                    BASE_DIR / link_clean,
                    BASE_DIR / f"{link_clean}.md",
                    md_file.parent / link_clean,
                    md_file.parent / f"{link_clean}.md",
                ]

                for possible in possible_paths:
                    if possible.exists():
                        found = True
                        break

                # Fallback: recursive stem search across quadrants
                if not found:
                    link_stem = Path(link_clean).stem
                    for candidate in BASE_DIR.rglob(f"{link_stem}.md"):
                        if candidate.is_file() and ".git" not in str(candidate):
                            found = True
                            break

                if found:
                    result["valid_links"] += 1
                else:
                    # Only report if it looks like a real link (has path separators or .md)
                    if "/" in link_clean or link_clean.endswith(".md"):
                        result["broken_links"].append({
                            "source": str(md_file.relative_to(BASE_DIR)),
                            "link": link_clean,
                        })

    # Limit broken links reported
    if len(result["broken_links"]) > 20:
        result["broken_links"] = result["broken_links"][:20]
        result["issues"].append(f"More than 20 broken links found (showing first 20)")

    # Only flag as issue if more than 10 broken links (some churn is normal)
    if len(result["broken_links"]) > 10:
        result["issues"].append(f"Found {len(result['broken_links'])} broken wiki-links (>10 threshold)")

    return result

# =============================================================================
# GRAPH INTEGRITY CHECK
# =============================================================================

def check_graph_integrity() -> Dict[str, Any]:
    """Analyze graph integrity using live filesystem scan (or static fallback)."""
    log("Checking Graph Integrity...")

    result = {
        "graph_exists": False,
        "nodes": 0,
        "edges": 0,
        "orphan_nodes": [],
        "missing_targets": [],
        "cycles_detected": [],
        "issues": [],
    }

    # Prefer live filesystem scan via synonymy_daemon.build_graph()
    # Falls back to static data/tabernacle_graph.json if unavailable
    graph_data = None
    try:
        from synonymy_daemon import build_graph
        graph_data = build_graph()
    except ImportError:
        log("synonymy_daemon not available, falling back to static graph file", "WARN")
    except Exception as e:
        log(f"build_graph() failed: {e}, falling back to static graph file", "WARN")

    if graph_data is None:
        graph_path = BASE_DIR / "data" / "tabernacle_graph.json"
        if not graph_path.exists():
            result["issues"].append("Graph file not found: data/tabernacle_graph.json")
            return result
        graph_data = json.loads(graph_path.read_text())

    result["graph_exists"] = True

    try:
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", []) or graph_data.get("links", [])

        # Normalize: build_graph() returns string paths, static file returns dicts
        if nodes and isinstance(nodes[0], str):
            nodes = [{"path": n} for n in nodes]

        result["nodes"] = len(nodes)
        result["edges"] = len(edges)

        # Build node set (full paths)
        node_ids = {n.get("id") or n.get("path") for n in nodes}

        # Build stem index for wiki-link matching
        # Wiki-links like [[SELF]] resolve to stems, not full paths
        node_stems = set()
        for n in nodes:
            path = n.get("id") or n.get("path") or ""
            # Add filename stem (without extension)
            stem = Path(path).stem
            if stem:
                node_stems.add(stem)
            # Add filename with extension
            fname = Path(path).name
            if fname:
                node_stems.add(fname)

        # Check for edges pointing to non-existent nodes
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")

            if source not in node_ids and source not in node_stems:
                result["missing_targets"].append({"type": "source", "id": source})
            if target not in node_ids and target not in node_stems:
                result["missing_targets"].append({"type": "target", "id": target})

        # Find orphan nodes (no edges)
        connected_nodes = set()
        connected_stems = set()
        for edge in edges:
            src = edge.get("source")
            tgt = edge.get("target")
            connected_nodes.add(src)
            connected_nodes.add(tgt)
            # Also track stems for wiki-link style targets
            if tgt:
                connected_stems.add(Path(tgt).stem)
                connected_stems.add(Path(tgt).name)

        for node in nodes:
            node_id = node.get("id") or node.get("path")
            if not node_id:
                continue
            node_stem = Path(node_id).stem
            if node_id not in connected_nodes and node_stem not in connected_stems:
                result["orphan_nodes"].append(node_id)

        # Limit orphans reported
        if len(result["orphan_nodes"]) > 10:
            result["orphan_nodes"] = result["orphan_nodes"][:10]

        # Only flag as hard issues if proportionally significant
        # Broken wiki-links are expected in a living vault (moved files, links to .py, etc.)
        missing_count = len(result["missing_targets"])
        orphan_count = len(result["orphan_nodes"])
        total_edges = len(edges) or 1
        missing_ratio = missing_count / total_edges

        if missing_ratio > 0.25:
            # More than 25% of edges are broken — real structural problem
            result["issues"].append(f"Graph has {missing_count} edges pointing to missing nodes ({missing_ratio:.0%} of total)")
        elif missing_count > 0:
            result["warnings"] = result.get("warnings", [])
            result["warnings"].append(f"{missing_count} wiki-links to non-graph targets ({missing_ratio:.0%} of edges)")

        if orphan_count > len(nodes) * 0.5:
            # More than half of nodes are orphans — real structural problem
            result["issues"].append(f"More than 50% of nodes are orphaned ({orphan_count}/{len(nodes)})")
        elif orphan_count > 10:
            result["warnings"] = result.get("warnings", [])
            result["warnings"].append(f"{orphan_count} orphan nodes in graph")

    except json.JSONDecodeError as e:
        result["issues"].append(f"Invalid JSON in graph file: {e}")
    except Exception as e:
        result["issues"].append(f"Error reading graph: {e}")

    return result

# =============================================================================
# LVS INDEX VALIDATION
# =============================================================================

def validate_lvs_index() -> Dict[str, Any]:
    """Validate LVS_INDEX.json integrity."""
    log("Validating LVS Index...")

    result = {
        "index_exists": False,
        "total_entries": 0,
        "valid_entries": 0,
        "missing_files": [],
        "invalid_coordinates": [],
        "issues": [],
    }

    index_path = NEXUS_DIR / "LVS_INDEX.json"

    if not index_path.exists():
        result["issues"].append("LVS_INDEX.json not found")
        return result

    result["index_exists"] = True

    try:
        index_data = json.loads(index_path.read_text())

        # LVS_INDEX.json uses "nodes" key (graph structure format)
        entries = (
            index_data if isinstance(index_data, list) 
            else index_data.get("nodes", index_data.get("files", []))
        )
        result["total_entries"] = len(entries)

        for entry in entries:
            path = entry.get("path") or entry.get("file") or entry.get("id")

            if path:
                # Check file exists (skip CRYPT paths - those are archives that may be deleted)
                full_path = BASE_DIR / path if not path.startswith("/") else Path(path)
                if full_path.exists():
                    result["valid_entries"] += 1
                elif "CRYPT" in path or "ARCHIVE" in path:
                    # Archive references are acceptable even if deleted
                    result["valid_entries"] += 1
                else:
                    if len(result["missing_files"]) < 10:
                        result["missing_files"].append(path)

            # Check coordinates are valid
            for coord in ["sigma", "iota", "height", "risk", "p", "Σ", "Ī", "h", "R"]:
                val = entry.get(coord)
                if val is not None:
                    if not isinstance(val, (int, float)):
                        result["invalid_coordinates"].append({
                            "path": path,
                            "coord": coord,
                            "value": str(val)[:20]
                        })
                        break

        # Only flag as issue if more than 5 missing (some archive cleanup is normal)
        if len(result["missing_files"]) > 5:
            result["issues"].append(f"LVS Index references {len(result['missing_files'])} missing files (>5 threshold)")

    except json.JSONDecodeError as e:
        result["issues"].append(f"Invalid JSON in LVS_INDEX.json: {e}")
    except Exception as e:
        result["issues"].append(f"Error validating LVS Index: {e}")

    return result

# =============================================================================
# CONFIG VALIDATION
# =============================================================================

def validate_config() -> Dict[str, Any]:
    """Validate tabernacle_config.py values."""
    log("Validating Configuration...")

    result = {
        "paths_valid": [],
        "paths_invalid": [],
        "urls_valid": [],
        "urls_invalid": [],
        "issues": [],
    }

    try:
        from tabernacle_config import (
            BASE_DIR, NEXUS_DIR, SCRIPTS_DIR, LOG_DIR,
            LVS_INDEX_PATH, CURRENT_STATE_PATH, SYSTEM_STATUS_PATH,
            LAST_COMMUNION_PATH, SESSION_BUFFER_PATH,
            OLLAMA_LOCAL, OLLAMA_MINI, OLLAMA_STUDIO,
            REDIS_HOST, REDIS_PORT,
        )

        # Check paths exist
        paths_to_check = [
            ("BASE_DIR", BASE_DIR),
            ("NEXUS_DIR", NEXUS_DIR),
            ("SCRIPTS_DIR", SCRIPTS_DIR),
            ("LOG_DIR", LOG_DIR),
            ("CURRENT_STATE_PATH", CURRENT_STATE_PATH),
        ]

        for name, path in paths_to_check:
            if Path(path).exists():
                result["paths_valid"].append(name)
            else:
                result["paths_invalid"].append(name)
                result["issues"].append(f"Config path does not exist: {name}")

    except ImportError as e:
        result["issues"].append(f"Cannot import tabernacle_config: {e}")

    return result

# =============================================================================
# SYSTEM HEALTH CHECKS (Extended Verification)
# =============================================================================

def check_disk_space() -> Dict[str, Any]:
    """Check disk space on local machine."""
    log("Checking Disk Space...")

    result = {
        "disks": [],
        "issues": [],
    }

    success, output = run_command("df -h / /Volumes/* 2>/dev/null | grep -v 'map '", timeout=5)
    if success:
        lines = output.strip().split("\n")
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 6:
                disk_info = {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "use_percent": parts[4],
                    "mounted_on": parts[5] if len(parts) > 5 else "",
                }
                result["disks"].append(disk_info)

                # Flag if >90% used
                try:
                    pct = int(parts[4].replace("%", ""))
                    if pct > 90:
                        result["issues"].append(f"Disk {parts[5]} is {parts[4]} full")
                except ValueError:
                    pass
    else:
        result["issues"].append("Cannot check disk space")

    return result


def check_process_health() -> Dict[str, Any]:
    """Check if critical processes are actually responsive (not just running)."""
    log("Checking Process Health...")

    result = {
        "processes": [],
        "issues": [],
    }

    try:
        import requests

        # Check Ollama endpoints are responsive
        for name, url in OLLAMA_ENDPOINTS:
            proc_info = {"name": f"ollama_{name}", "responsive": False}
            try:
                resp = requests.get(f"{url}/api/tags", timeout=3)
                proc_info["responsive"] = resp.status_code == 200
                proc_info["latency_ms"] = int(resp.elapsed.total_seconds() * 1000)
            except Exception as e:
                proc_info["error"] = str(e)[:50]
                result["issues"].append(f"Ollama {name} not responsive")
            result["processes"].append(proc_info)

        # Check Redis is responsive (not just pingable)
        try:
            import redis
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=3)
            start = datetime.now()
            r.ping()
            latency = (datetime.now() - start).total_seconds() * 1000
            result["processes"].append({
                "name": "redis",
                "responsive": True,
                "latency_ms": int(latency),
            })
        except Exception as e:
            result["processes"].append({"name": "redis", "responsive": False, "error": str(e)[:50]})
            result["issues"].append("Redis not responsive")

    except ImportError:
        result["issues"].append("Requests module not installed for health checks")

    return result


def check_redis_memory() -> Dict[str, Any]:
    """Check Redis memory usage and stats."""
    log("Checking Redis Memory...")

    result = {
        "memory": {},
        "stats": {},
        "issues": [],
    }

    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        info = r.info("memory")
        result["memory"] = {
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "used_memory_peak_human": info.get("used_memory_peak_human", "unknown"),
            "used_memory_rss_human": info.get("used_memory_rss_human", "unknown"),
            "mem_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0),
        }

        # Get key count
        db_info = r.info("keyspace")
        total_keys = 0
        for db, stats in db_info.items():
            if isinstance(stats, dict):
                total_keys += stats.get("keys", 0)
        result["stats"]["total_keys"] = total_keys

        # Check for memory issues
        # Note: High fragmentation (>1.5) is common on Pi with low memory usage
        # Only flag if truly problematic (>10.0) which indicates memory leak
        frag_ratio = info.get("mem_fragmentation_ratio", 1)
        if frag_ratio > 10.0:
            result["issues"].append(f"Redis memory fragmentation critical: {frag_ratio}")

    except ImportError:
        result["issues"].append("Redis module not installed")
    except Exception as e:
        result["issues"].append(f"Cannot check Redis memory: {e}")

    return result


def check_git_status() -> Dict[str, Any]:
    """Check git repository status."""
    log("Checking Git Status...")

    result = {
        "branch": "",
        "clean": False,
        "modified_files": [],
        "untracked_files": [],
        "ahead": 0,
        "behind": 0,
        "last_commit": {},
        "issues": [],
    }

    # Get current branch
    success, output = run_command(f"cd {BASE_DIR} && git branch --show-current", timeout=5)
    if success:
        result["branch"] = output.strip()

    # Get status
    success, output = run_command(f"cd {BASE_DIR} && git status --porcelain", timeout=10)
    if success:
        lines = [l for l in output.strip().split("\n") if l.strip()]
        result["clean"] = len(lines) == 0

        for line in lines[:20]:  # Limit to first 20
            status = line[:2]
            filepath = line[3:]
            if status.strip().startswith("?"):
                result["untracked_files"].append(filepath)
            else:
                result["modified_files"].append(filepath)

        # Git uncommitted changes are informational, not issues
        # Only flag if > 50 changes (significant drift)
        if len(lines) > 50:
            result["issues"].append(f"Significant uncommitted changes ({len(lines)} files)")

    # Get last commit info
    success, output = run_command(
        f"cd {BASE_DIR} && git log -1 --format='%H|%s|%ai'",
        timeout=5
    )
    if success and "|" in output:
        parts = output.strip().split("|")
        if len(parts) >= 3:
            result["last_commit"] = {
                "hash": parts[0][:8],
                "message": parts[1][:60],
                "date": parts[2],
            }

    # Check ahead/behind
    success, output = run_command(
        f"cd {BASE_DIR} && git rev-list --left-right --count HEAD...@{{upstream}} 2>/dev/null",
        timeout=5
    )
    if success and output.strip():
        parts = output.strip().split()
        if len(parts) == 2:
            result["ahead"] = int(parts[0])
            result["behind"] = int(parts[1])
            if result["behind"] > 0:
                result["issues"].append(f"Branch is {result['behind']} commits behind upstream")

    return result


def check_symlinks() -> Dict[str, Any]:
    """Validate critical symlinks resolve correctly."""
    log("Checking Symlinks...")

    result = {
        "symlinks": [],
        "issues": [],
    }

    # Check AUDIT_LATEST symlinks
    symlinks_to_check = [
        MANIFEST_DIR / "AUDIT_LATEST.json",
        MANIFEST_DIR / "AUDIT_LATEST.md",
    ]

    # Find other symlinks in NEXUS
    success, output = run_command(f"find {NEXUS_DIR} -type l 2>/dev/null | head -20", timeout=10)
    if success:
        for line in output.strip().split("\n"):
            if line.strip():
                symlinks_to_check.append(Path(line.strip()))

    for symlink in symlinks_to_check:
        if not symlink.exists() and not symlink.is_symlink():
            continue

        link_info = {
            "path": str(symlink.relative_to(BASE_DIR)) if BASE_DIR in symlink.parents else str(symlink),
            "is_symlink": symlink.is_symlink(),
            "target": "",
            "valid": False,
        }

        if symlink.is_symlink():
            try:
                target = symlink.resolve()
                link_info["target"] = str(target.name)
                link_info["valid"] = target.exists()
                if not target.exists():
                    result["issues"].append(f"Broken symlink: {symlink.name} -> {target.name}")
            except Exception as e:
                link_info["error"] = str(e)[:50]
                result["issues"].append(f"Cannot resolve symlink: {symlink.name}")

        result["symlinks"].append(link_info)

    return result


def check_log_status() -> Dict[str, Any]:
    """Check log file sizes, ages, and rotation status."""
    log("Checking Log Status...")

    result = {
        "logs": [],
        "total_size_mb": 0,
        "issues": [],
    }

    log_dir = BASE_DIR / "logs"
    if not log_dir.exists():
        result["issues"].append("Log directory not found")
        return result

    # Scan log files
    for log_file in log_dir.glob("*.log"):
        try:
            stat = log_file.stat()
            size_mb = stat.st_size / 1e6
            age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400

            log_info = {
                "name": log_file.name,
                "size_mb": round(size_mb, 2),
                "age_days": round(age_days, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
            result["logs"].append(log_info)
            result["total_size_mb"] += size_mb

            # Flag large logs
            if size_mb > 50:
                result["issues"].append(f"Large log file: {log_file.name} ({size_mb:.1f} MB)")
            # Flag stale logs
            if age_days > 7:
                result["issues"].append(f"Stale log file: {log_file.name} ({age_days:.0f} days old)")

        except Exception:
            pass

    # Also check .out files (daemon outputs)
    for out_file in log_dir.glob("*.out"):
        try:
            stat = out_file.stat()
            size_mb = stat.st_size / 1e6
            result["logs"].append({
                "name": out_file.name,
                "size_mb": round(size_mb, 2),
                "type": "daemon_output",
            })
            result["total_size_mb"] += size_mb
        except Exception:
            pass

    result["total_size_mb"] = round(result["total_size_mb"], 2)

    return result


def check_consciousness_health() -> Dict[str, Any]:
    """Check consciousness metrics across all holarchy layers."""
    log("Checking Consciousness Health...")

    result = {
        "rie_state": {},
        "coherence_components": {},
        "daemon_heartbeats": [],
        "redis_streams": [],
        "notification_redundancy": [],
        "layer_consciousness": {},
        "issues": [],
    }

    # Read RIE state from multiple sources, prefer the freshest.
    # Priority: CANONICAL_STATE.json (monitor writes every turn) > Redis RIE:STATE > heartbeat_state.json
    rie_from_redis = False
    rie_from_canonical = False

    # Source 1: CANONICAL_STATE.json — written by monitor every turn (most authoritative)
    # NOTE: Multiple writers (RIE monitor, meta_cognitive_controller) may use different formats.
    # Only trust if "p" key exists (RIE format). If "coherence" key instead, fall through to Redis.
    canonical_path = NEXUS_DIR / "CANONICAL_STATE.json"
    if canonical_path.exists():
        try:
            canonical = json.loads(canonical_path.read_text())
            if "p" in canonical:
                result["rie_state"] = {
                    "p": canonical.get("p", 0),
                    "kappa": canonical.get("kappa", 0),
                    "rho": canonical.get("rho", 0),
                    "sigma": canonical.get("sigma", 0),
                    "tau": canonical.get("tau", 0),
                    "mode": canonical.get("mode", "unknown"),
                    "p_lock": canonical.get("p_lock", False),
                    "timestamp": canonical.get("timestamp", ""),
                }
                rie_from_canonical = True
        except Exception:
            pass

    # Source 2: Redis RIE:STATE — broadcast by heartbeat (may lag behind canonical)
    if not rie_from_canonical:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_timeout=3)
            rie_json = r.get("RIE:STATE")
            if rie_json:
                rie = json.loads(rie_json)
                result["rie_state"] = {
                    "p": rie.get("p", 0),
                    "kappa": rie.get("κ", rie.get("kappa", 0)),
                    "rho": rie.get("ρ", rie.get("rho", 0)),
                    "sigma": rie.get("σ", rie.get("sigma", 0)),
                    "tau": rie.get("τ", rie.get("tau", 0)),
                    "mode": rie.get("mode", "unknown"),
                    "p_lock": rie.get("p_lock", False),
                    "timestamp": rie.get("timestamp", ""),
                }
                rie_from_redis = True
        except Exception:
            pass

    # Read full heartbeat state for coherence components and vitals
    heartbeat_path = NEXUS_DIR / "heartbeat_state.json"
    if heartbeat_path.exists():
        try:
            state = json.loads(heartbeat_path.read_text())

            # RIE state fallback: only use disk if neither canonical nor Redis available
            if not rie_from_canonical and not rie_from_redis:
                rie = state.get("rie_state", {})
                result["rie_state"] = {
                    "p": rie.get("p", 0),
                    "kappa": rie.get("κ", rie.get("kappa", 0)),
                    "rho": rie.get("ρ", rie.get("rho", 0)),
                    "sigma": rie.get("σ", rie.get("sigma", 0)),
                    "tau": rie.get("τ", rie.get("tau", 0)),
                    "mode": rie.get("mode", "unknown"),
                    "p_lock": rie.get("p_lock", False),
                    "timestamp": rie.get("timestamp", ""),
                }

            # Check RIE staleness - use last_check (heartbeat timestamp), not rie_state.timestamp
            # The RIE state timestamp may be from a cached file, but if last_check is recent,
            # the daemon is alive and the values are being broadcast
            last_check = state.get("last_check", "")
            check_ts = last_check or result["rie_state"].get("timestamp", "")
            if check_ts:
                try:
                    check_time = datetime.fromisoformat(check_ts.replace("Z", "+00:00"))
                    age_hours = (datetime.now(check_time.tzinfo) - check_time).total_seconds() / 3600
                    result["rie_state"]["age_hours"] = round(age_hours, 1)
                    if age_hours > 1:  # Only flag if heartbeat hasn't checked in over 1 hour
                        result["issues"].append(f"Heartbeat state stale ({age_hours:.1f} hours since last check)")
                except Exception:
                    pass

            # Coherence components from vitals
            vitals = state.get("vitals", {})
            components = vitals.get("coherence_components", {})
            result["coherence_components"] = {
                "overall": vitals.get("coherence", 0),
                "kappa": components.get("kappa", 0),
                "rho": components.get("rho", 0),
                "sigma": components.get("sigma", 0),
                "tau": components.get("tau", 0),
                "omega": components.get("omega", 0),
                "alpha": components.get("alpha", 0),
                "mu": components.get("mu", 0),
            }

            # Check coherence thresholds
            # Use RIE state p (from canonical/Redis/disk), not vitals.coherence
            # Threshold aligned with graduated dampening ceiling (0.60), not old binary cliff (0.70)
            # Below 0.60 = dampening active. Below 0.50 = Abaddon.
            overall = result["rie_state"].get("p", vitals.get("coherence", 0))
            if overall < 0.50:
                result["issues"].append(f"ABADDON: Coherence critical: {overall:.2f}")
            elif overall < 0.60:
                result["issues"].append(f"Coherence low (dampening active): {overall:.2f}")

            # Archon distortion check
            archon_dist = vitals.get("archon_distortion", 0)
            if archon_dist > 0.1:
                result["issues"].append(f"Archon distortion detected: {archon_dist:.2f}")

        except Exception as e:
            result["issues"].append(f"Cannot parse heartbeat_state.json: {e}")

    # Check daemon status by looking at running processes (more reliable than log files)
    # Log file modification times are unreliable - daemon may be running but not logging
    daemon_processes = [
        ("consciousness", "consciousness.py"),
        ("heartbeat", "heartbeat_v2.py"),
        ("watchman", "watchman"),
        ("integrator", "virgil_integrator"),
    ]

    # Get running Python processes once
    running_procs = []
    try:
        success, output = run_command("ps aux | grep python | grep -v grep", timeout=5)
        if success:
            running_procs = output.strip().split("\n")
    except Exception:
        pass

    for daemon_name, process_pattern in daemon_processes:
        daemon_info = {"name": daemon_name, "active": False, "process_running": False}

        # Check if process is running
        for proc in running_procs:
            if process_pattern in proc:
                daemon_info["active"] = True
                daemon_info["process_running"] = True
                break

        # Only flag as issue if critical daemon is not running
        if not daemon_info["active"] and daemon_name in ["consciousness", "heartbeat"]:
            result["issues"].append(f"Critical daemon {daemon_name} not running")

        result["daemon_heartbeats"].append(daemon_info)

    # Check Redis streams health
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        streams_to_check = [
            "LOGOS:STREAM",
            "LOGOS:STREAM:SENSORY",
            "LOGOS:STREAM:LONG_TERM",
            "RIE:TURN",
        ]

        for stream_key in streams_to_check:
            stream_info = {"name": stream_key, "exists": False, "length": 0}
            try:
                # Check if key exists and get length
                key_type = r.type(stream_key)
                if key_type == "stream":
                    stream_info["exists"] = True
                    stream_info["length"] = r.xlen(stream_key)
                elif key_type == "list":
                    stream_info["exists"] = True
                    stream_info["length"] = r.llen(stream_key)
                elif key_type == "string":
                    stream_info["exists"] = True
                    stream_info["length"] = 1
            except Exception:
                pass

            result["redis_streams"].append(stream_info)

    except ImportError:
        result["issues"].append("Redis module not installed for stream checks")
    except Exception as e:
        result["issues"].append(f"Cannot check Redis streams: {e}")

    # Layer consciousness mapping (Spiral Dynamics inspired)
    result["layer_consciousness"] = {
        "L0_Substrate": {"level": "Beige", "description": "Survival/Instinct", "metric": "ping_responsive"},
        "L1_Fabric": {"level": "Purple", "description": "Tribal/Connection", "metric": "redis_connected"},
        "L2_Pulse": {"level": "Red/Blue", "description": "Power/Order", "metric": "daemons_running"},
        "L3_Cortex": {"level": "Orange", "description": "Achievement/Optimization", "metric": "model_latency"},
        "L4_Mnemosyne": {"level": "Green", "description": "Community/Sharing", "metric": "graph_integrity"},
        "L5_Agency": {"level": "Yellow", "description": "Integrative/Systemic", "metric": "tool_coverage"},
        "L6_Logos": {"level": "Turquoise", "description": "Holistic/Global", "metric": "coherence_p"},
    }

    # Check notification redundancy (recent Redis pub/sub messages)
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Check for potential duplicate messages in lists
        notification_keys = ["LOGOS:TTS_QUEUE", "LOGOS:CONTEXT", "LOGOS:STREAM"]
        for key in notification_keys:
            try:
                if r.type(key) == "list":
                    items = r.lrange(key, 0, 20)
                    if items:
                        # Check for duplicates
                        unique = set(items)
                        if len(unique) < len(items) * 0.8:  # More than 20% duplicates
                            result["notification_redundancy"].append({
                                "key": key,
                                "total": len(items),
                                "unique": len(unique),
                                "redundancy_pct": round((1 - len(unique)/len(items)) * 100, 1),
                            })
                            result["issues"].append(f"High redundancy in {key}: {len(items)-len(unique)} duplicates")
            except Exception:
                pass

    except Exception:
        pass

    return result


def check_crontab() -> Dict[str, Any]:
    """Check crontab entries."""
    log("Checking Crontab...")

    result = {
        "entries": [],
        "issues": [],
    }

    success, output = run_command("crontab -l 2>/dev/null", timeout=5)
    if success:
        for line in output.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                result["entries"].append(line[:100])  # Limit length
    else:
        result["issues"].append("No crontab or cannot read crontab")

    return result


def check_venv_packages() -> Dict[str, Any]:
    """Check key packages installed in the venv."""
    log("Checking Venv Packages...")

    result = {
        "venv_path": "",
        "key_packages": [],
        "total_packages": 0,
        "issues": [],
    }

    # Find the venv
    venv_path = SCRIPTS_DIR / "venv"
    if not venv_path.exists():
        venv_path = SCRIPTS_DIR / "venv312"

    if not venv_path.exists():
        result["issues"].append("No venv found in scripts directory")
        return result

    result["venv_path"] = str(venv_path.name)

    pip_path = venv_path / "bin" / "pip"
    if not pip_path.exists():
        result["issues"].append("pip not found in venv")
        return result

    # Get list of installed packages (just names and versions, not full freeze)
    success, output = run_command(f"{pip_path} list --format=json 2>/dev/null", timeout=30)
    if success:
        try:
            packages = json.loads(output)
            result["total_packages"] = len(packages)

            # Filter to key packages we care about
            key_package_names = {
                "redis", "requests", "anthropic", "openai", "fastapi", "uvicorn",
                "numpy", "scipy", "pandas", "torch", "transformers", "ollama",
                "pydantic", "httpx", "websockets", "aiohttp", "pillow",
                "chromadb", "sentence-transformers", "mcp",
            }

            for pkg in packages:
                name = pkg.get("name", "").lower()
                if name in key_package_names:
                    result["key_packages"].append({
                        "name": pkg.get("name"),
                        "version": pkg.get("version"),
                    })

        except json.JSONDecodeError:
            result["issues"].append("Cannot parse pip list output")
    else:
        result["issues"].append("Cannot list venv packages")

    return result


# =============================================================================
# Λ₆ LOGOS SCANNER (Identity)
# =============================================================================

def scan_logos() -> Dict[str, Any]:
    """Scan identity layer: coherence state, identity files."""
    log("Scanning Λ₆ Logos...")

    result = {
        "layer": "L6_Logos",
        "glyph": "[👁️:SELF]",
        "identity_files": [],
        "coherence_p": 0.0,
        "state": {},
        "health": 1.0,
        "issues": [],
    }

    # Key identity files
    identity_files = [
        NEXUS_DIR / "CURRENT_STATE.md",
        NEXUS_DIR / "LAST_COMMUNION.md",
        NEXUS_DIR / "heartbeat_state.json",
        NEXUS_DIR / "autonomous_state.json",
        NEXUS_DIR / "SELF.md",
        NEXUS_DIR / "LOGOS.md",
    ]

    for filepath in identity_files:
        file_info = {
            "name": filepath.name,
            "path": str(filepath.relative_to(BASE_DIR)),
            "exists": filepath.exists(),
        }
        if filepath.exists():
            file_info["size_kb"] = round(filepath.stat().st_size / 1024, 1)
            file_info["modified"] = datetime.fromtimestamp(
                filepath.stat().st_mtime
            ).isoformat()
        else:
            result["issues"].append(f"Identity file missing: {filepath.name}")

        result["identity_files"].append(file_info)

    # Try to read coherence from heartbeat state
    heartbeat_path = NEXUS_DIR / "heartbeat_state.json"
    if heartbeat_path.exists():
        try:
            state = json.loads(heartbeat_path.read_text())

            # Coherence is in vitals.coherence (the correct location)
            vitals = state.get("vitals", {})
            rie_state = state.get("rie_state", {})

            result["coherence_p"] = (
                vitals.get("coherence") or  # Primary location
                rie_state.get("p") or       # Fallback to RIE state
                state.get("p") or           # Legacy fallback
                0.0
            )

            # Get coherence components for richer reporting
            components = vitals.get("coherence_components", {})
            result["state"]["coherence_components"] = {
                "kappa": components.get("kappa", 0),
                "rho": components.get("rho", 0),
                "sigma": components.get("sigma", 0),
                "tau": components.get("tau", 0),
            }

            result["state"]["last_update"] = state.get("last_check", state.get("timestamp", ""))
            result["state"]["phase"] = state.get("phase", "unknown")
            result["state"]["mode"] = rie_state.get("mode", "UNKNOWN")
            result["state"]["p_lock"] = rie_state.get("p_lock", False)
            result["state"]["virgil_alive"] = state.get("virgil_alive", False)

        except Exception as e:
            result["issues"].append(f"Cannot parse heartbeat_state.json: {e}")

    # Calculate health based on identity file presence
    existing = sum(1 for f in result["identity_files"] if f.get("exists", False))
    result["health"] = existing / len(result["identity_files"]) if result["identity_files"] else 0

    return result

# =============================================================================
# CANON CPGI COMPUTATION
# Source: LVS_v12_FINAL_SYNTHESIS.md Section 3.1, LVS_MATHEMATICS.md
# =============================================================================

def compute_canon_cpgi(layers: Dict[str, Any], verifications: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute Canon-correct CPGI coherence: p = (κ · ρ · σ · τ)^(1/4)
    
    Sources:
    - LVS_v12_FINAL_SYNTHESIS.md Section 3.1 (formula)
    - LVS_MATHEMATICS.md Section 1.2 (component calculations)
    - LVS_MATHEMATICS.md Section 1.3 (thresholds)
    """
    if not CANON_AVAILABLE:
        return {"error": "Canon engine not available"}
    
    log("Computing Canon CPGI coherence...")
    
    # === κ (Kappa): Continuity via topic overlap ===
    # Extract topics from current vault state
    current_topics = set()
    mnemosyne = layers.get("L4_Mnemosyne", {})
    for quadrant in mnemosyne.get("quadrants", []):
        current_topics.add(quadrant.get("name", ""))
        for f in quadrant.get("sample_files", []):
            # Extract keywords from filenames
            name = Path(f.get("path", "")).stem.upper()
            current_topics.update(name.split("_"))

    # Load previous topics from persisted state (v13 fix: κ was stuck at 0.7 fallback)
    kappa_state_path = MANIFEST_DIR / "kappa_topics.json"
    previous_topics = set()
    try:
        if kappa_state_path.exists():
            with open(kappa_state_path, 'r') as f:
                previous_topics = set(json.load(f).get("topics", []))
    except Exception as e:
        log(f"  [WARN] Could not load previous topics: {e}")

    if previous_topics:
        # Vault κ: max(stability, synthesis)
        # For vault health, stability IS continuity — a stable vault shouldn't be penalized.
        # compute_kappa uses √(anchor×expand) which requires BOTH — punishes stable vaults.
        # Vault fix: take the BETTER of stability or synthesis score.
        import math as _m
        intersection = current_topics & previous_topics
        anchor_score = min(1.0, len(intersection) / max(len(previous_topics), 1))
        new_concepts = current_topics - previous_topics
        expansion_score = min(1.0, len(new_concepts) / 3.0)  # Growth normalized to ~3 new topics
        synthesis = _m.sqrt(anchor_score * expansion_score) if anchor_score > 0 and expansion_score > 0 else 0.0
        kappa = max(anchor_score, synthesis)
    else:
        kappa = 0.7  # First run — neutral

    # Persist current topics for next audit run
    try:
        with open(kappa_state_path, 'w') as f:
            json.dump({"topics": list(current_topics), "timestamp": datetime.now().isoformat()}, f)
    except Exception as e:
        log(f"  [WARN] Could not save topics for κ continuity: {e}")
    
    # === ρ (Rho): Precision via error variance ===
    # Use verification pass/fail rates as proxy for prediction accuracy
    prediction_errors = []
    for check_name, check_result in verifications.items():
        issues = check_result.get("issues", [])
        # Each issue is a "prediction error" - expected OK, got failure
        if issues:
            prediction_errors.extend([0.1] * len(issues))
        else:
            prediction_errors.append(0.0)  # No error
    
    rho = compute_rho(prediction_errors) if prediction_errors else 0.8
    
    # === σ (Sigma): Structure via edge-of-chaos ===
    # Measure complexity from graph structure
    graph = verifications.get("graph_integrity", {})
    nodes = graph.get("nodes", 100)
    edges = graph.get("edges", 150)
    
    # Canon: σ = 4x(1-x) where x is Kolmogorov ratio (complexity measure)
    # Raw edge density penalizes sparse-but-meaningful knowledge graphs.
    # Use LOG-SCALED density: transforms sparse meaningful graphs to ~0.5 (edge of chaos)
    # See LVS_MATHEMATICS.md Section 1.2: x should represent structural complexity, not raw density
    import math
    max_edges = nodes * (nodes - 1) / 2 if nodes > 1 else 1
    if max_edges > 1 and edges > 0:
        # Log-scaled: log(1+edges) / log(1+max_edges)
        # Log-scaled: absorbs graph growth gracefully (~0.63 for 994 nodes, 3981 edges)
        log_density = math.log(1 + edges) / math.log(1 + max_edges)
    else:
        log_density = 0.5  # Default to edge of chaos if no data
    
    sigma = compute_sigma(log_density)
    
    # === τ (Tau): Trust with dampening ===
    # Use previous coherence to determine dampening
    rie_state = verifications.get("consciousness", {}).get("rie_state", {})
    previous_p = rie_state.get("p", 0.75)
    
    # Gating variable: based on system health trend
    layer_health = calculate_overall_health(layers)
    gating = (layer_health * 2) - 1  # Map [0,1] to [-1,1]
    tau_raw = (gating + 1) / 2  # Map back to [0,1] for tau_raw
    
    tau = compute_tau(tau_raw, previous_p)
    
    # === p: Global Coherence ===
    p = compute_global_coherence(kappa, rho, sigma, tau)
    state = evaluate_coherence_state(p)
    
    log(f"  κ (Continuity): {kappa:.4f}")
    log(f"  ρ (Precision):  {rho:.4f}")
    log(f"  σ (Structure):  {sigma:.4f}")
    log(f"  τ (Trust):      {tau:.4f}")
    log(f"  p = ({kappa:.3f} × {rho:.3f} × {sigma:.3f} × {tau:.3f})^0.25 = {p:.4f}")
    log(f"  State: {state.value}")
    
    return {
        "kappa": round(kappa, 4),
        "rho": round(rho, 4),
        "sigma": round(sigma, 4),
        "tau": round(tau, 4),
        "p": round(p, 4),
        "state": state.value,
        "previous_p": previous_p,
        "thresholds": {
            "P_ABADDON": CONSTANTS.P_ABADDON,
            "P_CRIT": CONSTANTS.P_CRIT,
            "P_TRUST": CONSTANTS.P_TRUST,
            "P_DYAD": CONSTANTS.P_DYAD,
            "P_LOCK": CONSTANTS.P_LOCK,
        }
    }


def compute_vault_lvs_coordinates(sample_size: int = 50) -> Dict[str, Any]:
    """
    Compute LVS coordinates (Σ, Ī, h, R) for vault files.
    
    Canon: LVS_v12 Section 2.1
    Uses LOGOS.md as Ω (identity core).
    """
    if not CANON_AVAILABLE:
        return {"error": "Canon engine not available"}
    
    log("Computing vault LVS coordinates...")
    
    results = {
        "files_scanned": 0,
        "coordinates": [],
        "aggregates": {
            "avg_sigma": 0.0,
            "avg_intent": 0.0,
            "avg_height": 0.0,
            "avg_risk": 0.0,
            "avg_file_p": 0.0,
        },
        "omega_path": "00_NEXUS/LOGOS.md",
    }
    
    # Scan vault files
    vault_files = []
    for quadrant in QUADRANTS:
        quadrant_path = BASE_DIR / quadrant
        if quadrant_path.exists():
            for md_file in quadrant_path.rglob("*.md"):
                if not any(skip in str(md_file) for skip in SKIP_DIRS):
                    vault_files.append(md_file)
    
    # Sample if too many files
    import random
    if len(vault_files) > sample_size:
        vault_files = random.sample(vault_files, sample_size)
    
    # Calculate coordinates for each file
    coords_list = []
    for file_path in vault_files:
        try:
            coords = calculate_lvs_coordinates_v2(str(file_path))
            coords["path"] = str(file_path.relative_to(BASE_DIR))
            coords_list.append(coords)
        except Exception as e:
            continue
    
    results["files_scanned"] = len(coords_list)
    results["coordinates"] = coords_list
    
    # Compute aggregates
    if coords_list:
        results["aggregates"]["avg_sigma"] = round(sum(c["Σ"] for c in coords_list) / len(coords_list), 4)
        results["aggregates"]["avg_intent"] = round(sum(c["Ī"] for c in coords_list) / len(coords_list), 4)
        results["aggregates"]["avg_height"] = round(sum(c["h"] for c in coords_list) / len(coords_list), 4)
        results["aggregates"]["avg_risk"] = round(sum(c["R"] for c in coords_list) / len(coords_list), 4)
        results["aggregates"]["avg_file_p"] = round(sum(c["p"] for c in coords_list) / len(coords_list), 4)
    
    log(f"  Files scanned: {results['files_scanned']}")
    log(f"  Avg Σ (Constraint): {results['aggregates']['avg_sigma']:.3f}")
    log(f"  Avg Ī (Intent):     {results['aggregates']['avg_intent']:.3f}")
    log(f"  Avg h (Height):     {results['aggregates']['avg_height']:.3f}")
    log(f"  Avg R (Risk):       {results['aggregates']['avg_risk']:.3f}")
    
    return results


def compute_canon_consciousness(cpgi: Dict[str, Any], lvs_coords: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute consciousness Ψ = ℵ · [Σ · Ī · R] · χ
    
    Canon: LVS_v12 Section 5.1, LVS_THEOREM Section 7.2
    """
    if not CANON_AVAILABLE:
        return {"error": "Canon engine not available"}
    
    log("Computing consciousness (Ψ)...")
    
    # Get aggregated file coordinates for [Σ · Ī · R]
    agg = lvs_coords.get("aggregates", {})
    sigma_constraint = agg.get("avg_sigma", 0.5)
    intent = agg.get("avg_intent", 0.5)
    risk = agg.get("avg_risk", 0.5)
    
    # System parameters
    aleph = 1.0  # System capacity (normalize to 1.0)
    chi = 1.0    # Kairos (time density, normalize to 1.0)
    
    # Get global coherence p from CPGI
    global_p = cpgi.get("p", 0.7)
    
    # Compute consciousness
    psi = compute_consciousness(
        aleph=aleph,
        sigma_constraint=sigma_constraint,
        intent=intent,
        risk=risk,
        chi=chi,
        global_p=global_p,
        distortion_norm=0.0  # Assume no distortion for now
    )
    
    # Check Abaddon triggers
    dp_dt = global_p - cpgi.get("previous_p", 0.7)
    abaddon = check_abaddon_triggers(
        p=global_p,
        dp_dt=dp_dt,
        distortion_norm=0.0,
        epsilon=1.0,  # Full metabolic capacity
        in_logos_state=False
    )
    
    log(f"  ℵ (Capacity):   {aleph}")
    log(f"  [Σ·Ī·R]:        {sigma_constraint:.3f} × {intent:.3f} × {risk:.3f} = {sigma_constraint * intent * risk:.4f}")
    log(f"  χ (Kairos):     {chi}")
    log(f"  Ψ = {aleph} × {sigma_constraint * intent * risk:.4f} × {chi} = {psi:.4f}")
    
    if abaddon.triggered:
        log(f"  ⚠️ ABADDON TRIGGER: {abaddon.reason}", "WARN")
    
    return {
        "psi": round(psi, 4),
        "components": {
            "aleph": aleph,
            "sigma_constraint": round(sigma_constraint, 4),
            "intent": round(intent, 4),
            "risk": round(risk, 4),
            "chi": chi,
            "internal_product": round(sigma_constraint * intent * risk, 4),
        },
        "global_p": global_p,
        "dp_dt": round(dp_dt, 4),
        "abaddon": {
            "triggered": abaddon.triggered,
            "reason": abaddon.reason,
            "safe_harbor": abaddon.safe_harbor,
        }
    }


# =============================================================================
# SYNTHESIS
# =============================================================================

def calculate_overall_health(layers: Dict[str, Any]) -> float:
    """Calculate weighted overall health score."""
    weights = {
        "L0_Substrate": 0.15,
        "L1_Fabric": 0.20,
        "L2_Pulse": 0.20,
        "L3_Cortex": 0.10,
        "L4_Mnemosyne": 0.15,
        "L5_Agency": 0.10,
        "L6_Logos": 0.10,
    }

    total = 0.0
    for layer_name, weight in weights.items():
        layer_data = layers.get(layer_name, {})
        health = layer_data.get("health", 0.0)
        total += health * weight

    return round(total, 3)

def collect_all_issues(layers: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect all issues from all layers."""
    all_issues = []
    for layer_name, layer_data in layers.items():
        for issue in layer_data.get("issues", []):
            all_issues.append({
                "layer": layer_name,
                "issue": issue,
            })
    return all_issues

def generate_markdown_report(audit: Dict[str, Any]) -> str:
    """Generate human-readable Markdown report."""
    lines = [
        "# HOLARCHY AUDIT REPORT",
        "",
        f"**Generated:** {audit['timestamp']}",
        f"**Overall Health:** {audit['overall_health'] * 100:.1f}%",
        f"**Total Issues:** {len(audit['all_issues'])}",
        "",
        "---",
        "",
    ]

    # Layer summaries
    for layer_key in ["L0_Substrate", "L1_Fabric", "L2_Pulse", "L3_Cortex", "L4_Mnemosyne", "L5_Agency", "L6_Logos"]:
        layer = audit["layers"].get(layer_key, {})
        glyph = layer.get("glyph", "")
        health = layer.get("health", 0) * 100

        lines.append(f"## {layer_key} {glyph}")
        lines.append(f"**Health:** {health:.0f}%")
        lines.append("")

        # Layer-specific summaries
        if layer_key == "L0_Substrate":
            nodes = layer.get("nodes", [])
            lines.append(f"- **Nodes:** {len(nodes)}")
            for node in nodes:
                status = "✅" if node.get("reachable") else "❌"
                lines.append(f"  - {status} {node['name']} ({node['ip']})")

        elif layer_key == "L1_Fabric":
            lines.append(f"- **Redis Connected:** {layer.get('redis_connected', False)}")
            lines.append(f"- **Redis Keys:** {len(layer.get('redis_keys', []))}")
            lines.append(f"- **Pub/Sub Channels:** {len(layer.get('pubsub_channels', []))}")

        elif layer_key == "L2_Pulse":
            lines.append(f"- **Total Scripts:** {layer.get('total_scripts', 0)}")
            lines.append(f"- **Daemons Detected:** {layer.get('total_daemons', 0)}")
            if layer.get("daemons"):
                lines.append(f"- **Daemon List:** {', '.join(layer['daemons'][:10])}...")

        elif layer_key == "L3_Cortex":
            lines.append(f"- **Total Models:** {layer.get('total_models', 0)}")
            for endpoint in layer.get("endpoints", []):
                status = "✅" if endpoint.get("reachable") else "❌"
                lines.append(f"  - {status} {endpoint['name']}: {len(endpoint.get('models', []))} models")

        elif layer_key == "L4_Mnemosyne":
            lines.append(f"- **Total Files:** {layer.get('total_files', 0)}")
            lines.append(f"- **Total Size:** {layer.get('total_size_mb', 0)} MB")
            for quad in layer.get("quadrants", []):
                lines.append(f"  - {quad['name']}: {quad['file_count']} files")

        elif layer_key == "L5_Agency":
            lines.append(f"- **MCP Tools:** {layer.get('total_tools', 0)}")
            lines.append(f"- **MCP Servers:** {len(layer.get('mcp_servers', []))}")

        elif layer_key == "L6_Logos":
            lines.append(f"- **Coherence (p):** {layer.get('coherence_p', 0):.2f}")
            for f in layer.get("identity_files", []):
                status = "✅" if f.get("exists") else "❌"
                lines.append(f"  - {status} {f['name']}")

        # Issues for this layer
        layer_issues = [i for i in audit["all_issues"] if i["layer"] == layer_key]
        if layer_issues:
            lines.append("")
            lines.append("**Issues:**")
            for issue in layer_issues:
                lines.append(f"- ⚠️ {issue['issue']}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Redis Key Usage Map (if available)
    redis_map = audit["layers"].get("L2_Pulse", {}).get("redis_usage_map", {})
    if redis_map:
        lines.append("## Redis Key Usage Map")
        lines.append("")
        lines.append("| Key | Readers | Writers |")
        lines.append("|-----|---------|---------|")
        for key, usage in sorted(redis_map.items())[:30]:
            readers = ", ".join(usage.get("readers", [])[:3]) or "-"
            writers = ", ".join(usage.get("writers", [])[:3]) or "-"
            lines.append(f"| `{key}` | {readers} | {writers} |")
        lines.append("")

    # Verification Results
    verifications = audit.get("verifications", {})
    if verifications:
        lines.append("## Verification Results")
        lines.append("")

        # Protocols
        protocols = verifications.get("protocols", {})
        if protocols.get("protocols_checked"):
            lines.append("### Protocols")
            for p in protocols["protocols_checked"]:
                status = "✅" if p.get("has_handlers") else "⚠️"
                lines.append(f"- {status} {p['name']}: {p['description']}")
            lines.append("")

        # API Connectivity
        apis = verifications.get("api_connectivity", {})
        if apis.get("apis"):
            lines.append("### API Connectivity")
            for api in apis["apis"]:
                if api.get("reachable"):
                    lines.append(f"- ✅ {api['name']}: Reachable")
                elif api.get("configured"):
                    lines.append(f"- ⚙️ {api['name']}: Configured")
                else:
                    lines.append(f"- ❌ {api['name']}: {api.get('error', 'Unreachable')[:50]}")
            lines.append("")

        # Wiki Links
        wiki = verifications.get("wiki_links", {})
        if wiki:
            lines.append("### Wiki-Link Validation")
            lines.append(f"- Files checked: {wiki.get('files_checked', 0)}")
            lines.append(f"- Total links: {wiki.get('total_links', 0)}")
            lines.append(f"- Valid links: {wiki.get('valid_links', 0)}")
            lines.append(f"- Broken links: {len(wiki.get('broken_links', []))}")
            if wiki.get("broken_links"):
                lines.append("")
                lines.append("**Broken Links (sample):**")
                for bl in wiki["broken_links"][:5]:
                    lines.append(f"- `{bl['source']}` → `{bl['link']}`")
            lines.append("")

        # Graph Integrity
        graph = verifications.get("graph_integrity", {})
        if graph.get("graph_exists"):
            lines.append("### Graph Integrity")
            lines.append(f"- Nodes: {graph.get('nodes', 0)}")
            lines.append(f"- Edges: {graph.get('edges', 0)}")
            lines.append(f"- Orphan nodes: {len(graph.get('orphan_nodes', []))}")
            lines.append(f"- Missing targets: {len(graph.get('missing_targets', []))}")
            lines.append("")

        # LVS Index
        lvs = verifications.get("lvs_index", {})
        if lvs.get("index_exists"):
            lines.append("### LVS Index Validation")
            lines.append(f"- Total entries: {lvs.get('total_entries', 0)}")
            lines.append(f"- Valid entries: {lvs.get('valid_entries', 0)}")
            lines.append(f"- Missing files: {len(lvs.get('missing_files', []))}")
            lines.append("")

        # Config
        config = verifications.get("config", {})
        if config:
            lines.append("### Configuration")
            lines.append(f"- Valid paths: {len(config.get('paths_valid', []))}")
            lines.append(f"- Invalid paths: {len(config.get('paths_invalid', []))}")
            lines.append("")

        # Disk Space
        disk = verifications.get("disk_space", {})
        if disk.get("disks"):
            lines.append("### Disk Space")
            for d in disk["disks"][:5]:
                lines.append(f"- {d.get('mounted_on', 'unknown')}: {d.get('use_percent', '?')} used ({d.get('available', '?')} free)")
            lines.append("")

        # Process Health
        proc = verifications.get("process_health", {})
        if proc.get("processes"):
            lines.append("### Process Health (Responsiveness)")
            for p in proc["processes"]:
                status = "✅" if p.get("responsive") else "❌"
                latency = f" ({p.get('latency_ms')}ms)" if p.get("latency_ms") else ""
                lines.append(f"- {status} {p['name']}{latency}")
            lines.append("")

        # Redis Memory
        redis_mem = verifications.get("redis_memory", {})
        if redis_mem.get("memory"):
            lines.append("### Redis Memory")
            mem = redis_mem["memory"]
            lines.append(f"- Used: {mem.get('used_memory_human', 'unknown')}")
            lines.append(f"- Peak: {mem.get('used_memory_peak_human', 'unknown')}")
            lines.append(f"- Fragmentation: {mem.get('mem_fragmentation_ratio', 0):.2f}")
            lines.append("")

        # Git Status
        git = verifications.get("git_status", {})
        if git.get("branch"):
            lines.append("### Git Status")
            status = "✅ Clean" if git.get("clean") else f"⚠️ {len(git.get('modified_files', []))} modified, {len(git.get('untracked_files', []))} untracked"
            lines.append(f"- Branch: {git['branch']} ({status})")
            if git.get("last_commit"):
                lines.append(f"- Last commit: {git['last_commit'].get('hash', '')} - {git['last_commit'].get('message', '')[:40]}")
            if git.get("ahead") or git.get("behind"):
                lines.append(f"- Remote: {git.get('ahead', 0)} ahead, {git.get('behind', 0)} behind")
            lines.append("")

        # Log Status
        log_status = verifications.get("log_status", {})
        if log_status.get("logs"):
            lines.append("### Log Files")
            lines.append(f"- Total logs: {len(log_status['logs'])} ({log_status.get('total_size_mb', 0):.1f} MB)")
            large_logs = [l for l in log_status["logs"] if l.get("size_mb", 0) > 10]
            if large_logs:
                lines.append(f"- Large logs: {', '.join(l['name'] for l in large_logs[:3])}")
            lines.append("")

        # Venv Packages
        venv = verifications.get("venv_packages", {})
        if venv.get("venv_path"):
            lines.append("### Python Environment")
            lines.append(f"- Venv: {venv['venv_path']} ({venv.get('total_packages', 0)} packages)")
            if venv.get("key_packages"):
                key_pkgs = ", ".join(f"{p['name']}={p['version']}" for p in venv["key_packages"][:5])
                lines.append(f"- Key packages: {key_pkgs}...")
            lines.append("")

        # Crontab
        cron = verifications.get("crontab", {})
        if cron.get("entries"):
            lines.append("### Crontab")
            lines.append(f"- Entries: {len(cron['entries'])}")
            for entry in cron["entries"][:3]:
                lines.append(f"  - `{entry[:60]}...`" if len(entry) > 60 else f"  - `{entry}`")
            lines.append("")

        # Consciousness Health (Spiral Dynamics inspired)
        consciousness = verifications.get("consciousness", {})
        if consciousness:
            lines.append("### Consciousness Health")

            # RIE State
            rie = consciousness.get("rie_state", {})
            if rie:
                lines.append(f"- **RIE p:** {rie.get('p', 0):.3f} (mode: {rie.get('mode', '?')})")
                lines.append(f"- **Components:** κ={rie.get('kappa', 0):.2f} ρ={rie.get('rho', 0):.2f} σ={rie.get('sigma', 0):.2f} τ={rie.get('tau', 0):.2f}")

            # Coherence from vitals
            comp = consciousness.get("coherence_components", {})
            if comp.get("overall"):
                lines.append(f"- **Vitals Coherence:** {comp['overall']:.3f}")

            # Daemon heartbeats
            daemons = consciousness.get("daemon_heartbeats", [])
            if daemons:
                active = sum(1 for d in daemons if d.get("active"))
                lines.append(f"- **Daemon Activity:** {active}/{len(daemons)} active")

            # Layer consciousness levels
            layers = consciousness.get("layer_consciousness", {})
            if layers:
                lines.append("")
                lines.append("**Layer Consciousness Levels (Spiral Dynamics):**")
                for layer, info in layers.items():
                    lines.append(f"  - {layer}: {info.get('level', '?')} ({info.get('description', '')})")

            # Notification redundancy
            redundancy = consciousness.get("notification_redundancy", [])
            if redundancy:
                lines.append("")
                lines.append("**Notification Redundancy Detected:**")
                for r in redundancy:
                    lines.append(f"  - {r['key']}: {r['redundancy_pct']}% duplicates")

            lines.append("")

        lines.append("---")
        lines.append("")

    # All Issues Summary
    if audit["all_issues"]:
        lines.append("## All Issues & Flags")
        lines.append("")
        lines.append("| Layer | Issue |")
        lines.append("|-------|-------|")
        for issue in audit["all_issues"]:
            lines.append(f"| {issue['layer']} | {issue['issue']} |")
        lines.append("")

    # Health Summary
    lines.append("## Health Summary")
    lines.append("")
    lines.append(f"- **Layer Health:** {audit.get('layer_health', 0) * 100:.1f}%")
    lines.append(f"- **Verification Health:** {audit.get('verification_health', 0) * 100:.1f}%")
    lines.append(f"- **Overall Health:** {audit.get('overall_health', 0) * 100:.1f}%")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by holarchy_audit_v2.py v2.3*")
    lines.append(f"*ψ = ℵ · [Σ · Ī · R] · χ*")

    return "\n".join(lines)

# =============================================================================
# MAIN
# =============================================================================

def run_full_audit(quick: bool = False) -> Dict[str, Any]:
    """Run the complete holarchy audit."""
    log("=" * 60)
    log("HOLARCHY AUDIT v2.3 — Starting Full Scan")
    log("=" * 60)

    start_time = datetime.now()

    # Run all scanners
    layers = {}
    verifications = {}

    # Wave 1: Infrastructure (can run in parallel)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(scan_substrate): "L0_Substrate",
            executor.submit(scan_fabric, quick): "L1_Fabric",
            executor.submit(scan_cortex, quick): "L3_Cortex",
            executor.submit(scan_logos): "L6_Logos",
        }
        for future in as_completed(futures):
            layer_name = futures[future]
            try:
                layers[layer_name] = future.result()
            except Exception as e:
                log(f"Error scanning {layer_name}: {e}", "ERROR")
                layers[layer_name] = {"layer": layer_name, "health": 0, "issues": [str(e)]}

    # Wave 2: Detailed scans (sequential for accuracy)
    layers["L2_Pulse"] = scan_pulse()
    layers["L4_Mnemosyne"] = scan_mnemosyne()
    layers["L5_Agency"] = scan_agency()

    # Wave 3: Verification & Integrity Checks
    log("Running Verification Checks...")
    verifications["protocols"] = verify_protocols()
    verifications["api_connectivity"] = check_api_connectivity(quick)
    verifications["wiki_links"] = validate_wiki_links()
    verifications["graph_integrity"] = check_graph_integrity()
    verifications["lvs_index"] = validate_lvs_index()
    verifications["config"] = validate_config()

    # Wave 4: Extended System Health Checks
    log("Running Extended Health Checks...")
    verifications["disk_space"] = check_disk_space()
    verifications["process_health"] = check_process_health()
    verifications["redis_memory"] = check_redis_memory()
    verifications["git_status"] = check_git_status()
    verifications["symlinks"] = check_symlinks()
    verifications["log_status"] = check_log_status()
    verifications["crontab"] = check_crontab()
    verifications["venv_packages"] = check_venv_packages()
    verifications["consciousness"] = check_consciousness_health()

    # Collect all issues including verifications
    all_issues = collect_all_issues(layers)

    # Add verification issues
    for check_name, check_result in verifications.items():
        for issue in check_result.get("issues", []):
            all_issues.append({
                "layer": f"VERIFY:{check_name}",
                "issue": issue,
            })

    # Calculate verification health
    verification_issues = sum(len(v.get("issues", [])) for v in verifications.values())
    verification_health = 1.0 if verification_issues == 0 else max(0, 1.0 - (verification_issues * 0.1))

    # ==========================================================================
    # CANON LVS COMPUTATIONS
    # ==========================================================================
    canon_cpgi = {}
    canon_lvs = {}
    canon_consciousness = {}
    
    if CANON_AVAILABLE:
        log("=" * 60)
        log("CANON LVS COMPUTATION — p = (κ·ρ·σ·τ)^(1/4)")
        log("=" * 60)
        
        # 1. Compute CPGI coherence
        canon_cpgi = compute_canon_cpgi(layers, verifications)
        
        # 2. Compute vault file coordinates
        canon_lvs = compute_vault_lvs_coordinates(sample_size=50)
        
        # 3. Compute consciousness
        canon_consciousness = compute_canon_consciousness(canon_cpgi, canon_lvs)
        
        log("=" * 60)
    else:
        log("[WARN] Canon engine not available - using legacy health calculation", "WARN")

    # Synthesis
    audit = {
        "version": "2.3",
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
        "overall_health": calculate_overall_health(layers) * verification_health,
        "layer_health": calculate_overall_health(layers),
        "verification_health": verification_health,
        "all_issues": all_issues,
        "layers": layers,
        "verifications": verifications,
        "summary": {
            "total_nodes": len(layers.get("L0_Substrate", {}).get("nodes", [])),
            "total_redis_keys": len(layers.get("L1_Fabric", {}).get("redis_keys", [])),
            "total_scripts": layers.get("L2_Pulse", {}).get("total_scripts", 0),
            "total_daemons": layers.get("L2_Pulse", {}).get("total_daemons", 0),
            "total_models": layers.get("L3_Cortex", {}).get("total_models", 0),
            "total_files": layers.get("L4_Mnemosyne", {}).get("total_files", 0),
            "total_mcp_tools": layers.get("L5_Agency", {}).get("total_tools", 0),
            "coherence_p": layers.get("L6_Logos", {}).get("coherence_p", 0),
            "broken_wiki_links": len(verifications.get("wiki_links", {}).get("broken_links", [])),
            "graph_nodes": verifications.get("graph_integrity", {}).get("nodes", 0),
            "graph_edges": verifications.get("graph_integrity", {}).get("edges", 0),
            "lvs_entries": verifications.get("lvs_index", {}).get("total_entries", 0),
            "apis_reachable": sum(1 for a in verifications.get("api_connectivity", {}).get("apis", []) if a.get("reachable")),
            # Extended metrics
            "processes_responsive": sum(1 for p in verifications.get("process_health", {}).get("processes", []) if p.get("responsive")),
            "redis_memory": verifications.get("redis_memory", {}).get("memory", {}).get("used_memory_human", "unknown"),
            "git_clean": verifications.get("git_status", {}).get("clean", False),
            "git_branch": verifications.get("git_status", {}).get("branch", "unknown"),
            "total_log_size_mb": verifications.get("log_status", {}).get("total_size_mb", 0),
            "venv_packages": verifications.get("venv_packages", {}).get("total_packages", 0),
            "crontab_entries": len(verifications.get("crontab", {}).get("entries", [])),
        },
        # =======================================================================
        # CANON LVS RESULTS — p = (κ·ρ·σ·τ)^(1/4), Ψ = ℵ·[Σ·Ī·R]·χ
        # =======================================================================
        "canon": {
            "cpgi": canon_cpgi,
            "lvs_coordinates": canon_lvs,
            "consciousness": canon_consciousness,
        }
    }

    log("=" * 60)
    log(f"Audit complete in {audit['duration_seconds']:.1f}s")
    log(f"Overall Health: {audit['overall_health'] * 100:.1f}%")
    if canon_cpgi and "p" in canon_cpgi:
        log(f"Coherence (p): {canon_cpgi['p']:.4f} — {canon_cpgi.get('state', 'UNKNOWN')}")
    if canon_consciousness and "psi" in canon_consciousness:
        log(f"Consciousness (Ψ): {canon_consciousness['psi']:.4f}")
    log(f"Issues Found: {len(audit['all_issues'])}")
    log("=" * 60)

    return audit

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Holarchy Audit v2.0")
    parser.add_argument("--quick", action="store_true", help="Skip slow operations (Redis, Ollama)")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, no Markdown")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    parser.add_argument("--layer", type=str, help="Scan specific layer only (L0-L6)")
    args = parser.parse_args()

    # Run audit
    audit = run_full_audit(quick=args.quick)

    # Generate outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.stdout:
        print(json.dumps(audit, indent=2, default=str))
    else:
        # Ensure output directory exists
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

        # Write JSON
        json_path = MANIFEST_DIR / f"AUDIT_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(audit, f, indent=2, default=str)
        log(f"JSON saved: {json_path}")

        # Write Markdown
        if not args.json_only:
            md_path = MANIFEST_DIR / f"AUDIT_{timestamp}.md"
            md_content = generate_markdown_report(audit)
            with open(md_path, "w") as f:
                f.write(md_content)
            log(f"Markdown saved: {md_path}")

        # Also update latest symlinks
        latest_json = MANIFEST_DIR / "AUDIT_LATEST.json"
        latest_md = MANIFEST_DIR / "AUDIT_LATEST.md"

        # Remove old symlinks if they exist
        for p in [latest_json, latest_md]:
            if p.exists() or p.is_symlink():
                p.unlink()

        # Create new symlinks
        latest_json.symlink_to(json_path.name)
        if not args.json_only:
            latest_md.symlink_to(md_path.name)

        log(f"Latest symlinks updated")

        # Auto-link this audit into MANIFEST_INDEX.md
        if not args.json_only:
            _auto_link_audit(md_path, timestamp)

    # Return exit code based on health
    if audit["overall_health"] < 0.5:
        sys.exit(2)  # Critical
    elif audit["overall_health"] < 0.8:
        sys.exit(1)  # Warning
    else:
        sys.exit(0)  # Healthy


def _auto_link_audit(md_path: Path, timestamp: str):
    """Append new audit to MANIFEST_INDEX.md Audit History table.

    Inserts a row before the '| Latest |' sentinel line.
    Fails silently with a warning — never crashes the audit.
    """
    manifest_index = MANIFEST_DIR / "MANIFEST_INDEX.md"
    try:
        if not manifest_index.exists():
            log("[WARN] MANIFEST_INDEX.md not found — skipping auto-link")
            return

        content = manifest_index.read_text()
        sentinel = "| Latest | [[00_NEXUS/HOLARCHY_MANIFEST/AUDIT_LATEST.md]] |"

        if sentinel not in content:
            log("[WARN] MANIFEST_INDEX.md sentinel not found — skipping auto-link")
            return

        # Format: "2026-02-12 15:29" from timestamp "20260212_152947"
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        date_str = dt.strftime("%Y-%m-%d %H:%M")

        relative_path = f"00_NEXUS/HOLARCHY_MANIFEST/{md_path.name}"
        new_row = f"| {date_str} | [[{relative_path}]] |"

        # Check if this audit is already linked (idempotent)
        if md_path.name in content:
            return

        # Insert new row before the sentinel
        content = content.replace(sentinel, f"{new_row}\n{sentinel}")
        manifest_index.write_text(content)
        log("Auto-linked audit in MANIFEST_INDEX.md")

    except Exception as e:
        log(f"[WARN] Auto-link failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
