#!/usr/bin/env python3
"""
HOLARCHY HEALTH CHECK
=====================
Run this to verify the entire system is operational.
Exit code 0 = healthy, non-zero = issues found.

Usage:
    python3 holarchy_health_check.py
    python3 holarchy_health_check.py --verbose
    python3 holarchy_health_check.py --json
"""

import subprocess
import json
import sys
import os
import socket
import urllib.error
import urllib.request
from datetime import datetime

# Load manifest
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "holarchy_manifest.json")

def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)

def check_daemon(launchd_id: str, script_name: str = None) -> dict:
    """Check if a daemon is running via launchctl OR pgrep."""
    try:
        # First check launchctl
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines():
            if launchd_id in line:
                parts = line.split()
                pid = parts[0] if parts[0] != "-" else None
                if pid:
                    return {"running": True, "pid": pid, "source": "launchctl"}

        # Fallback: check via pgrep for the script
        if script_name:
            result = subprocess.run(
                ["pgrep", "-f", script_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                pid = result.stdout.strip().split('\n')[0]
                return {"running": True, "pid": pid, "source": "pgrep"}

        return {"running": False, "pid": None}
    except Exception as e:
        return {"running": False, "error": str(e)}

def check_redis(host: str, port: int = 6379) -> dict:
    """Check if Redis is reachable using redis-cli."""
    try:
        result = subprocess.run(
            ["redis-cli", "-h", host, "-p", str(port), "ping"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            return {"reachable": True}
        return {"reachable": False, "error": result.stderr or "No PONG"}
    except subprocess.TimeoutExpired:
        return {"reachable": False, "error": "timeout"}
    except Exception as e:
        return {"reachable": False, "error": str(e)}

def check_l_stream(host: str = "10.0.0.50") -> dict:
    """Check L-STREAM has thoughts using redis-cli."""
    try:
        result = subprocess.run(
            ["redis-cli", "-h", host, "-p", "6379", "LLEN", "LOGOS:STREAM"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            count = int(result.stdout.strip())
            return {"count": count, "healthy": count > 0}
        return {"count": 0, "healthy": False, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"count": 0, "healthy": False, "error": "timeout"}
    except Exception as e:
        return {"count": 0, "healthy": False, "error": str(e)}

def check_http_endpoint(url: str) -> dict:
    """Check if an HTTP endpoint is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            # Any 2xx response is success
            return {"reachable": response.status < 400, "status": response.status}
    except urllib.error.HTTPError as e:
        # Even a 4xx/5xx means server is up
        return {"reachable": True, "status": e.code}
    except Exception as e:
        # Also try a simple socket check
        try:
            import socket
            host = url.split("://")[1].split("/")[0].split(":")[0]
            port = int(url.split("://")[1].split("/")[0].split(":")[1]) if ":" in url.split("://")[1].split("/")[0] else 80
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return {"reachable": True, "status": "port_open"}
        except:
            pass
        return {"reachable": False, "error": str(e)}

def check_ollama(url: str) -> dict:
    """Check if Ollama is responding."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as response:
            data = json.loads(response.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"reachable": True, "models": models}
    except Exception as e:
        return {"reachable": False, "error": str(e)}

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    as_json = "--json" in sys.argv

    manifest = load_manifest()
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall": "healthy",
        "critical_issues": [],
        "warnings": [],
        "checks": {}
    }

    # ═══════════════════════════════════════════════════════════════
    # CHECK CRITICAL DAEMONS
    # ═══════════════════════════════════════════════════════════════
    if not as_json:
        print("=" * 60)
        print("HOLARCHY HEALTH CHECK")
        print("=" * 60)
        print("\n[1/5] CRITICAL DAEMONS")

    results["checks"]["critical_daemons"] = {}
    for daemon in manifest["daemons"]["critical"]:
        status = check_daemon(daemon["launchd_id"], daemon.get("script"))
        results["checks"]["critical_daemons"][daemon["name"]] = status

        if not status.get("running"):
            results["critical_issues"].append(f"CRITICAL: {daemon['name']} not running")
            results["overall"] = "critical"

        if not as_json:
            icon = "✅" if status.get("running") else "❌"
            pid = f"PID {status['pid']}" if status.get("pid") else "NOT RUNNING"
            print(f"  {icon} {daemon['name']}: {pid}")

    # ═══════════════════════════════════════════════════════════════
    # CHECK ESSENTIAL DAEMONS
    # ═══════════════════════════════════════════════════════════════
    if not as_json:
        print("\n[2/5] ESSENTIAL DAEMONS")

    results["checks"]["essential_daemons"] = {}
    for daemon in manifest["daemons"]["essential"]:
        status = check_daemon(daemon["launchd_id"], daemon.get("script"))
        results["checks"]["essential_daemons"][daemon["name"]] = status

        if not status.get("running"):
            results["warnings"].append(f"WARNING: {daemon['name']} not running")
            if results["overall"] == "healthy":
                results["overall"] = "degraded"

        if not as_json:
            icon = "✅" if status.get("running") else "⚠️"
            pid = f"PID {status['pid']}" if status.get("pid") else "NOT RUNNING"
            print(f"  {icon} {daemon['name']}: {pid}")

    # ═══════════════════════════════════════════════════════════════
    # CHECK REDIS
    # ═══════════════════════════════════════════════════════════════
    if not as_json:
        print("\n[3/5] REDIS (Pi @ 10.0.0.50)")

    redis_status = check_redis("10.0.0.50")
    results["checks"]["redis"] = redis_status

    if not redis_status.get("reachable"):
        results["critical_issues"].append("CRITICAL: Pi Redis not reachable")
        results["overall"] = "critical"

    if not as_json:
        icon = "✅" if redis_status.get("reachable") else "❌"
        print(f"  {icon} Pi Redis: {'reachable' if redis_status.get('reachable') else 'DOWN'}")

    # Check L-STREAM
    l_stream = check_l_stream()
    results["checks"]["l_stream"] = l_stream

    if not l_stream.get("healthy"):
        results["warnings"].append(f"WARNING: L-STREAM empty ({l_stream.get('count', 0)} thoughts)")
        if results["overall"] == "healthy":
            results["overall"] = "degraded"

    if not as_json:
        icon = "✅" if l_stream.get("healthy") else "⚠️"
        print(f"  {icon} L-STREAM: {l_stream.get('count', 0)} thoughts")

    # ═══════════════════════════════════════════════════════════════
    # CHECK HTTP ENDPOINTS
    # ═══════════════════════════════════════════════════════════════
    if not as_json:
        print("\n[4/5] HTTP ENDPOINTS")

    results["checks"]["endpoints"] = {}

    librarian = check_http_endpoint("http://localhost:8080/health")
    results["checks"]["endpoints"]["librarian_http"] = librarian

    if not librarian.get("reachable"):
        results["critical_issues"].append("CRITICAL: Librarian HTTP not responding")
        results["overall"] = "critical"

    if not as_json:
        icon = "✅" if librarian.get("reachable") else "❌"
        print(f"  {icon} Librarian HTTP (8080): {'OK' if librarian.get('reachable') else 'DOWN'}")

    # ═══════════════════════════════════════════════════════════════
    # CHECK OLLAMA
    # ═══════════════════════════════════════════════════════════════
    if not as_json:
        print("\n[5/5] OLLAMA ENDPOINTS")

    results["checks"]["ollama"] = {}

    for name, url in [("local", "http://localhost:11434"),
                      ("studio", "http://10.0.0.96:11434"),
                      ("mini", "http://10.0.0.120:11434")]:
        status = check_ollama(url)
        results["checks"]["ollama"][name] = status

        if not as_json:
            icon = "✅" if status.get("reachable") else "⚠️"
            models = ", ".join(status.get("models", [])[:3]) if status.get("models") else "unreachable"
            print(f"  {icon} {name}: {models}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    if as_json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        if results["overall"] == "healthy":
            print("✅ HOLARCHY HEALTHY")
        elif results["overall"] == "degraded":
            print("⚠️  HOLARCHY DEGRADED")
            for w in results["warnings"]:
                print(f"   {w}")
        else:
            print("❌ HOLARCHY CRITICAL")
            for c in results["critical_issues"]:
                print(f"   {c}")
        print("=" * 60)

    # Exit code
    if results["overall"] == "healthy":
        sys.exit(0)
    elif results["overall"] == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
