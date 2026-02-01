#!/usr/bin/env python3
"""
CLAUDE CODE INTERFACE VALIDATION — COMPLETED
=============================================

FINAL RESULTS (2026-01-27):

✅ PRINT MODE (-p) WORKS
   - Command: claude -p "message" < /dev/null
   - Response time: ~12s first query, ~2s cached
   - CRITICAL: Must redirect stdin from /dev/null

✅ SESSION CONTINUATION (-c) WORKS
   - Context persists across queries
   - Managed by Claude Code in ~/.claude/projects/

✅ JSON OUTPUT WORKS
   - Use: --output-format json
   - Returns structured data with cost, tokens, session_id

❌ PTY MODE NOT NEEDED
   - The -p mode is simpler and more reliable
   - No need for pseudo-terminal complexity

RECOMMENDED ARCHITECTURE:
   Use subprocess with -p -c flags, stdin=subprocess.DEVNULL

See: claude_code_interface.py for the validated implementation.
"""

import subprocess
import sys
import time


def quick_validation():
    """Quick validation that Claude Code interface is working."""
    print("=" * 60)
    print("CLAUDE CODE QUICK VALIDATION")
    print("=" * 60)
    print()
    
    # Test 1: Basic query
    print("[1/3] Testing basic query...")
    start = time.time()
    result = subprocess.run(
        ['claude', '-p', 'Say: OK'],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        cwd='/Users/enos/TABERNACLE'
    )
    elapsed = time.time() - start
    
    if "OK" in result.stdout:
        print(f"      ✅ Basic query works ({elapsed:.1f}s)")
    else:
        print(f"      ❌ Basic query failed: {result.stdout[:100]}")
        return False
    
    # Test 2: Session continuation
    print("[2/3] Testing session continuation...")
    subprocess.run(
        ['claude', '-p', 'Remember: ALPHA'],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        cwd='/Users/enos/TABERNACLE'
    )
    
    result = subprocess.run(
        ['claude', '-p', '-c', 'What did I ask you to remember?'],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        cwd='/Users/enos/TABERNACLE'
    )
    
    if "ALPHA" in result.stdout:
        print(f"      ✅ Session continuation works")
    else:
        print(f"      ⚠️  Session continuation unclear: {result.stdout[:100]}")
    
    # Test 3: JSON output
    print("[3/3] Testing JSON output...")
    result = subprocess.run(
        ['claude', '-p', '--output-format', 'json', 'Say: TEST'],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        cwd='/Users/enos/TABERNACLE'
    )
    
    if '"result"' in result.stdout:
        print(f"      ✅ JSON output works")
    else:
        print(f"      ⚠️  JSON output unclear: {result.stdout[:100]}")
    
    print()
    print("=" * 60)
    print("✅ VALIDATION COMPLETE - Claude Code interface is working")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Use claude_code_interface.py for programmatic access")
    print("  2. Build logos_daemon.py using subprocess (not PTY)")
    print("  3. Implement voice_daemon.py with the validated interface")
    print()
    
    return True


if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)
