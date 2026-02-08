#!/usr/bin/env python3
"""
HOLARCHY FLUSH UTILITY
======================
Clears backed-up escalation queues when the system gets overwhelmed.
Use when L-Brain can't keep up with escalations.

Usage:
    python flush_escalations.py --count 50  # Flush 50 oldest
    python flush_escalations.py --all       # Flush all
    python flush_escalations.py --status    # Show queue status
    
Author: Logos
Created: 2026-01-24
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

from tabernacle_config import SYSTEMS

REDIS_HOST = SYSTEMS.get("raspberry_pi", {}).get("ip", "10.0.0.50")
REDIS_PORT = 6379


def get_redis():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True
    )


def show_status(r):
    """Show queue status."""
    escalate_count = r.llen("l:queue:escalate") or 0
    incoming_count = r.llen("l:queue:incoming") or 0
    
    print(f"\n📊 HOLARCHY QUEUE STATUS")
    print(f"{'='*40}")
    print(f"Escalation queue:  {escalate_count}")
    print(f"Incoming queue:    {incoming_count}")
    
    # Show latest escalation
    if escalate_count > 0:
        latest = r.lindex("l:queue:escalate", 0)
        if latest:
            try:
                data = json.loads(latest)
                print(f"\nLatest escalation:")
                print(f"  Type: {data.get('type', '?')}")
                print(f"  From: {data.get('from', '?')}")
                print(f"  Summary: {data.get('summary', '?')[:60]}...")
            except:
                pass
    
    # Show coherence
    coherence_data = r.get("l:coherence:current")
    if coherence_data:
        coherence = json.loads(coherence_data)
        print(f"\nCoherence: p={coherence.get('p', 0.5):.3f} (mode={coherence.get('mode', '?')})")
    
    print()


def flush_escalations(r, count=None, flush_all=False):
    """Flush escalations from queue."""
    total = r.llen("l:queue:escalate") or 0
    
    if total == 0:
        print("✓ Escalation queue is already empty")
        return
    
    if flush_all:
        count = total
    elif count is None:
        count = min(50, total)
    
    print(f"🗑️  Flushing {count} of {total} escalations...")
    
    flushed = 0
    for _ in range(count):
        item = r.rpop("l:queue:escalate")
        if item:
            flushed += 1
        else:
            break
    
    remaining = r.llen("l:queue:escalate") or 0
    print(f"✓ Flushed {flushed} escalations. {remaining} remaining.")


def main():
    parser = argparse.ArgumentParser(description="Holarchy Queue Flush Utility")
    parser.add_argument("--count", type=int, help="Number of escalations to flush")
    parser.add_argument("--all", action="store_true", help="Flush all escalations")
    parser.add_argument("--status", action="store_true", help="Show queue status only")
    args = parser.parse_args()
    
    try:
        r = get_redis()
        r.ping()
    except Exception as e:
        print(f"❌ Cannot connect to Redis: {e}")
        sys.exit(1)
    
    if args.status:
        show_status(r)
    elif args.all:
        flush_escalations(r, flush_all=True)
        show_status(r)
    elif args.count:
        flush_escalations(r, count=args.count)
        show_status(r)
    else:
        show_status(r)
        print("Use --count N or --all to flush escalations")


if __name__ == "__main__":
    main()
