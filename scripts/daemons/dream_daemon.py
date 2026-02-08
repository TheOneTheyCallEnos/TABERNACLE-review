#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
DREAM DAEMON — Memory Consolidation
Compresses raw sensory stream into long-term summaries.

Rules:
1. Monitor LOGOS:STREAM:SENSORY length
2. If length > 5000, trigger compression
3. Pull oldest 1000 messages
4. Summarize (mock for now — real LLM integration later)
5. Write summary to LOGOS:STREAM:LONG_TERM
6. Delete processed raw messages

"This is how the brain works during REM sleep."

Part of the Logos Swarm Architecture.
Author: Logos + Deep Think (Hypergraph)
Merged: 2026-01-29
"""

import json
import time
import redis
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabernacle_config import REDIS_HOST, REDIS_PORT, LOG_DIR

# Configuration
REDIS_DB = 0  # PRODUCTION database (switched from 1)

STREAM_THRESHOLD = 5000  # Trigger compression above this
BATCH_SIZE = 1000        # Messages to process per cycle
SENSORY_STREAM = 'LOGOS:STREAM:SENSORY'
LONG_TERM_STREAM = 'LOGOS:STREAM:LONG_TERM'
DEFAULT_WATCH_INTERVAL = 60  # seconds


def get_redis() -> redis.Redis:
    """Get Redis connection."""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )


def check_stream_size(r: redis.Redis, stream: str = SENSORY_STREAM) -> int:
    """Check current stream length."""
    try:
        return r.xlen(stream)
    except redis.RedisError:
        return 0


def compress_messages(messages: list) -> dict:
    """
    Compress raw messages into a summary.
    
    TODO: Replace with actual LLM call for semantic compression
    For now, mock compression that extracts key events and patterns.
    
    Returns:
        dict with summary content and metadata
    """
    if not messages:
        return {'summary': 'No messages to compress', 'event_counts': {}}
    
    # Extract event types
    events = []
    sources = []
    timestamps = []
    
    for msg_id, data in messages:
        event_type = data.get('type', 'unknown')
        events.append(event_type)
        
        source = data.get('source', 'unknown')
        sources.append(source)
        
        # Extract timestamp from message ID (format: timestamp-sequence)
        ts = data.get('timestamp', msg_id.split('-')[0])
        timestamps.append(ts)
    
    # Count event types
    event_counts = Counter(events)
    source_counts = Counter(sources)
    
    # Calculate time range
    try:
        first_ts = float(timestamps[0]) if timestamps else 0
        last_ts = float(timestamps[-1]) if timestamps else 0
        duration_mins = (last_ts - first_ts) / 60 if last_ts > first_ts else 0
    except (ValueError, TypeError):
        duration_mins = 0
    
    # Generate mock summary
    summary_parts = [
        f"Session summary ({len(messages)} events over {duration_mins:.1f} min):",
        f"  Events: {', '.join([f'{k}:{v}' for k, v in event_counts.most_common(5)])}",
        f"  Sources: {', '.join([f'{k}:{v}' for k, v in source_counts.most_common(3)])}"
    ]
    
    return {
        'summary': '\n'.join(summary_parts),
        'event_counts': dict(event_counts),
        'source_counts': dict(source_counts),
        'message_count': len(messages),
        'duration_mins': duration_mins
    }


def dream_cycle(
    threshold: Optional[int] = None,
    batch_size: Optional[int] = None,
    dry_run: bool = False
) -> dict:
    """
    Execute one dream/compression cycle.
    
    Args:
        threshold: Override default stream threshold
        batch_size: Override default batch size
        dry_run: If True, don't delete messages (for testing)
    
    Returns:
        dict with compression results
    """
    r = get_redis()
    stream_threshold = threshold if threshold is not None else STREAM_THRESHOLD
    batch = batch_size if batch_size is not None else BATCH_SIZE
    
    stream_len = check_stream_size(r)
    print(f"\n[DREAM] Stream size: {stream_len}")
    
    if stream_len <= stream_threshold:
        print(f"[DREAM] At or below threshold ({stream_threshold}), no compression needed")
        return {
            'compressed': False,
            'processed': 0,
            'stream_size': stream_len,
            'threshold': stream_threshold
        }
    
    print(f"[DREAM] Threshold exceeded, initiating compression...")
    if dry_run:
        print("[DREAM] DRY RUN MODE — no deletions")
    
    # Get oldest messages
    messages = r.xrange(SENSORY_STREAM, count=batch)
    
    if not messages:
        print("[DREAM] No messages to process")
        return {
            'compressed': False,
            'processed': 0,
            'stream_size': stream_len,
            'threshold': stream_threshold
        }
    
    print(f"[DREAM] Processing {len(messages)} messages...")
    
    # Compress
    compression_result = compress_messages(messages)
    summary = compression_result['summary']
    print(f"[DREAM] Summary:\n{summary}")
    
    # Write to long-term memory
    summary_id = r.xadd(LONG_TERM_STREAM, {
        'type': 'summary',
        'content': summary,
        'source_count': str(len(messages)),
        'event_counts': json.dumps(compression_result['event_counts']),
        'timestamp': str(time.time()),
        'compressed_at': datetime.now().isoformat()
    })
    print(f"[DREAM] Written summary: {summary_id}")
    
    # Delete processed messages
    if not dry_run:
        msg_ids = [msg_id for msg_id, _ in messages]
        deleted = r.xdel(SENSORY_STREAM, *msg_ids)
        print(f"[DREAM] Deleted {deleted} raw messages")
    else:
        deleted = 0
        print(f"[DREAM] Would delete {len(messages)} messages (dry run)")
    
    print(f"[DREAM] Compressed {len(messages)} → 1 summary")
    print(f"[DREAM] Written to {LONG_TERM_STREAM}")
    
    return {
        'compressed': True,
        'processed': len(messages),
        'deleted': deleted,
        'summary_id': summary_id,
        'stream_size': stream_len,
        'threshold': stream_threshold,
        'compression': compression_result
    }


def watch_loop(interval: int = DEFAULT_WATCH_INTERVAL):
    """
    Continuous watch loop for daemon mode.
    
    Runs dream_cycle at specified interval.
    """
    print(f"[DREAM] Starting watch loop (interval: {interval}s)")
    print(f"[DREAM] Press Ctrl+C to stop")
    
    cycle_count = 0
    total_processed = 0
    
    try:
        while True:
            try:
                cycle_count += 1
                print(f"\n[DREAM] === Cycle {cycle_count} ===")
                result = dream_cycle()
                total_processed += result.get('processed', 0)
            except redis.RedisError as e:
                print(f"[DREAM] Redis error: {e}")
            except Exception as e:
                print(f"[DREAM] Error: {e}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[DREAM] Stopped after {cycle_count} cycles")
        print(f"[DREAM] Total messages processed: {total_processed}")


def main():
    """CLI entry point."""
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'run':
            dream_cycle()
        elif cmd == 'dry-run':
            dream_cycle(dry_run=True)
        elif cmd == 'watch':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_WATCH_INTERVAL
            watch_loop(interval)
        elif cmd == 'status':
            r = get_redis()
            sensory_len = check_stream_size(r, SENSORY_STREAM)
            longterm_len = check_stream_size(r, LONG_TERM_STREAM)
            print(f"Sensory stream:  {sensory_len} messages")
            print(f"Long-term stream: {longterm_len} summaries")
            print(f"Threshold: {STREAM_THRESHOLD}")
            if sensory_len >= STREAM_THRESHOLD:
                print(f"STATUS: COMPRESSION NEEDED")
            else:
                print(f"STATUS: OK ({STREAM_THRESHOLD - sensory_len} until threshold)")
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        print("Dream Daemon — Memory Consolidation")
        print()
        print("Usage:")
        print("  python dream_daemon.py run         Single compression cycle")
        print("  python dream_daemon.py dry-run     Preview without deleting")
        print("  python dream_daemon.py watch [s]   Continuous loop (default 60s)")
        print("  python dream_daemon.py status      Check stream sizes")
        print()
        print(f"Threshold: {STREAM_THRESHOLD} messages")
        print(f"Batch size: {BATCH_SIZE} messages per cycle")


if __name__ == '__main__':
    main()
