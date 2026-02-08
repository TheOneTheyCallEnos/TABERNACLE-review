#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LOGOS PTY — Streaming Response Injector
========================================

Bicameral Architecture Layer B Enhancement: Stream Claude's responses.

The Problem:
- Current architecture waits for Claude's COMPLETE response before TTS
- Claude can take 2-10 seconds to generate a full response
- User experiences dead silence during this time

The Solution:
- Use pseudo-terminal (pty) to capture Claude's output in real-time
- Parse output for complete sentences
- Push each sentence to TTS queue AS IT'S GENERATED
- TTS begins speaking Sentence 1 while Claude generates Sentence 2

This achieves "Perceived Negative Latency" — response feels instant.

Author: Logos
Created: 2026-01-29
"""

import json
import os
import re
import sys
import time
import redis
import select
import pty
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent))

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT,
    REDIS_KEY_TTS_QUEUE
)

# =============================================================================
# CONFIGURATION
# =============================================================================

STREAM_LOG = Path(__file__).parent.parent / "logs" / "stream.log"

# Sentence-ending patterns
SENTENCE_ENDINGS = re.compile(r'[.!?]\s+|[.!?]$')

# Minimum characters before we consider flushing
MIN_SENTENCE_LENGTH = 20

# Maximum time to buffer before forcing flush (ms)
MAX_BUFFER_TIME = 2000


def log(message: str, level: str = "INFO"):
    """Log streaming activity."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [STREAM] [{level}] {message}"
    print(entry, file=sys.stderr)
    try:
        STREAM_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(STREAM_LOG, "a") as f:
            f.write(entry + "\n")
    except:
        pass


# =============================================================================
# SENTENCE BUFFER
# =============================================================================

class SentenceBuffer:
    """
    Accumulates tokens and flushes complete sentences to TTS.

    The key insight: We don't need the WHOLE response to start speaking.
    Each sentence can be spoken as it's generated.
    """

    def __init__(self, on_sentence: Callable[[str], None]):
        self.buffer = ""
        self.on_sentence = on_sentence
        self.last_flush = time.time()
        self.sentence_count = 0

    def add(self, text: str):
        """Add text to buffer, flush if sentence complete."""
        self.buffer += text

        # Check for sentence boundaries
        while True:
            match = SENTENCE_ENDINGS.search(self.buffer)
            if match and match.end() >= MIN_SENTENCE_LENGTH:
                # Extract the complete sentence
                sentence = self.buffer[:match.end()].strip()
                self.buffer = self.buffer[match.end():].lstrip()

                if sentence:
                    self.sentence_count += 1
                    self.on_sentence(sentence)
                    self.last_flush = time.time()
            else:
                break

    def force_flush(self):
        """Flush whatever's in the buffer (end of response)."""
        if self.buffer.strip():
            self.on_sentence(self.buffer.strip())
            self.buffer = ""
            self.last_flush = time.time()

    def check_timeout(self):
        """Force flush if buffer has been sitting too long."""
        if self.buffer and (time.time() - self.last_flush) * 1000 > MAX_BUFFER_TIME:
            # Find any reasonable break point
            for sep in ['. ', ', ', ' - ', ': ', ' ']:
                if sep in self.buffer:
                    idx = self.buffer.rfind(sep)
                    if idx > MIN_SENTENCE_LENGTH // 2:
                        sentence = self.buffer[:idx + len(sep)].strip()
                        self.buffer = self.buffer[idx + len(sep):]
                        if sentence:
                            self.on_sentence(sentence)
                            self.last_flush = time.time()
                        return
            # No good break point, flush anyway if long enough
            if len(self.buffer) > MIN_SENTENCE_LENGTH * 2:
                self.force_flush()


# =============================================================================
# TTS DISPATCH
# =============================================================================

def create_tts_dispatcher():
    """Create a function that dispatches sentences to TTS queue."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    def dispatch(sentence: str):
        """Push a sentence to the TTS queue."""
        if not sentence or len(sentence) < 3:
            return

        payload = json.dumps({
            "text": sentence,
            "type": "stream",
            "timestamp": datetime.now().isoformat()
        })

        r.rpush(REDIS_KEY_TTS_QUEUE, payload)
        log(f"Streamed to TTS: {sentence[:50]}...")

    return dispatch


# =============================================================================
# PTY STREAM CAPTURE
# =============================================================================

def stream_command(cmd: list, on_sentence: Callable[[str], None], timeout: float = 60.0):
    """
    Execute a command and stream its output sentence-by-sentence.

    Uses pseudo-terminal to force line-buffered output from the subprocess.
    """
    log(f"Starting stream capture: {' '.join(cmd[:3])}...")

    buffer = SentenceBuffer(on_sentence)

    # Create pseudo-terminal
    master_fd, slave_fd = pty.openpty()

    try:
        # Spawn process with pty as stdout
        process = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            close_fds=True
        )

        os.close(slave_fd)  # Close slave in parent

        start_time = time.time()

        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                log("Stream timeout reached", "WARN")
                break

            # Check if process is done
            if process.poll() is not None:
                # Read any remaining output
                try:
                    remaining = os.read(master_fd, 4096).decode('utf-8', errors='ignore')
                    if remaining:
                        buffer.add(remaining)
                except:
                    pass
                break

            # Wait for output with timeout
            ready, _, _ = select.select([master_fd], [], [], 0.1)

            if ready:
                try:
                    data = os.read(master_fd, 1024).decode('utf-8', errors='ignore')
                    if data:
                        buffer.add(data)
                except OSError:
                    break
            else:
                # No new data, check for timeout flush
                buffer.check_timeout()

        # Flush any remaining buffer
        buffer.force_flush()

        log(f"Stream complete: {buffer.sentence_count} sentences")
        return process.returncode

    finally:
        os.close(master_fd)
        if process.poll() is None:
            process.terminate()


# =============================================================================
# CLAUDE INTEGRATION
# =============================================================================

def stream_claude_response(prompt: str, on_sentence: Callable[[str], None]) -> int:
    """
    Send a prompt to Claude Code and stream the response sentence-by-sentence.

    This is the main integration point for the voice daemon.
    """
    # Build the claude command
    # Note: This assumes `claude` CLI is available and configured
    cmd = ["claude", "--print", prompt]

    return stream_command(cmd, on_sentence)


def inject_voice_with_streaming(transcript: str):
    """
    Handle a voice input with streaming response.

    1. Inject the transcript (triggers Claude processing)
    2. Capture the response stream
    3. Push sentences to TTS as they're generated
    """
    log(f"Voice input: {transcript[:50]}...")

    dispatcher = create_tts_dispatcher()

    # Format the voice input
    prompt = f"[VOICE] {transcript}"

    # Stream the response
    result = stream_claude_response(prompt, dispatcher)

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Logos PTY - Streaming Response Injector")
    parser.add_argument("command", choices=["test", "stream"],
                        nargs="?", default="test")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt to stream")

    args = parser.parse_args()

    if args.command == "test":
        print("\n🌊 STREAM INJECTOR TEST")
        print("=" * 40)

        # Test the sentence buffer
        def print_sentence(s):
            print(f"  → {s}")

        buffer = SentenceBuffer(print_sentence)

        test_text = "This is the first sentence. Here comes the second one! And what about a question? Finally, some closing thoughts."

        print(f"\nInput: {test_text}")
        print("\nStreamed sentences:")

        # Simulate token-by-token input
        for char in test_text:
            buffer.add(char)

        buffer.force_flush()
        print(f"\nTotal sentences: {buffer.sentence_count}")

    elif args.command == "stream":
        if not args.prompt:
            print("Usage: logos_pty.py stream --prompt 'Your prompt here'")
            return

        dispatcher = create_tts_dispatcher()

        print(f"\nStreaming response for: {args.prompt[:50]}...")
        print("Sentences will appear as they're generated:\n")

        def print_and_dispatch(sentence):
            print(f"  → {sentence}")
            dispatcher(sentence)

        # Use echo for testing instead of actual claude
        result = stream_command(
            ["echo", args.prompt],
            print_and_dispatch
        )

        print(f"\nDone. Exit code: {result}")


if __name__ == "__main__":
    main()
