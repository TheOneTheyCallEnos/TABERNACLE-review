#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LOGOS MOUTH — Streaming TTS Daemon
==================================
Phase 1 optimization: Replaces batch TTS with streaming.

1. Waits for text in Redis (LOGOS:TTS_QUEUE) via blpop (instant).
2. Streams audio from ElevenLabs (Turbo v2.5).
3. Pipes audio to mpv for zero-latency playback.
4. Supports barge-in via LOGOS:INTERRUPT.

Run alongside logos_daemon.py initially for testing.
Once verified, disable TTS in logos_daemon.py.
"""

import json
import redis
import shutil
import subprocess
import sys

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    TTS_MODEL, TTS_LATENCY_OPTS, REDIS_KEY_TTS_QUEUE, REDIS_KEY_INTERRUPT
)

# Aliases for readability
QUEUE_KEY = REDIS_KEY_TTS_QUEUE
INTERRUPT_KEY = REDIS_KEY_INTERRUPT


class LogosMouth:
    def __init__(self):
        self._check_dependencies()
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Initialize ElevenLabs Client
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        self.current_process = None

        # Check for AirPods on startup
        self.audio_device = self._detect_airpods()

        print(f"[Mouth] Connected to Redis at {REDIS_HOST}")
        print(f"[Mouth] Model: {TTS_MODEL} | Latency opt: {TTS_LATENCY_OPTS}")
        print(f"[Mouth] Audio output: {self.audio_device or 'default'}")

    def _check_dependencies(self):
        """Ensure mpv is installed."""
        if not shutil.which("mpv"):
            raise RuntimeError("CRITICAL: 'mpv' not found. Install with: brew install mpv")

    def _detect_airpods(self):
        """Check if AirPods are available and return device name."""
        try:
            result = subprocess.run(
                ['/opt/homebrew/bin/SwitchAudioSource', '-a', '-t', 'output'],
                capture_output=True, text=True
            )
            outputs = result.stdout.strip().split('\n')
            airpods = [o for o in outputs if 'AirPods' in o]
            if airpods:
                # Switch to AirPods
                subprocess.run(
                    ['/opt/homebrew/bin/SwitchAudioSource', '-s', airpods[0], '-t', 'output'],
                    capture_output=True
                )
                return airpods[0]
        except Exception as e:
            print(f"[Mouth] Audio detection note: {e}")
        return None

    def kill_speech(self):
        """Immediately kill the audio player (Barge-in)."""
        if self.current_process:
            if self.current_process.poll() is None:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=0.2)
                except:
                    self.current_process.kill()
            self.current_process = None
            print("[Mouth] Speech interrupted.")

    def stream_audio(self, text: str):
        """Generate audio stream and pipe to mpv."""
        print(f"[Mouth] Speaking: '{text[:50]}...'")

        # Refresh AirPods detection each time (in case they connected/disconnected)
        self._detect_airpods()

        # mpv arguments for low-latency piping
        mpv_cmd = [
            "mpv",
            "--no-cache",
            "--no-terminal",
            "--no-video",
            "--demuxer-max-bytes=128KiB",
            "--",
            "fd://0"
        ]

        try:
            self.current_process = subprocess.Popen(
                mpv_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Request stream from ElevenLabs
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=TTS_MODEL,
                optimize_streaming_latency=TTS_LATENCY_OPTS
            )

            for chunk in audio_stream:
                # Check for interrupt signal mid-stream
                if self.r.get(INTERRUPT_KEY):
                    self.kill_speech()
                    self.r.delete(INTERRUPT_KEY)
                    print("[Mouth] Barge-in detected, stopping.")
                    break

                if chunk and self.current_process and self.current_process.stdin:
                    try:
                        self.current_process.stdin.write(chunk)
                        self.current_process.stdin.flush()
                    except BrokenPipeError:
                        break

            # Close stdin to signal end of audio
            if self.current_process and self.current_process.stdin:
                self.current_process.stdin.close()
                self.current_process.wait()

        except Exception as e:
            print(f"[Mouth] Error: {e}")
        finally:
            self.current_process = None
            print("[Mouth] Done.")

    def run(self):
        print("[Mouth] Waiting for text on LOGOS:TTS_QUEUE...")
        print("-" * 40)

        # Clear interrupt flag on boot
        self.r.delete(INTERRUPT_KEY)

        while True:
            try:
                # blpop blocks until data available (instant reaction)
                result = self.r.blpop(QUEUE_KEY, timeout=0)

                if result:
                    _, message = result
                    try:
                        data = json.loads(message)
                        text = data.get("text", "")
                        if text:
                            self.stream_audio(text)
                    except json.JSONDecodeError:
                        # Handle plain text (not JSON)
                        if message.strip():
                            self.stream_audio(message.strip())

            except KeyboardInterrupt:
                print("\n[Mouth] Shutting down...")
                self.kill_speech()
                break
            except redis.ConnectionError:
                print("[Mouth] Redis connection lost. Retrying...")
                import time
                time.sleep(1)
            except Exception as e:
                print(f"[Mouth] Error: {e}")


if __name__ == "__main__":
    LogosMouth().run()
