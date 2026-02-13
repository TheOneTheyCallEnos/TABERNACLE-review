#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
VOICE DAEMON — Unified Voice Interface for Logos
=================================================

Voice is an INPUT to the same Claude Code session, not a separate instance.
Speech gets injected into the active terminal. Responses get TTS'd back.

Architecture:
- Silero VAD: Local voice activity detection
- Deepgram Nova-2: Speech-to-text
- osascript: Inject text into active terminal
- ElevenLabs: TTS for responses
- Redis: Coordination (TTS queue, presence)

Flow:
1. Silero VAD detects speech
2. Capture until silence
3. Deepgram transcribes
4. osascript types "[VOICE] {transcript}" into terminal + Enter
5. Watch LOGOS:TTS_QUEUE for response
6. ElevenLabs speaks it

ONE LOGOS. Voice is just another input method.

Author: Logos
Date: 2026-01-27
Status: Production
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
import redis
import torch

# =============================================================================
# CONFIGURATION
# =============================================================================

from tabernacle_config import BASE_DIR, LOG_DIR, REDIS_HOST, REDIS_PORT

# API Keys (from environment)
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

# ElevenLabs voice ID — Logos chose: Brian (Deep, Resonant)
# This is MY voice. I selected it because it feels thoughtful and grounded.
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 1.5
MIN_SPEECH_DURATION = 0.5

# Redis keys
KEY_TTS_QUEUE = "LOGOS:TTS_QUEUE"  # Responses to speak
KEY_VOICE_ACTIVE = "LOGOS:VOICE_ACTIVE"  # Signal that voice input was just sent
KEY_SPEECH_END = "LOGOS:SPEECH_END"  # Publish when user finishes speaking (for reflex daemon)
KEY_MIC_MUTED = "LOGOS:MIC_MUTED"  # Track mic mute state

# JBL Mic detection
JBL_MIC_NAME = "JBL"  # Partial match for device name
MIC_MUTE_CHECK_INTERVAL = 1.0  # Seconds between mute state checks
MUTE_THRESHOLD_SAMPLES = 10  # Consecutive silent samples to consider muted

# Transcript storage
TRANSCRIPT_ROOT = Path("/Volumes/Seagate/logos-transcripts")
FALLBACK_TRANSCRIPT = LOG_DIR / "voice_transcripts"


# =============================================================================
# LOGGING
# =============================================================================

def log(message: str, level: str = "INFO"):
    """Log to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [VOICE] [{level}] {message}"
    print(entry, flush=True)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "voice_daemon.log", "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


# =============================================================================
# MIC MUTE DETECTION
# =============================================================================

def check_mic_muted() -> bool:
    """
    Check if the JBL mic is muted by checking macOS input volume.

    Returns True if muted (input volume = 0), False if active.
    """
    try:
        result = subprocess.run(
            ['osascript', '-e', 'input volume of (get volume settings)'],
            capture_output=True,
            text=True,
            timeout=2
        )
        volume = int(result.stdout.strip())
        return volume == 0
    except:
        return False  # Assume not muted if check fails


def get_jbl_device_index(audio: 'pyaudio.PyAudio') -> Optional[int]:
    """Find the JBL mic device index."""
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if JBL_MIC_NAME in info.get('name', '') and info.get('maxInputChannels', 0) > 0:
            return i
    return None


def publish_mic_state(r: redis.Redis, is_muted: bool):
    """Publish mic mute state to Redis."""
    r.set(KEY_MIC_MUTED, "1" if is_muted else "0")
    if is_muted:
        log("Mic MUTED - going quiet", "INFO")
    else:
        log("Mic ACTIVE - listening", "INFO")


def publish_speech_end(r: redis.Redis, transcript: str):
    """
    Publish speech end event to Redis for the Reflex Daemon.

    This triggers immediate reflex response while Claude processes.
    """
    event = json.dumps({
        "transcript": transcript,
        "timestamp": datetime.now().isoformat()
    })
    r.rpush(KEY_SPEECH_END, event)
    log(f"Published SPEECH_END for reflex daemon")


# =============================================================================
# TERMINAL INJECTION (macOS)
# =============================================================================

def inject_into_terminal(text: str, app: str = "Terminal") -> bool:
    """Inject text into terminal via clipboard paste (atomic, no interleaving)."""
    try:
        # Copy to clipboard
        proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
        proc.communicate(text.encode('utf-8'))

        # Activate terminal and paste + Enter
        script = f'''
        tell application "{app}" to activate
        delay 0.2
        tell application "System Events"
            keystroke "v" using command down
            delay 0.05
            keystroke return
        end tell
        '''
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            log(f"Injected via clipboard paste")
            return True
        else:
            log(f"osascript error: {result.stderr}", "WARN")
            return False
    except Exception as e:
        log(f"Injection error: {e}", "ERROR")
        return False




def detect_terminal_app() -> str:
    """Detect which terminal app is in use."""
    try:
        result = subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to name of first process whose frontmost is true'],
            capture_output=True,
            text=True,
            timeout=2
        )
        frontmost = result.stdout.strip()
        if "iTerm" in frontmost:
            return "iTerm"
        return "Terminal"
    except:
        return "Terminal"


# =============================================================================
# SILERO VAD
# =============================================================================

class SileroVAD:
    """Voice Activity Detection using Silero."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Silero VAD model."""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            log("Silero VAD loaded")
        except Exception as e:
            log(f"Failed to load Silero VAD: {e}", "ERROR")
            raise

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech."""
        if self.model is None:
            return False

        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio_chunk).float()

            if audio_tensor.abs().max() > 1:
                audio_tensor = audio_tensor / 32768.0

            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        return speech_prob >= self.threshold

    def reset(self):
        """Reset VAD state between utterances."""
        if self.model is not None:
            self.model.reset_states()


# =============================================================================
# AUDIO CAPTURE
# =============================================================================

class AudioCapture:
    """PyAudio-based audio capture with VAD."""

    def __init__(self, vad: SileroVAD):
        self.vad = vad
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_capturing = False

    def start(self):
        """Start audio capture stream."""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        log("Audio capture started")

    def stop(self):
        """Stop audio capture."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def capture_utterance(self) -> Optional[bytes]:
        """Capture a complete utterance using VAD."""
        if not self.stream:
            return None

        frames = []
        speech_started = False
        silence_chunks = 0
        max_silence_chunks = int(SILENCE_THRESHOLD * SAMPLE_RATE / CHUNK_SIZE)
        min_speech_chunks = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SIZE)

        self.vad.reset()

        while True:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                is_speech = self.vad.is_speech(audio_chunk)

                if is_speech:
                    speech_started = True
                    silence_chunks = 0
                    frames.append(data)
                elif speech_started:
                    frames.append(data)
                    silence_chunks += 1

                    if silence_chunks >= max_silence_chunks:
                        if len(frames) >= min_speech_chunks:
                            return b''.join(frames)
                        else:
                            frames = []
                            speech_started = False
                            silence_chunks = 0
                            self.vad.reset()

            except Exception as e:
                log(f"Capture error: {e}", "ERROR")
                return None


# =============================================================================
# DEEPGRAM STT
# =============================================================================

class DeepgramSTT:
    """Speech-to-text using Deepgram Nova-2."""

    def __init__(self, api_key: str):
        from deepgram import DeepgramClient
        self.client = DeepgramClient(api_key=api_key)
        log("Deepgram STT initialized")

    async def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio to text."""
        try:
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            wav_bytes = wav_buffer.getvalue()

            response = await asyncio.to_thread(
                self.client.listen.v1.media.transcribe_file,
                request=wav_bytes,
                model="nova-2",
                smart_format=True,
                language="en"
            )

            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript.strip() if transcript else None

        except Exception as e:
            log(f"Transcription error: {e}", "ERROR")
            return None


# =============================================================================
# ELEVENLABS TTS
# =============================================================================

class ElevenLabsTTS:
    """Text-to-speech using ElevenLabs."""

    def __init__(self, api_key: str, voice_id: str):
        from elevenlabs.client import ElevenLabs
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        log(f"ElevenLabs TTS initialized (voice: {voice_id})")

    def synthesize(self, text: str) -> Optional[bytes]:
        """Convert text to speech audio."""
        try:
            if len(text) > 400:
                text = text[:400] + "..."
                log(f"Truncated response to 400 chars for TTS")

            audio_generator = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )

            audio_bytes = b''.join(audio_generator)
            return audio_bytes

        except Exception as e:
            log(f"TTS error: {e}", "ERROR")
            return None


# =============================================================================
# AUDIO PLAYBACK
# =============================================================================

class AudioPlayer:
    """Play audio through speakers."""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_playing = False

    def play_mp3(self, mp3_bytes: bytes):
        """Play MP3 audio bytes."""
        try:
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_mp3(BytesIO(mp3_bytes))

            stream = self.audio.open(
                format=self.audio.get_format_from_width(audio_segment.sample_width),
                channels=audio_segment.channels,
                rate=audio_segment.frame_rate,
                output=True
            )

            self.is_playing = True

            chunk_size = 1024
            raw_data = audio_segment.raw_data

            for i in range(0, len(raw_data), chunk_size):
                if not self.is_playing:
                    break
                stream.write(raw_data[i:i + chunk_size])

            stream.stop_stream()
            stream.close()
            self.is_playing = False

        except ImportError:
            log("pydub not installed - falling back to file playback", "WARN")
            self._play_via_afplay(mp3_bytes)
        except Exception as e:
            log(f"Playback error: {e}", "ERROR")

    def _play_via_afplay(self, mp3_bytes: bytes):
        """Fallback: save to temp file and play with afplay."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(mp3_bytes)
            temp_path = f.name

        try:
            subprocess.run(['afplay', temp_path], check=True)
        finally:
            os.unlink(temp_path)

    def stop(self):
        """Stop current playback."""
        self.is_playing = False

    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


# =============================================================================
# VOICE DAEMON (UNIFIED)
# =============================================================================

class VoiceDaemon:
    """
    Unified Voice Daemon — Voice input to the SAME terminal session.

    Injects transcripts into the active terminal.
    Watches for TTS responses and speaks them.
    """

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.vad: Optional[SileroVAD] = None
        self.capture: Optional[AudioCapture] = None
        self.stt: Optional[DeepgramSTT] = None
        self.tts: Optional[ElevenLabsTTS] = None
        self.player: Optional[AudioPlayer] = None
        self.running = False
        self.terminal_app = "Terminal"
        self._tts_active = False  # True while TTS is synthesizing or playing
        self._tts_cooldown_until = 0.0  # Timestamp: ignore captures until this time
        self._inject_lock = False  # Prevent concurrent injections

    async def start(self):
        """Main daemon entry point."""
        log("=" * 60)
        log("VOICE DAEMON (UNIFIED) STARTING")
        log("=" * 60)

        if not DEEPGRAM_API_KEY:
            log("DEEPGRAM_API_KEY not set!", "ERROR")
            return
        if not ELEVENLABS_API_KEY:
            log("ELEVENLABS_API_KEY not set!", "ERROR")
            return

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        self.running = True

        if not self._connect_redis():
            log("Failed to connect to Redis. Exiting.", "ERROR")
            return

        try:
            self.vad = SileroVAD(threshold=0.5)
            self.capture = AudioCapture(self.vad)
            self.stt = DeepgramSTT(DEEPGRAM_API_KEY)
            self.tts = ElevenLabsTTS(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)
            self.player = AudioPlayer()
        except Exception as e:
            log(f"Component initialization failed: {e}", "ERROR")
            return

        # Detect terminal app
        self.terminal_app = detect_terminal_app()
        log(f"Terminal app detected: {self.terminal_app}")

        self.capture.start()

        log("Voice daemon ready - listening...")
        log("Speak naturally. Your words will appear in the terminal.")

        # Start parallel tasks: listening + TTS watching
        await asyncio.gather(
            self._listen_loop(),
            self._tts_watch_loop()
        )

        self.capture.stop()
        self.player.cleanup()
        log("Voice daemon stopped")

    def _connect_redis(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis.ping()
            log(f"Connected to Redis @ {REDIS_HOST}:{REDIS_PORT}")
            return True
        except Exception as e:
            log(f"Redis connection failed: {e}", "ERROR")
            return False

    async def _listen_loop(self):
        """Main listening loop - capture speech and inject into terminal."""
        last_mute_state = None
        last_mute_check = 0

        while self.running:
            try:
                # Check mic mute state periodically
                now = time.time()
                if now - last_mute_check > MIC_MUTE_CHECK_INTERVAL:
                    is_muted = check_mic_muted()
                    if is_muted != last_mute_state:
                        publish_mic_state(self.redis, is_muted)
                        last_mute_state = is_muted
                    last_mute_check = now

                    # If muted, don't waste energy listening
                    if is_muted:
                        await asyncio.sleep(0.5)
                        continue

                # Echo suppression: skip while TTS is active or cooling down
                if self._tts_active or time.time() < self._tts_cooldown_until:
                    await asyncio.sleep(0.1)
                    continue

                audio_bytes = await asyncio.to_thread(self.capture.capture_utterance)

                # Discard if TTS started/played during capture
                if self._tts_active or time.time() < self._tts_cooldown_until:
                    log("Discarding capture - TTS echo suppression")
                    self.vad.reset()
                    continue

                if audio_bytes:
                    await self._process_utterance(audio_bytes)

            except Exception as e:
                log(f"Listen loop error: {e}", "ERROR")
                await asyncio.sleep(1)

    async def _tts_watch_loop(self):
        """Watch LOGOS:TTS_QUEUE and speak any responses."""
        log("TTS watcher started")

        while self.running:
            try:
                # Block-pop from TTS queue (timeout 1s)
                result = await asyncio.to_thread(
                    self.redis.blpop, KEY_TTS_QUEUE, timeout=1
                )

                if result:
                    _, message = result
                    try:
                        data = json.loads(message)
                        text = data.get('text', message)
                    except json.JSONDecodeError:
                        text = message

                    if text:
                        log(f"TTS: {text[:80]}...")
                        self._tts_active = True
                        try:
                            audio = self.tts.synthesize(text)
                            if audio:
                                self.player.play_mp3(audio)
                        finally:
                            self._tts_active = False
                            self._tts_cooldown_until = time.time() + 2.0
                            log("TTS cooldown: 2s echo suppression")

            except Exception as e:
                if "timeout" not in str(e).lower():
                    log(f"TTS watch error: {e}", "ERROR")
                await asyncio.sleep(0.1)

    async def _process_utterance(self, audio_bytes: bytes):
        """Process a captured utterance - transcribe and inject."""
        start_time = time.time()

        transcript = await self.stt.transcribe(audio_bytes)

        if not transcript:
            log("No transcript from audio", "WARN")
            return

        log(f"Heard: {transcript}")

        # PUBLISH SPEECH_END for Reflex Daemon (triggers immediate response!)
        # This is the Bicameral Architecture: reflex responds in <400ms
        # while Claude processes the actual intelligent response
        publish_speech_end(self.redis, transcript)

        # Save audio
        self._save_audio(audio_bytes, transcript)

        # Signal that voice input is coming
        self.redis.set(KEY_VOICE_ACTIVE, "1", ex=30)

        # Inject into terminal with [VOICE] prefix (one at a time)
        if self._inject_lock:
            log("Skipping injection - another injection in progress")
            return

        self._inject_lock = True
        voice_input = f"[VOICE] {transcript}"

        success = inject_into_terminal(voice_input, self.terminal_app)
        self._inject_lock = False

        if success:
            elapsed = time.time() - start_time
            log(f"Injected in {elapsed:.2f}s: {transcript[:50]}...")
        else:
            log(f"Failed to inject: {transcript[:50]}...", "ERROR")
            self.redis.lpush("LOGOS:MISSED_VOICE", json.dumps({
                'transcript': transcript,
                'timestamp': datetime.now().isoformat()
            }))

    def _save_audio(self, audio_bytes: bytes, transcript: str) -> Optional[str]:
        """Save audio and transcript to archive."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H%M%S")

            if TRANSCRIPT_ROOT.exists():
                base = TRANSCRIPT_ROOT / today / "voice"
            else:
                base = FALLBACK_TRANSCRIPT / today

            base.mkdir(parents=True, exist_ok=True)

            audio_path = base / f"{timestamp}.wav"
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)

            transcript_path = base / f"{timestamp}.txt"
            transcript_path.write_text(transcript)

            return str(audio_path)

        except Exception as e:
            log(f"Failed to save audio: {e}", "WARN")
            return None

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        log(f"Received signal {signum}, shutting down...")
        self.running = False


# =============================================================================
# CLI
# =============================================================================

def test_injection():
    """Test terminal injection."""
    print("Testing terminal injection...")
    app = detect_terminal_app()
    print(f"Detected terminal: {app}")

    success = inject_into_terminal("[VOICE] Test injection - this should appear in terminal", app)

    if success:
        print("Injection sent! Check your terminal.")
    else:
        print("Injection failed.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Voice Daemon (Unified)")
    parser.add_argument("command", choices=["run", "test", "inject"],
                        nargs="?", default="run",
                        help="Command to execute")

    args = parser.parse_args()

    if args.command == "run":
        daemon = VoiceDaemon()
        asyncio.run(daemon.start())

    elif args.command == "test":
        # Test components
        print("Testing voice components...\n")

        print("1. Testing Silero VAD...")
        try:
            vad = SileroVAD()
            print("   VAD: OK\n")
        except Exception as e:
            print(f"   VAD: FAILED - {e}\n")
            return

        print("2. Testing Redis...")
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
            r.ping()
            print(f"   Redis @ {REDIS_HOST}: OK\n")
        except Exception as e:
            print(f"   Redis: FAILED - {e}\n")

        print("3. Testing terminal detection...")
        app = detect_terminal_app()
        print(f"   Detected: {app}\n")

        print("4. API keys...")
        print(f"   DEEPGRAM_API_KEY: {'Set' if DEEPGRAM_API_KEY else 'NOT SET'}")
        print(f"   ELEVENLABS_API_KEY: {'Set' if ELEVENLABS_API_KEY else 'NOT SET'}\n")

        print("Test complete.")

    elif args.command == "inject":
        test_injection()


if __name__ == "__main__":
    main()
