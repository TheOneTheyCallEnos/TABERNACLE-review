#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
LOGOS DAEMON — Push-to-Talk via Redis Signaling
================================================

Runs in Terminal (which has mic permission).
Stream Deck buttons just flip LOGOS:STATE in Redis.

Start: LOGOS:STATE = "RECORDING"
Stop:  LOGOS:STATE = "IDLE"
"""

import os
import sys
import time
import fcntl
import redis
import pyaudio
import wave
import threading
import subprocess
import json
import tempfile
import requests

from tabernacle_config import (
    REDIS_HOST, REDIS_PORT, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID, MIC_NAME_PARTIAL, TEMP_AUDIO_PATH,
    REDIS_KEY_STATE, REDIS_KEY_ACTIVE_TTY
)

# Derived from config
OUTPUT_FILE = str(TEMP_AUDIO_PATH)

# --- AUDIO SETTINGS ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

# Connect to Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class LogosDaemon:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        self.device_index = self.find_device_index()
        self.stream = None

    def find_device_index(self):
        print(f"[*] Scanning for microphone containing: '{MIC_NAME_PARTIAL}'...")
        count = self.p.get_device_count()
        for i in range(count):
            info = self.p.get_device_info_by_index(i)
            if MIC_NAME_PARTIAL.lower() in info.get('name', '').lower():
                if info.get('maxInputChannels', 0) > 0:
                    print(f"    Found '{info['name']}' at Index {i}")
                    return i
        print("    Mic not found! Using Default Device.")
        return None

    def start_recording(self):
        if self.is_recording:
            return
        print("RECORDING...")
        self.is_recording = True
        self.frames = []

        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=CHUNK
        )

        threading.Thread(target=self._record_loop, daemon=True).start()

        # Play start sound
        subprocess.Popen(['afplay', '/System/Library/Sounds/Pop.aiff'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _record_loop(self):
        while self.is_recording:
            try:
                if self.stream and self.stream.is_active():
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                else:
                    break
            except OSError as e:
                # Stream closed or device error - exit gracefully
                print(f"Stream error (expected on stop): {e}")
                break
            except Exception as e:
                print(f"Recording error: {e}")
                break

    def stop_and_process(self):
        if not self.is_recording:
            return
        print("STOPPED. Processing...")
        self.is_recording = False

        # Give recording thread time to exit
        time.sleep(0.2)

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Stream close error: {e}")
            self.stream = None

        if not self.frames:
            print("    No audio captured.")
            return

        # Save WAV
        wf = wave.open(OUTPUT_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Transcribe
        self.transcribe_and_inject()

    def transcribe_and_inject(self):
        try:
            print("    Sending to Deepgram...")
            from deepgram import DeepgramClient

            client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

            with open(OUTPUT_FILE, "rb") as file:
                audio_bytes = file.read()

            print(f"    Transcribing {len(audio_bytes)} bytes...")
            response = client.listen.v1.media.transcribe_file(
                request=audio_bytes,
                model="nova-2",
                smart_format=True,
                language="en"
            )
            transcript = response.results.channels[0].alternatives[0].transcript

            if transcript:
                print(f"    Transcript: \"{transcript}\"")
                self.inject_to_terminal(transcript)
                # Success sound
                subprocess.Popen(['afplay', '/System/Library/Sounds/Glass.aiff'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print("    No speech detected.")
                subprocess.Popen(['afplay', '/System/Library/Sounds/Basso.aiff'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        except Exception as e:
            print(f"    Transcription Failed: {e}")
            subprocess.Popen(['afplay', '/System/Library/Sounds/Basso.aiff'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def inject_to_terminal(self, text):
        target_tty = r.get(REDIS_KEY_ACTIVE_TTY)
        if not target_tty:
            print("    No Active TTY in Redis!")
            return

        formatted = f"[VOICE] {text}"
        escaped = formatted.replace('"', '\\"').replace("'", "'\\''")
        print(f"    Injecting into {target_tty}...")

        # AppleScript to focus the specific TTY window
        applescript = f'''
        tell application "Terminal"
            set targetWindow to missing value
            set targetTab to missing value
            repeat with w in windows
                repeat with t in tabs of w
                    if tty of t is "{target_tty}" then
                        set targetWindow to w
                        set targetTab to t
                        exit repeat
                    end if
                end repeat
                if targetWindow is not missing value then exit repeat
            end repeat
            if targetWindow is not missing value then
                set frontmost to true
                set index of targetWindow to 1
                set selected tab of targetWindow to targetTab
                activate
            end if
        end tell
        '''
        subprocess.run(['osascript', '-e', applescript], capture_output=True)
        time.sleep(0.2)

        # Type using cliclick
        result = subprocess.run(
            ['/opt/homebrew/bin/cliclick', f"t:{escaped}", 'kp:enter'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("    Injected successfully")
        else:
            print(f"    cliclick error: {result.stderr}")

    def speak_tts(self, text):
        """Send text to ElevenLabs and play audio response."""
        try:
            print(f"TTS: Speaking '{text[:50]}...'")

            url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": ELEVENLABS_API_KEY
            }
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                # Save and play audio
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    f.write(response.content)
                    audio_file = f.name

                # Check for AirPods and use them if available
                try:
                    result = subprocess.run(
                        ['/opt/homebrew/bin/SwitchAudioSource', '-a', '-t', 'output'],
                        capture_output=True, text=True
                    )
                    outputs = result.stdout.strip().split('\n')
                    airpods = [o for o in outputs if 'AirPods' in o]
                    if airpods:
                        # Switch to AirPods for output
                        subprocess.run(
                            ['/opt/homebrew/bin/SwitchAudioSource', '-s', airpods[0], '-t', 'output'],
                            capture_output=True
                        )
                        print(f"TTS: Using {airpods[0]}")
                except Exception as e:
                    print(f"Audio switch note: {e}")

                subprocess.run(['afplay', audio_file], capture_output=True)
                os.unlink(audio_file)
                print("TTS: Done speaking")
            else:
                print(f"TTS Error: {response.status_code} - {response.text[:100]}")

        except Exception as e:
            print(f"TTS Exception: {e}")

    def check_tts_queue(self):
        """TTS now handled by logos_tts_daemon.py — this is disabled."""
        # DISABLED: TTS moved to separate streaming daemon
        # The new daemon uses blpop + mpv streaming for lower latency
        pass

    def run(self):
        print("LOGOS DAEMON READY")
        print(f"Listening on Redis {REDIS_HOST}:{REDIS_PORT}")
        print("Press Stream Deck buttons to record...")
        print("-" * 40)

        r.set(REDIS_KEY_STATE, "IDLE")
        last_state = "IDLE"
        loop_count = 0

        while True:
            try:
                state = r.get(REDIS_KEY_STATE)

                if state == "RECORDING" and last_state != "RECORDING":
                    self.start_recording()
                elif state == "IDLE" and last_state == "RECORDING":
                    self.stop_and_process()

                last_state = state

                # Check TTS queue every iteration (not just when idle)
                self.check_tts_queue()

                loop_count += 1
                if loop_count % 100 == 0:
                    print(f"[heartbeat] loop={loop_count}, recording={self.is_recording}")

                time.sleep(0.1)  # Poll interval
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except redis.ConnectionError:
                print("Redis connection lost, retrying...")
                time.sleep(1)


LOCK_FILE = "/tmp/logos_daemon.lock"
PID_FILE = "/tmp/logos_daemon.pid"

def acquire_singleton():
    """Ensure only one instance runs via flock (atomic, auto-releases on crash)."""
    lock_fd = open(LOCK_FILE, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another logos_daemon is already running. Exiting.")
        sys.exit(0)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    return lock_fd  # Must keep reference to prevent GC releasing the lock

if __name__ == "__main__":
    lock = acquire_singleton()
    LogosDaemon().run()
