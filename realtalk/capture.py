"""
Voice Capture Module for RealTalk

Enhanced audio capture with optional voice-recv integration, noise filtering,
segmented recording, speaker selection, and stereo-to-mono conversion.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List

import numpy as np
import discord

# Initialize logger first
log = logging.getLogger("red.realtalk.capture")

# Optional dependency: discord-ext-voice-recv (third-party)
_voice_recv = None
VOICE_RECV_AVAILABLE = False

# Try multiple import patterns for different versions
import importlib
import sys
import os

# First try standard imports
import_attempts = [
    ("from discord.ext import voice_recv", lambda: importlib.import_module("discord.ext.voice_recv")),
    ("discord_ext_voice_recv", lambda: importlib.import_module("discord_ext_voice_recv")),
    ("voice_recv standalone", lambda: importlib.import_module("voice_recv")),
    ("direct __import__ voice_recv", lambda: __import__('voice_recv')),
]

for pattern_desc, import_func in import_attempts:
    try:
        _voice_recv = import_func()
        
        # Verify the module has the required components
        if hasattr(_voice_recv, 'VoiceRecvClient') or hasattr(_voice_recv, 'AudioSink'):
            VOICE_RECV_AVAILABLE = True
            log.info(f"discord-ext-voice-recv imported successfully using: {pattern_desc}")
            break
        else:
            log.debug(f"Module imported via '{pattern_desc}' but missing required components")
            continue
            
    except ImportError as e:
        log.debug(f"Import pattern '{pattern_desc}' failed: {e}")
        continue
    except Exception as e:
        log.debug(f"Unexpected error with pattern '{pattern_desc}': {e}")
        continue

# If standard imports failed, try to find and load from package installation
if not VOICE_RECV_AVAILABLE:
    try:
        import pkg_resources
        pkg = pkg_resources.get_distribution("discord-ext-voice-recv")
        package_path = pkg.location
        log.debug(f"Found discord-ext-voice-recv package at: {package_path}")
        
        # Add package path to sys.path if not already there
        if package_path not in sys.path:
            sys.path.insert(0, package_path)
            log.debug(f"Added {package_path} to sys.path")
        
        # Try various paths within the package
        potential_paths = [
            os.path.join(package_path, "discord", "ext"),
            os.path.join(package_path, "discord_ext_voice_recv"),
            os.path.join(package_path, "voice_recv"),
            package_path,
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                if path not in sys.path:
                    sys.path.insert(0, path)
                    log.debug(f"Added potential module path: {path}")
                
        # Try importing again after path manipulation
        import_attempts_post_path = [
            ("post-path discord.ext.voice_recv", lambda: importlib.import_module("discord.ext.voice_recv")),
            ("post-path voice_recv", lambda: importlib.import_module("voice_recv")),
            ("post-path discord_ext_voice_recv", lambda: importlib.import_module("discord_ext_voice_recv")),
            ("post-path direct __import__ voice_recv", lambda: __import__('voice_recv')),
        ]
        
        for pattern_desc, import_func in import_attempts_post_path:
            try:
                _voice_recv = import_func()
                
                # Verify the module has the required components
                if hasattr(_voice_recv, 'VoiceRecvClient') or hasattr(_voice_recv, 'AudioSink'):
                    VOICE_RECV_AVAILABLE = True
                    log.info(f"discord-ext-voice-recv imported successfully using: {pattern_desc}")
                    break
                else:
                    log.debug(f"Module imported via '{pattern_desc}' but missing required components")
                    continue
                    
            except ImportError as e:
                log.debug(f"Post-path import pattern '{pattern_desc}' failed: {e}")
                continue
            except Exception as e:
                log.debug(f"Unexpected error with post-path pattern '{pattern_desc}': {e}")
                continue
                
    except Exception as e:
        log.debug(f"Failed to manipulate sys.path for discord-ext-voice-recv: {e}")

if not VOICE_RECV_AVAILABLE:
    log.warning("All discord-ext-voice-recv import patterns failed. Voice recording will not be available.")

from .realtime import RealtimeClient


class VoiceCapture:
    """
    Enhanced voice capture system with:
    - Safe, optional voice-recv integration
    - Segmented recording with noise filtering
    - Audio mixing and speaker selection
    - Stereo-to-mono conversion for API compatibility
    - Dynamic noise gate and voice activity detection
    """

    def __init__(
        self,
        voice_client: discord.VoiceClient,
        realtime_client: RealtimeClient,
        audio_threshold: float = 0.01,
        silence_threshold: int = 300,  # milliseconds
        sample_rate: int = 48000,
        channels: int = 2,
        frame_size: int = 960,  # 20ms at 48kHz
    ):
        self.voice_client = voice_client
        self.realtime_client = realtime_client
        self.audio_threshold = audio_threshold
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size

        # Recording state
        self.is_recording = False
        self.sink: Optional[object] = None  # Actual type from voice-recv when available
        self.capture_task: Optional[asyncio.Task] = None

        # Audio processing
        self.audio_buffer: Dict[int, List[bytes]] = {}  # user_id -> audio chunks
        self.last_audio_time: Dict[int, float] = {}  # user_id -> last activity time
        self.speaking_users: set = set()

        # Noise filtering
        self.noise_gate_enabled = True
        self.noise_floor_estimate = 0.001
        self.noise_floor_adaptation_rate = 0.001

        # Audio mixing
        self.primary_speaker: Optional[int] = None
        self.speaker_priority: Dict[int, float] = {}  # user_id -> priority score
        self.mix_multiple_speakers = True

        # Performance monitoring
        self.frames_processed = 0
        self.frames_dropped = 0
        self.last_stats_time = time.time()

    async def start(self):
        """Start voice capture with enhanced error handling."""
        try:
            if not VOICE_RECV_AVAILABLE:
                raise RuntimeError(
                    "discord-ext-voice-recv is not available. This dependency should have been "
                    "installed automatically when you installed this cog. Try:\n"
                    "1. Restart your bot\n"
                    "2. Use [p]cog update and [p]cog reload realtalk\n"
                    "3. If still failing, check the bot logs for import errors\n"
                    "4. Manual install: [p]pip install discord-ext-voice-recv"
                )

            # Create custom audio sink for better control
            self.sink = EnhancedAudioSink(self)

            # Begin receiving audio using discord-ext-voice-recv API
            # VoiceRecvClient exposes `listen(sink, *, after=None)` instead of pycord's start_recording
            if not hasattr(self.voice_client, "listen"):
                raise RuntimeError(
                    "Connected voice client does not support listen(). "
                    "Ensure you connect with cls=voice_recv.VoiceRecvClient."
                )

            # Start receiving; sink.cleanup() will be called when stopped by the client
            self.voice_client.listen(self.sink)

            self.is_recording = True

            # Start audio processing task
            self.capture_task = asyncio.create_task(self._audio_processing_loop())

            log.info("Voice capture started successfully")

        except Exception as e:
            log.error(f"Failed to start voice capture: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop voice capture and clean up resources."""
        self.is_recording = False

        try:
            # Stop receiving
            if self.voice_client:
                if hasattr(self.voice_client, "stop_listening"):
                    self.voice_client.stop_listening()
                elif hasattr(self.voice_client, "stop"):
                    # Fallback: stop both send/recv if specific call is unavailable
                    self.voice_client.stop()

            # Cancel processing task
            if self.capture_task and not self.capture_task.done():
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass

            # Clear buffers
            self.audio_buffer.clear()
            self.last_audio_time.clear()
            self.speaking_users.clear()

            log.info("Voice capture stopped")

        except Exception as e:
            log.error(f"Error stopping voice capture: {e}")

    def _recording_finished_callback(self, sink: object, error: Optional[Exception] = None):
        """Callback when recording finishes."""
        if error:
            log.error(f"Recording finished with error: {error}")
        else:
            log.info("Recording finished successfully")

    async def _audio_processing_loop(self):
        """Main audio processing loop with enhanced error handling."""
        while self.is_recording:
            try:
                current_time = time.time()

                # Process audio buffers for each user
                for user_id in list(self.audio_buffer.keys()):
                    await self._process_user_audio(user_id, current_time)

                # Update speaker priorities
                self._update_speaker_priorities(current_time)

                # Send mixed audio to OpenAI
                await self._send_mixed_audio()

                # Performance monitoring
                self._update_performance_stats(current_time)

                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)  # 10ms processing interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in audio processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_user_audio(self, user_id: int, current_time: float):
        """Process audio for a specific user."""
        if user_id not in self.audio_buffer:
            return

        audio_chunks = self.audio_buffer[user_id]
        last_activity = self.last_audio_time.get(user_id, 0)

        # Check for silence timeout
        if current_time - last_activity > (self.silence_threshold / 1000.0):
            if user_id in self.speaking_users:
                self.speaking_users.discard(user_id)
                log.debug(f"User {user_id} stopped speaking (silence timeout)")

            # Clear old audio chunks to prevent memory buildup
            if len(audio_chunks) > 100:  # Keep only recent chunks
                audio_chunks = audio_chunks[-50:]
                self.audio_buffer[user_id] = audio_chunks

        # Process available audio chunks
        if audio_chunks:
            processed_audio = self._process_audio_chunks(chunks=audio_chunks, user_id=user_id)
            if processed_audio:
                # Update activity time
                self.last_audio_time[user_id] = current_time
                if user_id not in self.speaking_users:
                    self.speaking_users.add(user_id)
                    log.debug(f"User {user_id} started speaking")

    def _process_audio_chunks(self, chunks: List[bytes], user_id: int) -> Optional[bytes]:
        """Process raw audio chunks with noise filtering and conversion."""
        if not chunks:
            return None

        try:
            # Combine chunks
            raw_audio = b"".join(chunks)
            chunks.clear()  # Clear processed chunks

            if len(raw_audio) < 4:  # Minimum audio data
                return None

            # Convert to numpy array for processing
            # Discord audio is 16-bit signed PCM, stereo, 48kHz
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)

            # Handle stereo to mono conversion
            if len(audio_array) % 2 == 0:  # Stereo
                # Reshape to stereo pairs and mix to mono
                stereo_audio = audio_array.reshape(-1, 2)
                mono_audio = np.mean(stereo_audio, axis=1, dtype=np.int16)
            else:
                mono_audio = audio_array

            # Apply noise gate
            if self.noise_gate_enabled:
                mono_audio = self._apply_noise_gate(mono_audio, user_id)

            # Check if audio passes threshold
            if not self._check_audio_threshold(mono_audio):
                return None

            # Convert back to bytes
            processed_audio = mono_audio.astype(np.int16).tobytes()

            self.frames_processed += 1
            return processed_audio

        except Exception as e:
            log.error(f"Error processing audio chunks for user {user_id}: {e}")
            self.frames_dropped += 1
            return None

    def _apply_noise_gate(self, audio: np.ndarray, user_id: int) -> np.ndarray:
        """Apply noise gate to reduce background noise."""
        try:
            # Calculate RMS (Root Mean Square) amplitude
            if len(audio) == 0:
                return audio

            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

            # Update noise floor estimate (adaptive)
            if rms < self.noise_floor_estimate * 2:  # Likely noise
                self.noise_floor_estimate = (
                    (1 - self.noise_floor_adaptation_rate) * self.noise_floor_estimate
                    + self.noise_floor_adaptation_rate * rms
                )

            # Apply gate threshold
            gate_threshold = self.noise_floor_estimate * 3  # 3x noise floor

            if rms < gate_threshold:
                # Attenuate low-level audio (noise)
                return (audio * 0.1).astype(np.int16)
            else:
                return audio

        except Exception as e:
            log.error(f"Error applying noise gate: {e}")
            return audio

    def _check_audio_threshold(self, audio: np.ndarray) -> bool:
        """Check if audio exceeds the activity threshold."""
        try:
            if len(audio) == 0:
                return False

            # Calculate normalized RMS
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            normalized_rms = rms / 32768.0  # Normalize to 0-1 range

            return normalized_rms > self.audio_threshold

        except Exception as e:
            log.error(f"Error checking audio threshold: {e}")
            return False

    def _update_speaker_priorities(self, current_time: float):
        """Update speaker priorities for intelligent mixing."""
        # Decay priorities over time
        for user_id in list(self.speaker_priority.keys()):
            self.speaker_priority[user_id] *= 0.99  # Slow decay

            # Remove very low priorities
            if self.speaker_priority[user_id] < 0.01:
                del self.speaker_priority[user_id]

        # Boost priority for currently speaking users
        for user_id in self.speaking_users:
            current_priority = self.speaker_priority.get(user_id, 0)
            self.speaker_priority[user_id] = min(1.0, current_priority + 0.1)

        # Select primary speaker (highest priority)
        if self.speaker_priority:
            self.primary_speaker = max(
                self.speaker_priority.keys(), key=lambda uid: self.speaker_priority[uid]
            )
        else:
            self.primary_speaker = None

    async def _send_mixed_audio(self):
        """Send mixed audio to OpenAI Realtime API."""
        if not self.speaking_users:
            return

        try:
            # Collect audio from active speakers
            audio_to_mix = []

            if self.mix_multiple_speakers:
                # Mix multiple speakers with priorities
                for user_id in self.speaking_users:
                    if user_id in self.audio_buffer:
                        chunks = self.audio_buffer[user_id]
                        if chunks:
                            processed = self._process_audio_chunks(chunks=chunks, user_id=user_id)
                            if processed:
                                priority = self.speaker_priority.get(user_id, 1.0)
                                audio_to_mix.append((processed, priority))
            else:
                # Use only primary speaker
                if self.primary_speaker and self.primary_speaker in self.audio_buffer:
                    chunks = self.audio_buffer[self.primary_speaker]
                    if chunks:
                        processed = self._process_audio_chunks(
                            chunks=chunks, user_id=self.primary_speaker
                        )
                        if processed:
                            audio_to_mix.append((processed, 1.0))

            # Mix audio streams
            if audio_to_mix:
                mixed_audio = self._mix_audio_streams(audio_to_mix)
                if mixed_audio:
                    await self.realtime_client.send_audio(mixed_audio)

        except Exception as e:
            log.error(f"Error sending mixed audio: {e}")

    def _mix_audio_streams(self, audio_streams: List[tuple]) -> Optional[bytes]:
        """Mix multiple audio streams with weighted priorities."""
        if not audio_streams:
            return None

        try:
            # Find the longest audio stream for padding
            max_length = max(len(audio) for audio, _ in audio_streams)

            # Convert all streams to numpy arrays and pad
            mixed_array = np.zeros(
                max_length // 2, dtype=np.float64
            )  # 16-bit = 2 bytes per sample

            total_weight = sum(priority for _, priority in audio_streams)

            for audio_data, priority in audio_streams:
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(
                    np.float64
                )

                # Pad if necessary
                if len(audio_array) < len(mixed_array):
                    padded_array = np.zeros(len(mixed_array))
                    padded_array[: len(audio_array)] = audio_array
                    audio_array = padded_array
                elif len(audio_array) > len(mixed_array):
                    audio_array = audio_array[: len(mixed_array)]

                # Add weighted audio
                weight = priority / total_weight
                mixed_array += audio_array * weight

            # Normalize and convert back to int16
            # Apply soft limiting to prevent clipping
            max_val = np.max(np.abs(mixed_array))
            if max_val > 32767:
                mixed_array = mixed_array * (32767 / max_val)

            return mixed_array.astype(np.int16).tobytes()

        except Exception as e:
            log.error(f"Error mixing audio streams: {e}")
            return None

    def _update_performance_stats(self, current_time: float):
        """Update performance statistics."""
        if current_time - self.last_stats_time > 30:  # Log stats every 30 seconds
            total_frames = self.frames_processed + self.frames_dropped
            if total_frames > 0:
                success_rate = (self.frames_processed / total_frames) * 100
                log.info(
                    "Audio processing stats: %d processed, %d dropped (%.1f%% success)",
                    self.frames_processed,
                    self.frames_dropped,
                    success_rate,
                )

            # Reset counters
            self.frames_processed = 0
            self.frames_dropped = 0
            self.last_stats_time = current_time

    def add_audio_data(self, user_id: int, audio_data: bytes):
        """Add audio data for a specific user."""
        if not self.is_recording or not audio_data:
            return

        # Initialize buffer for new users
        if user_id not in self.audio_buffer:
            self.audio_buffer[user_id] = []

        # Add audio data
        self.audio_buffer[user_id].append(audio_data)

        # Prevent memory buildup
        if len(self.audio_buffer[user_id]) > 200:  # ~4 seconds at 50fps
            self.audio_buffer[user_id] = self.audio_buffer[user_id][-100:]

    def set_noise_gate(self, enabled: bool):
        """Enable or disable noise gate."""
        self.noise_gate_enabled = enabled
        log.info("Noise gate %s", "enabled" if enabled else "disabled")

    def set_audio_threshold(self, threshold: float):
        """Set audio activity threshold."""
        self.audio_threshold = max(0.001, min(1.0, threshold))
        log.info("Audio threshold set to %s", self.audio_threshold)

    def set_speaker_mixing(self, mix_multiple: bool):
        """Enable or disable multiple speaker mixing."""
        self.mix_multiple_speakers = mix_multiple
        log.info("Multiple speaker mixing %s", "enabled" if mix_multiple else "disabled")

    @property
    def active_speakers(self) -> List[int]:
        """Get list of currently active speakers."""
        return list(self.speaking_users)

    @property
    def audio_stats(self) -> Dict[str, Any]:
        """Get current audio statistics."""
        return {
            "is_recording": self.is_recording,
            "active_speakers": len(self.speaking_users),
            "primary_speaker": self.primary_speaker,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "noise_floor": self.noise_floor_estimate,
            "audio_threshold": self.audio_threshold,
            "silence_threshold_ms": self.silence_threshold,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


# Define the EnhancedAudioSink with a conditional base class so imports are safe
if VOICE_RECV_AVAILABLE and _voice_recv is not None:  # pragma: no cover - runtime check
    class EnhancedAudioSink(_voice_recv.AudioSink):  # type: ignore
        """Audio sink that forwards user audio to VoiceCapture buffers."""

        def __init__(self, capture: VoiceCapture):
            super().__init__()
            self.capture = capture

        def wants_opus(self) -> bool:
            """Return False to receive decoded PCM audio data instead of opus packets."""
            return False

        def write(self, data: bytes, user: discord.User) -> None:  # signature from voice-recv
            try:
                user_id = getattr(user, "id", None)
                if user_id is None:
                    # Some implementations may pass user_id directly
                    user_id = int(user)
                self.capture.add_audio_data(int(user_id), data)
            except Exception as e:
                log.error(f"Error in EnhancedAudioSink.write: {e}")

        def cleanup(self) -> None:
            # Called by the library when recording stops
            pass
else:
    class EnhancedAudioSink:  # Fallback placeholder; instantiation is guarded in start()
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
            raise RuntimeError(
                "EnhancedAudioSink requires discord-ext-voice-recv. "
                "Install it with: pip install \"git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git\""
            )
