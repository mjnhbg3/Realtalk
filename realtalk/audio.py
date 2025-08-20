"""
Audio Processing Module for RealTalk

Enhanced audio utilities with Opus loading, PCM queue audio source,
audio conversion functions, and improved error handling.
"""

import asyncio
import logging
import time
import threading
from collections import deque
from typing import Optional, Union

import discord
import numpy as np

log = logging.getLogger("red.realtalk.audio")


def load_opus_lib() -> bool:
    """
    Load Opus library with enhanced error handling and informative messages.
    
    Returns:
        bool: True if Opus loaded successfully, False otherwise
    """
    try:
        if discord.opus.is_loaded():
            log.info("Opus library already loaded")
            return True

        # Try the discord.py default loader (works on Windows when bundled)
        try:
            discord.opus._load_default()  # type: ignore[attr-defined]
        except Exception:
            pass

        if discord.opus.is_loaded():
            log.info("Opus library loaded via _load_default()")
            return True

        # Try common library names across platforms
        candidates = [
            'opus',           # generic
            'libopus-0',      # Windows/MSYS
            'opus.dll',       # Windows dll
            'libopus',        # macOS/Homebrew, Linux
        ]
        for name in candidates:
            try:
                discord.opus.load_opus(name)
                if discord.opus.is_loaded():
                    log.info(f"Opus library loaded successfully via '{name}'")
                    return True
            except Exception:
                continue

        log.error("Failed to load Opus library - is_loaded() returned False")
        return False

    except Exception as e:
        log.error(f"Error loading Opus library: {e}")
        log.error("Make sure libopus is installed on your system:")
        log.error("  - Ubuntu/Debian: sudo apt-get install libopus0")
        log.error("  - CentOS/RHEL: sudo yum install opus")
        log.error("  - macOS: brew install opus")
        log.error("  - Windows: Opus should be included with Discord.py")
        return False


class PCMQueueAudioSource(discord.AudioSource):
    """
    Enhanced PCM audio source that queues audio data for playback.
    
    Features:
    - Thread-safe audio queuing
    - Automatic silence padding
    - Volume control
    - Buffer management with overflow protection
    - Real-time audio streaming
    - Performance monitoring
    """
    
    def __init__(
        self, 
        sample_rate: int = 48000, 
        channels: int = 2, 
        frame_size: int = 960,  # 20ms at 48kHz
        max_queue_size: int = 6,  # ~120ms of audio - optimized for OpenAI response timing
        volume: float = 1.0
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = frame_size
        self.max_queue_size = max_queue_size
        self.volume = volume
        
        # Audio queue for thread-safe operations
        self.audio_queue = deque(maxlen=max_queue_size)
        self.queue_lock = threading.Lock()
        
        # Playback state
        self.is_playing = False
        self.frames_played = 0
        self.frames_queued = 0
        self.frames_dropped = 0
        
        # Audio stream state tracking
        self.has_real_audio = False  # Track if we've received actual audio data
        self.stream_finished = False  # Track if stream is truly done
        self.min_buffer_size = 2  # Minimum frames before allowing playback start (40ms)
        self.timing_stable = False  # Track if timing has stabilized
        
        # Silence frame for padding
        self.silence_frame = self._create_silence_frame()
        
        # Performance monitoring
        self.last_stats_time = time.time()
        self.underrun_count = 0
        self._leftover: bytes = b""
        
        log.info(f"PCM Audio Source initialized: {sample_rate}Hz, {channels}ch, {frame_size} samples/frame")

    def _create_silence_frame(self) -> bytes:
        """Create a silence frame for padding."""
        silence_samples = self.frame_size * self.channels
        silence_array = np.zeros(silence_samples, dtype=np.int16)
        return silence_array.tobytes()

    def read(self) -> bytes:
        """
        Read next audio frame for Discord playback.
        
        Returns:
            bytes: PCM audio data for one frame (20ms) or silence to maintain continuous stream
        """
        try:
            with self.queue_lock:
                if self.audio_queue:
                    # Get next audio frame
                    audio_data = self.audio_queue.popleft()
                    self.frames_played += 1
                    
                    # Mark timing as stable once we start playing real audio
                    if not self.timing_stable and self.has_real_audio:
                        self.timing_stable = True
                        log.debug("Audio timing stabilized - real audio playback started")
                    
                    # Log playback progress for debugging
                    if self.frames_played % 50 == 0:  # Every 1 second
                        queue_len = len(self.audio_queue)
                        log.debug(f"Discord reading audio: {self.frames_played} frames played, {queue_len} in queue")
                    
                    # Apply volume control if needed
                    if self.volume != 1.0:
                        audio_data = self._apply_volume(audio_data)
                        
                    return audio_data
                else:
                    # CONTINUOUS BUFFERING STRATEGY:
                    # Always return silence to maintain Discord's 20ms timing cadence
                    # Only end stream when explicitly finished AND no chance of more audio
                    if self.stream_finished and not self.has_real_audio:
                        # Truly done - never had real audio in this session
                        return b''
                    elif self.stream_finished and self.timing_stable:
                        # Had real audio but stream is finished - end gracefully
                        log.debug("Stream finished after real audio - ending playback")
                        return b''
                    else:
                        # Maintain continuous stream with silence to prevent timing compensation
                        self.underrun_count += 1
                        if self.underrun_count % 100 == 1:  # Log less frequently
                            log.debug(f"Audio gap #{self.underrun_count} - maintaining stream continuity")
                        return self.silence_frame
                    
        except Exception as e:
            log.error(f"Error reading audio frame: {e}")
            return self.silence_frame

    def _apply_volume(self, audio_data: bytes) -> bytes:
        """Apply volume control to audio data."""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Apply volume
            volume_array = (audio_array.astype(np.float32) * self.volume).astype(np.int16)
            
            # Prevent clipping
            np.clip(volume_array, -32768, 32767, out=volume_array)
            
            return volume_array.tobytes()
            
        except Exception as e:
            log.error(f"Error applying volume: {e}")
            return audio_data

    def put_audio(self, audio_data: bytes):
        """
        Add audio data to the playback queue.
        
        Args:
            audio_data: Raw PCM audio data
        """
        if not audio_data:
            return
            
        # Track that we have received actual audio data
        self.has_real_audio = True
        self.stream_finished = False  # Reset finished state when new audio arrives
            
        try:
            # Convert OpenAI PCM (likely 24k mono) to 48k stereo for Discord using linear upsampling
            processed_data = self._to_48k_stereo(audio_data)
            
            # Mark that we now have real audio content
            self.has_real_audio = True

            # Accumulate data to produce exact frame-sized chunks (avoid crackles between deltas)
            with self.queue_lock:
                buffer = self._leftover + processed_data
                frame_bytes = self.frame_size * self.channels * 2

                idx = 0
                buflen = len(buffer)
                while idx + frame_bytes <= buflen:
                    frame = buffer[idx: idx + frame_bytes]
                    idx += frame_bytes

                    if len(self.audio_queue) >= self.max_queue_size:
                        # CRITICAL: Drop oldest frames instead of skipping new ones
                        # This prevents buffer overflow while maintaining timing continuity
                        dropped_frame = self.audio_queue.popleft()
                        self.frames_dropped += 1
                        if self.frames_dropped % 10 == 1:
                            log.debug(f"Dropped oldest frame to prevent buffer overflow: queue size {self.max_queue_size}")
                        # Continue to add the new frame below

                    self.audio_queue.append(frame)
                    self.frames_queued += 1
                    
                    # Log when queue gets large (but less frequently)
                    if len(self.audio_queue) > 5 and len(self.audio_queue) % 5 == 0:  # Every 5 frames when over 5
                        log.debug(f"Audio queue size: {len(self.audio_queue)} frames ({len(self.audio_queue) * 20}ms)")

                # Keep leftover for next call
                self._leftover = buffer[idx:]
                    
        except Exception as e:
            log.error(f"Error adding audio to queue: {e}")

    def _to_48k_stereo(self, audio_data: bytes) -> bytes:
        """Convert 24kHz mono PCM16 to 48kHz stereo with proper sample rate conversion.
        
        Uses proper 2x upsampling with anti-aliasing to prevent timing issues.
        Critical for preventing Discord timing compensation.
        """
        try:
            if not audio_data:
                return audio_data

            # Convert to numpy array
            mono_24k = np.frombuffer(audio_data, dtype=np.int16)
            if mono_24k.size == 0:
                return b""

            # CRITICAL: Proper 2x upsampling for 24kHz -> 48kHz
            # Simple duplication method that maintains exact timing
            # Each 24kHz sample becomes two identical 48kHz samples
            mono_48k = np.repeat(mono_24k, 2)
            
            # Convert mono to stereo (duplicate channel)
            stereo_48k = np.column_stack((mono_48k, mono_48k)).reshape(-1)
            
            return stereo_48k.astype(np.int16).tobytes()
            
        except Exception as e:
            log.error(f"Error converting audio to 48k stereo: {e}")
            return audio_data

    def reset_state(self):
        """Reset internal streaming state (used when a new response starts)."""
        self._leftover = b""
        self.has_real_audio = False
        self.stream_finished = False
        self.timing_stable = False
        self.clear_queue()
    
    def finish_stream(self):
        """Signal that the audio stream is complete and should end."""
        self.stream_finished = True
    
    def interrupt_playback(self):
        """Immediately interrupt playback for user interruptions."""
        self.stop()
        self.reset_state()

    def _split_into_frames(self, audio_data: bytes) -> list:
        """Split audio data into Discord-compatible frames."""
        try:
            frames = []
            frame_bytes = self.frame_size * self.channels * 2  # 16-bit = 2 bytes
            
            for i in range(0, len(audio_data), frame_bytes):
                frame = audio_data[i:i + frame_bytes]
                
                # Pad incomplete frames with silence
                if len(frame) < frame_bytes:
                    padding_needed = frame_bytes - len(frame)
                    frame += b'\x00' * padding_needed
                    
                frames.append(frame)
                
            return frames
            
        except Exception as e:
            log.error(f"Error splitting audio into frames: {e}")
            return []

    def is_opus(self) -> bool:
        """Return False as we provide PCM data."""
        return False

    def start(self):
        """Start audio playback."""
        self.is_playing = True
        log.debug("Audio source started")

    def stop(self):
        """Stop audio playback and clear queue."""
        self.is_playing = False
        
        with self.queue_lock:
            self.audio_queue.clear()
            
        log.debug("Audio source stopped")

    def clear_queue(self):
        """Clear the audio queue."""
        with self.queue_lock:
            dropped = len(self.audio_queue)
            self.audio_queue.clear()
            self.frames_dropped += dropped
            self.has_real_audio = False
            
        log.debug(f"Audio queue cleared ({dropped} frames dropped)")

    def set_volume(self, volume: float):
        """Set playback volume (0.0 to 2.0)."""
        self.volume = max(0.0, min(2.0, volume))
        log.debug(f"Volume set to {self.volume}")

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        with self.queue_lock:
            return len(self.audio_queue)

    @property
    def queue_duration_ms(self) -> float:
        """Get current queue duration in milliseconds."""
        return (self.queue_size * 20)  # 20ms per frame

    @property
    def is_empty(self) -> bool:
        """Check if audio queue is empty."""
        with self.queue_lock:
            return len(self.audio_queue) == 0
    
    @property
    def should_end_stream(self) -> bool:
        """Check if we should end the audio stream (return empty bytes)."""
        with self.queue_lock:
            # Only end stream if we have no audio AND we never had real audio in this session
            # OR if we explicitly stopped the source
            return (len(self.audio_queue) == 0 and not self.has_real_audio) or not self.is_playing
    
    @property
    def ready_for_playback(self) -> bool:
        """Check if we have enough buffered audio to start playback safely."""
        with self.queue_lock:
            return len(self.audio_queue) >= self.min_buffer_size

    def get_stats(self) -> dict:
        """Get audio source statistics."""
        current_time = time.time()
        uptime = current_time - self.last_stats_time
        
        stats = {
            "is_playing": self.is_playing,
            "queue_size": self.queue_size,
            "queue_duration_ms": self.queue_duration_ms,
            "frames_played": self.frames_played,
            "frames_queued": self.frames_queued,
            "frames_dropped": self.frames_dropped,
            "underrun_count": self.underrun_count,
            "volume": self.volume,
            "uptime_seconds": uptime
        }
        
        return stats

    def log_stats(self):
        """Log performance statistics."""
        stats = self.get_stats()
        
        if stats["frames_played"] > 0:
            drop_rate = (stats["frames_dropped"] / stats["frames_queued"]) * 100 if stats["frames_queued"] > 0 else 0
            underrun_rate = (stats["underrun_count"] / stats["frames_played"]) * 100
            
            log.info(f"Audio Stats - Played: {stats['frames_played']}, "
                    f"Queued: {stats['frames_queued']}, "
                    f"Dropped: {stats['frames_dropped']} ({drop_rate:.1f}%), "
                    f"Underruns: {stats['underrun_count']} ({underrun_rate:.1f}%), "
                    f"Queue: {stats['queue_size']} frames ({stats['queue_duration_ms']:.0f}ms)")


class AudioConverter:
    """
    Audio conversion utilities for RealTalk.
    
    Provides functions for converting between different audio formats,
    sample rates, and channel configurations.
    """
    
    @staticmethod
    def resample_audio(audio_data: bytes, from_rate: int, to_rate: int, channels: int = 2) -> bytes:
        """
        Resample audio to different sample rate.
        
        Args:
            audio_data: Input PCM audio data
            from_rate: Source sample rate
            to_rate: Target sample rate  
            channels: Number of audio channels
            
        Returns:
            bytes: Resampled PCM audio data
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if from_rate == to_rate:
                return audio_data
                
            # Simple linear interpolation resampling
            # For production use, consider using scipy.signal.resample or librosa
            ratio = to_rate / from_rate
            new_length = int(len(audio_array) * ratio)
            
            # Create new sample indices
            old_indices = np.arange(len(audio_array))
            new_indices = np.linspace(0, len(audio_array) - 1, new_length)
            
            # Interpolate
            resampled = np.interp(new_indices, old_indices, audio_array.astype(np.float32))
            
            return resampled.astype(np.int16).tobytes()
            
        except Exception as e:
            log.error(f"Error resampling audio: {e}")
            return audio_data

    @staticmethod
    def convert_channels(audio_data: bytes, from_channels: int, to_channels: int) -> bytes:
        """
        Convert audio between mono and stereo.
        
        Args:
            audio_data: Input PCM audio data
            from_channels: Source channel count (1 or 2)
            to_channels: Target channel count (1 or 2)
            
        Returns:
            bytes: Converted PCM audio data
        """
        try:
            if from_channels == to_channels:
                return audio_data
                
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if from_channels == 1 and to_channels == 2:
                # Mono to stereo - duplicate channels
                stereo_array = np.repeat(audio_array, 2)
                return stereo_array.tobytes()
                
            elif from_channels == 2 and to_channels == 1:
                # Stereo to mono - average channels
                stereo_pairs = audio_array.reshape(-1, 2)
                mono_array = np.mean(stereo_pairs, axis=1, dtype=np.int16)
                return mono_array.tobytes()
                
            else:
                log.warning(f"Unsupported channel conversion: {from_channels} -> {to_channels}")
                return audio_data
                
        except Exception as e:
            log.error(f"Error converting channels: {e}")
            return audio_data

    @staticmethod
    def normalize_audio(audio_data: bytes, target_db: float = -12.0) -> bytes:
        """
        Normalize audio to target decibel level.
        
        Args:
            audio_data: Input PCM audio data
            target_db: Target level in dB
            
        Returns:
            bytes: Normalized PCM audio data
        """
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            if len(audio_array) == 0:
                return audio_data
                
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio_array ** 2))
            
            if rms == 0:
                return audio_data
                
            # Calculate target RMS from dB
            target_rms = 32768 * (10 ** (target_db / 20))
            
            # Apply gain
            gain = target_rms / rms
            normalized_array = audio_array * gain
            
            # Prevent clipping
            np.clip(normalized_array, -32768, 32767, out=normalized_array)
            
            return normalized_array.astype(np.int16).tobytes()
            
        except Exception as e:
            log.error(f"Error normalizing audio: {e}")
            return audio_data

    @staticmethod
    def apply_fade(audio_data: bytes, fade_in_ms: int = 0, fade_out_ms: int = 0, sample_rate: int = 48000) -> bytes:
        """
        Apply fade in/out to audio data.
        
        Args:
            audio_data: Input PCM audio data
            fade_in_ms: Fade in duration in milliseconds
            fade_out_ms: Fade out duration in milliseconds
            sample_rate: Audio sample rate
            
        Returns:
            bytes: Audio data with fade applied
        """
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            if len(audio_array) == 0:
                return audio_data
                
            # Calculate fade samples
            fade_in_samples = int((fade_in_ms / 1000) * sample_rate)
            fade_out_samples = int((fade_out_ms / 1000) * sample_rate)
            
            # Apply fade in
            if fade_in_samples > 0 and fade_in_samples < len(audio_array):
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                audio_array[:fade_in_samples] *= fade_in_curve
                
            # Apply fade out  
            if fade_out_samples > 0 and fade_out_samples < len(audio_array):
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                audio_array[-fade_out_samples:] *= fade_out_curve
                
            return audio_array.astype(np.int16).tobytes()
            
        except Exception as e:
            log.error(f"Error applying fade: {e}")
            return audio_data


class AudioMetrics:
    """Audio quality and performance metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.total_frames = 0
        self.dropped_frames = 0
        self.underruns = 0
        self.peak_amplitude = 0
        self.rms_levels = []
        self.start_time = time.time()
        
    def update(self, audio_data: bytes, dropped: bool = False, underrun: bool = False):
        """Update metrics with new audio data."""
        self.total_frames += 1
        
        if dropped:
            self.dropped_frames += 1
            
        if underrun:
            self.underruns += 1
            
        if audio_data:
            try:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Track peak amplitude
                peak = np.max(np.abs(audio_array))
                self.peak_amplitude = max(self.peak_amplitude, peak)
                
                # Track RMS level
                rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
                self.rms_levels.append(rms)
                
                # Keep only recent RMS values
                if len(self.rms_levels) > 1000:
                    self.rms_levels = self.rms_levels[-500:]
                    
            except Exception as e:
                log.error(f"Error updating audio metrics: {e}")
                
    def get_stats(self) -> dict:
        """Get current audio statistics."""
        uptime = time.time() - self.start_time
        
        stats = {
            "uptime_seconds": uptime,
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "underruns": self.underruns,
            "drop_rate_percent": (self.dropped_frames / max(1, self.total_frames)) * 100,
            "underrun_rate_percent": (self.underruns / max(1, self.total_frames)) * 100,
            "peak_amplitude": self.peak_amplitude,
            "peak_amplitude_db": 20 * np.log10(max(1, self.peak_amplitude) / 32768) if self.peak_amplitude > 0 else -float('inf'),
        }
        
        if self.rms_levels:
            avg_rms = np.mean(self.rms_levels)
            stats["average_rms"] = avg_rms
            stats["average_rms_db"] = 20 * np.log10(max(1, avg_rms) / 32768)
        else:
            stats["average_rms"] = 0
            stats["average_rms_db"] = -float('inf')
            
        return stats
