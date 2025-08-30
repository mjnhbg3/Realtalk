"""
RateLimitedAudioSource class - extracted from working commit f9afcd9
"""

import asyncio
import logging
import time
from typing import Optional

import discord

from .audio import PCMQueueAudioSource

log = logging.getLogger("red.realtalk.audio")


class RateLimitedAudioSource(discord.AudioSource):
    """
    Rate-limited audio source that paces OpenAI audio delivery to real-time playback speed.
    
    This class solves the Discord timing compensation issue by converting OpenAI's "burst"
    audio delivery into a consistent real-time stream that mimics live microphone input.
    
    Features:
    - Clock-based pacing to maintain 20ms frame intervals
    - Prevents Discord's timing compensation from triggering
    - Wraps existing PCMQueueAudioSource for compatibility
    - Handles network delays and audio gaps gracefully
    - Memory-efficient with bounded buffers
    """
    
    def __init__(self, 
                 target_sample_rate: int = 48000,  # Discord requires 48kHz
                 channels: int = 2, 
                 frame_size: int = 960,  # 20ms at 48kHz
                 rate_limit_enabled: bool = True,  # ENABLED by default
                 target_buffer_ms: int = 200):
        super().__init__()
        
        # Wrap existing PCMQueueAudioSource
        self.pcm_source = PCMQueueAudioSource(target_sample_rate, channels, frame_size, max_queue_size=25)
        
        # Rate control configuration
        self.rate_limit_enabled = rate_limit_enabled
        self.target_frame_interval = 0.02  # 20ms per frame (50 FPS)
        self.target_buffer_ms = target_buffer_ms
        
        # Clock-based timing state
        self.playback_start_time: Optional[float] = None
        self.frames_delivered = 0
        self.last_frame_time = 0.0
        
        # Pacing buffer for smooth delivery (bounded to prevent memory issues)
        max_buffer_frames = int((target_buffer_ms * 2) / 20)  # 2x target buffer as max
        self.pacing_queue = asyncio.Queue(maxsize=max_buffer_frames)
        self.pacing_task: Optional[asyncio.Task] = None
        self._stop_pacing = False
        
        # Audio duration tracking for rate calculation
        self.accumulated_audio_time = 0.0
        self._last_ingest_ts = time.monotonic()
        
        # Performance monitoring
        self.rate_control_stats = {
            'frames_paced': 0,
            'frames_skipped': 0,
            'timing_errors': 0,
            'buffer_overruns': 0
        }

        # Internal 24kHz mono PCM buffer so we can emit one 20ms frame per tick
        self._pcm24_buffer = bytearray()

        # Cached sizes
        self._pcm24_frame_bytes = 480 * 2  # 480 samples @24kHz mono, 16-bit
        self._pcm48_frame_bytes = frame_size * channels * 2  # 3840 bytes

        # Finish-handling: defer actual stream end until all buffers drain
        self._finish_when_drained = False
        
        log.info(f"RateLimitedAudioSource initialized: rate_limit={rate_limit_enabled}, "
                f"target_buffer={target_buffer_ms}ms")
    
    def _pump_one_20ms_frame(self) -> bool:
        """Convert one 20ms 24kHz mono slice to 48kHz stereo and enqueue it.

        Returns True if a frame was produced and queued, False otherwise.
        """
        try:
            # Ensure we have at least 20ms of 24kHz PCM available
            while len(self._pcm24_buffer) < self._pcm24_frame_bytes:
                try:
                    chunk = self.pacing_queue.get_nowait()
                    if chunk:
                        self._pcm24_buffer.extend(chunk)
                except asyncio.QueueEmpty:
                    break

            if len(self._pcm24_buffer) < self._pcm24_frame_bytes:
                return False

            # Take exactly one 20ms 24kHz mono slice
            slice24 = self._pcm24_buffer[:self._pcm24_frame_bytes]
            del self._pcm24_buffer[:self._pcm24_frame_bytes]

            # Convert to one 20ms 48kHz stereo frame and enqueue directly
            frame48 = self.pcm_source._to_48k_stereo(slice24)
            # Ensure exact frame size
            if len(frame48) != self._pcm48_frame_bytes:
                # Normalize by pad/truncate just in case
                if len(frame48) < self._pcm48_frame_bytes:
                    frame48 = frame48 + b"\x00" * (self._pcm48_frame_bytes - len(frame48))
                else:
                    frame48 = frame48[:self._pcm48_frame_bytes]
            self.pcm_source.put_48k_stereo_frame(frame48)
            return True
        except Exception as e:
            log.error(f"Error pumping 20ms frame: {e}")
            return False

    async def _audio_pacing_loop(self):
        """
        Clock-based audio pacing loop that delivers audio chunks at real-time speed.
        
        This prevents Discord timing compensation by maintaining consistent 20ms delivery cadence,
        making the audio source behave like a live microphone feed rather than a pre-buffered file.
        """
        log.debug("Starting audio pacing loop")
        loop = asyncio.get_event_loop()
        
        while self.rate_limit_enabled and not self._stop_pacing:
            try:
                # Use event loop's monotonic time instead of system time to prevent heartbeat blocking
                loop_start_time = loop.time()
                
                # Calculate expected frame time based on playback position
                if self.playback_start_time is None:
                    self.playback_start_time = loop_start_time
                    
                expected_frame_time = (self.playback_start_time + 
                                     self.frames_delivered * self.target_frame_interval)
                
                # Wait until it's time for the next frame using cached time
                sleep_duration = expected_frame_time - loop_start_time
                
                # Only sleep if we're ahead of schedule
                if sleep_duration > 0:
                    # Cap sleep duration to prevent issues with clock drift
                    sleep_duration = min(sleep_duration, self.target_frame_interval)
                    await asyncio.sleep(sleep_duration)
                elif sleep_duration < -0.1:  # More than 100ms behind
                    # We're falling behind - log and reset timing to prevent drift
                    log.debug(f"Audio pacing falling behind by {-sleep_duration:.3f}s - resetting timing")
                    self.playback_start_time = loop_start_time
                    self.frames_delivered = 0
                    self.rate_control_stats['timing_errors'] += 1
                
                # Deliver frames to maintain a small PCM buffer ahead of consumption
                desired_frames = min(15, max(self.pcm_source.min_buffer_size, self.target_buffer_ms // 20))
                produced_any = False
                produced_this_tick = 0
                while self.pcm_source.queue_size < desired_frames and produced_this_tick < 3:
                    if not self._pump_one_20ms_frame():
                        break
                    produced_any = True
                    produced_this_tick += 1
                    self.frames_delivered += 1
                    self.rate_control_stats['frames_paced'] += 1
                if self.frames_delivered % 50 == 0:  # Every ~1s if active
                    q_frames = self.pacing_queue.qsize()
                    log.debug(f"Pacer: delivered={self.frames_delivered}, pcm_queue={self.pcm_source.queue_size}, "
                              f"pacing_chunks={q_frames}")
                
                # Track actual loop timing for debugging using event loop time
                self.last_frame_time = loop.time()

                # Explicit yield point to prevent event loop blocking
                if self.frames_delivered % 10 == 0:  # Every 200ms
                    await asyncio.sleep(0)

                # If a finish was requested, end only after all buffers are drained
                if self._finish_when_drained:
                    if (self.pacing_queue.qsize() == 0 and
                        len(self._pcm24_buffer) == 0 and
                        self.pcm_source.queue_size == 0):
                        self.pcm_source.finish_stream()
                        # Stop pacing loop gracefully
                        self._stop_pacing = True
                        continue
                else:
                    # Auto-finish on inactivity when nothing buffered anywhere
                    if (self.pacing_queue.qsize() == 0 and len(self._pcm24_buffer) == 0 and
                        self.pcm_source.queue_size == 0):
                        if (loop.time() - self._last_ingest_ts) > 1.2:
                            self.pcm_source.finish_stream()
                            self._stop_pacing = True
                            continue
                
            except asyncio.CancelledError:
                log.debug("Audio pacing loop cancelled")
                break
            except Exception as e:
                log.error(f"Error in audio pacing loop: {e}")
                self.rate_control_stats['timing_errors'] += 1
                # Brief sleep to prevent tight error loops
                await asyncio.sleep(0.001)
        
        log.debug("Audio pacing loop stopped")
    
    def queue_audio_for_pacing(self, audio_data: bytes, duration_ms: float = None):
        """
        Queue audio data for rate-controlled delivery.
        
        Args:
            audio_data: Raw 24kHz mono PCM from OpenAI
            duration_ms: Expected playback duration of this chunk (calculated if None)
        """
        try:
            if not audio_data:
                return
            
            # Calculate duration if not provided (assuming 24kHz mono 16-bit)
            if duration_ms is None:
                samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
                duration_ms = (samples / 24000) * 1000  # OpenAI sends 24kHz
            
            # Track accumulated audio time for monitoring
            self.accumulated_audio_time += duration_ms
            self._last_ingest_ts = time.monotonic()
            
            # Queue for paced delivery with overflow protection
            try:
                self.pacing_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                # Buffer is full - drop oldest audio to prevent memory buildup
                try:
                    discarded = self.pacing_queue.get_nowait()
                    self.pacing_queue.put_nowait(audio_data)
                    self.rate_control_stats['buffer_overruns'] += 1
                    log.debug("Dropped oldest audio chunk to prevent buffer overflow")
                except asyncio.QueueEmpty:
                    # Race condition - queue became empty, just add new audio
                    try:
                        self.pacing_queue.put_nowait(audio_data)
                    except asyncio.QueueFull:
                        # Still full - skip this chunk
                        self.rate_control_stats['frames_skipped'] += 1
                        log.warning("Skipped audio chunk - pacing buffer full")
            
        except Exception as e:
            log.error(f"Error queuing audio for pacing: {e}")
    
    def start_pacing(self):
        """Start rate-limited audio delivery."""
        if self.rate_limit_enabled and (self.pacing_task is None or self.pacing_task.done()):
            self._stop_pacing = False
            # Use event loop time instead of system time
            try:
                loop = asyncio.get_event_loop()
                self.playback_start_time = loop.time()
            except RuntimeError:
                # Fallback if no event loop is running
                self.playback_start_time = time.time()
            self.frames_delivered = 0
            self.pacing_task = asyncio.create_task(self._audio_pacing_loop())
            log.info("Started clock-based audio pacing for rate limiting")
    
    def stop_pacing(self):
        """Stop rate-limited audio delivery."""
        self._stop_pacing = True
        if self.pacing_task and not self.pacing_task.done():
            self.pacing_task.cancel()
        log.debug("Stopped audio pacing")
    
    # AudioSource interface methods - delegate to wrapped PCM source
    
    def read(self) -> bytes:
        """Read next audio frame for Discord playback."""
        return self.pcm_source.read()
    
    def is_opus(self) -> bool:
        """Return False as we provide PCM data."""
        return self.pcm_source.is_opus()
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_pacing()
        if hasattr(self.pcm_source, 'cleanup'):
            self.pcm_source.cleanup()
    
    # Delegate methods
    
    def put_audio(self, audio_data: bytes):
        """Add audio data - routes through rate limiting if enabled."""
        # Always queue to pacing buffer; delivery decides pacing strategy
        self.queue_audio_for_pacing(audio_data)
        if self.rate_limit_enabled:
            if not self.pacing_task or self.pacing_task.done():
                self.start_pacing()

    def reset_state(self):
        """Reset internal streaming state."""
        self.stop_pacing()
        self.pcm_source.reset_state()
        
        # Clear pacing buffers
        try:
            while True:
                self.pacing_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        
        # Reset timing state
        self.playback_start_time = None
        self.frames_delivered = 0
        self.accumulated_audio_time = 0.0
        self._pcm24_buffer = bytearray()
    
    def finish_stream(self):
        """Signal that the audio stream is complete."""
        self._finish_when_drained = True
        # If there's nothing buffered anywhere, finish immediately
        if (self.pacing_queue.qsize() == 0 and len(self._pcm24_buffer) == 0 and
            self.pcm_source.queue_size == 0):
            self.pcm_source.finish_stream()
    
    def interrupt_playback(self):
        """Immediately interrupt playback for user interruptions."""
        self.stop_pacing()
        self.pcm_source.interrupt_playback()
        self.reset_state()
    
    @property
    def queue_size(self) -> int:
        """Get current PCM queue size."""
        return self.pcm_source.queue_size
    
    @property
    def ready_for_playback(self) -> bool:
        """Check if we have enough buffered audio to start playback safely."""
        if self.rate_limit_enabled:
            # For rate-limited mode, check both pacing and PCM buffers
            pacing_buffer_ms = self.pacing_queue.qsize() * 20
            return (pacing_buffer_ms >= self.target_buffer_ms // 2 and  # Half target buffer in pacing
                   self.pcm_source.ready_for_playback)  # Plus PCM buffer ready
        else:
            return self.pcm_source.ready_for_playback