"""
Multi-User Voice Capture System for RealTalk

Handles per-user audio streams with individual VAD and OpenAI Realtime connections.
Each user gets their own audio processing pipeline while routing decisions are centralized.
"""

import asyncio
import logging
import time
from typing import Dict, Set, Optional, Any
import discord

from .realtime import RealtimeClient
from .router import ConversationRouter, Turn

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

log = logging.getLogger("red.realtalk.multi_capture")

class UserAudioProcessor:
    """Processes audio for a single user with local VAD and OpenAI Realtime connection"""
    
    def __init__(self, user_id: str, display_name: str, realtime_config: Dict[str, Any], 
                 router: ConversationRouter, main_realtime_client: RealtimeClient):
        self.user_id = user_id
        self.display_name = display_name
        self.router = router
        self.main_realtime_client = main_realtime_client
        
        # Individual OpenAI Realtime client for this user's STT
        self.realtime_client: Optional[RealtimeClient] = None
        self.realtime_config = realtime_config
        
        # Local VAD state
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.last_voice_time = 0.0
        self.audio_buffer = bytearray()
        
        # VAD configuration
        self.vad_threshold = 0.004
        self.silence_ms = 300
        
        # Audio processing
        self.sample_rate = 48000
        self.frame_size = 960  # 20ms at 48kHz
        
        # Stats
        self.total_audio_chunks = 0
        self.speech_segments = 0
        self.last_transcript_time = 0.0
        
        log.info(f"Created UserAudioProcessor for {display_name} ({user_id})")
    
    async def initialize(self):
        """Initialize the user's OpenAI Realtime connection"""
        try:
            # Create individual realtime client for this user's transcription
            self.realtime_client = RealtimeClient(
                api_key=self.realtime_config['api_key'],
                model=self.realtime_config['model'],
                voice=self.realtime_config['voice'],
                transcribe=self.realtime_config['transcribe'],
                server_vad=self.realtime_config.get('server_vad'),
                instructions="You are transcribing audio. Only provide transcriptions, no responses."
            )
            
            # Connect to OpenAI
            await self.realtime_client.connect()
            
            # Set up transcript callback
            self.realtime_client.on_input_transcript = self._on_transcript_received
            
            log.info(f"Initialized realtime client for {self.display_name}")
            
        except Exception as e:
            log.error(f"Failed to initialize realtime client for {self.display_name}: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.realtime_client:
            await self.realtime_client.disconnect()
            self.realtime_client = None
        log.info(f"Cleaned up UserAudioProcessor for {self.display_name}")
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk with local VAD"""
        if not self.realtime_client or not audio_data:
            return
        
        self.total_audio_chunks += 1
        
        try:
            # Calculate audio energy for VAD
            energy = self._calculate_audio_energy(audio_data)
            now = time.time()
            
            # VAD logic
            voice_detected = energy > self.vad_threshold
            
            if voice_detected:
                self.last_voice_time = now
                
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.speech_start_time = now
                    self.audio_buffer.clear()
                    self.speech_segments += 1
                    log.debug(f"User {self.display_name} started speaking (segment {self.speech_segments})")
                
                # Buffer audio during speech
                self.audio_buffer.extend(audio_data)
                
                # Send to user's OpenAI Realtime client (streaming)
                await self.realtime_client.send_audio(audio_data)
                
            elif self.is_speaking:
                # Check for end of speech
                silence_duration = now - self.last_voice_time
                
                if silence_duration > (self.silence_ms / 1000.0):
                    # Speech ended
                    self.is_speaking = False
                    speech_duration = now - self.speech_start_time
                    log.debug(f"User {self.display_name} stopped speaking "
                             f"(duration: {speech_duration:.1f}s, buffer: {len(self.audio_buffer)} bytes)")
                    
                    # Commit audio buffer to get transcript
                    await self.realtime_client.commit_audio_buffer()
                    
        except Exception as e:
            log.error(f"Error processing audio for {self.display_name}: {e}")
    
    def _calculate_audio_energy(self, audio_data: bytes) -> float:
        """Calculate RMS energy for VAD"""
        if not audio_data:
            return 0.0
        
        try:
            if HAS_NUMPY:
                # Efficient numpy-based calculation
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                if len(samples) == 0:
                    return 0.0
                rms = np.sqrt(np.mean(samples ** 2))
                return rms / 32768.0
            else:
                # Fallback calculation without numpy
                samples = []
                for i in range(0, len(audio_data) - 1, 2):
                    sample = int.from_bytes(audio_data[i:i+2], 'little', signed=True)
                    samples.append(sample)
                
                if not samples:
                    return 0.0
                
                rms = (sum(s**2 for s in samples) / len(samples))**0.5
                return rms / 32768.0
                
        except Exception as e:
            log.error(f"Error calculating audio energy: {e}")
            return 0.0
    
    def _on_transcript_received(self, transcript: str):
        """Handle transcript from user's OpenAI Realtime API"""
        transcript = transcript.strip()
        if not transcript:
            return
        
        self.last_transcript_time = time.time()
        
        log.info(f"Transcript from {self.display_name}: '{transcript}'")
        
        try:
            # Create Turn with user identification
            turn = Turn(
                user_id=self.user_id,
                display_name=self.display_name,
                text=transcript,
                confidence=1.0,
                timestamp=time.time()
            )
            
            # Route through the conversation router
            decision = self.router.route_turn(turn)
            
            log.debug(f"Router decision for {self.display_name}: {decision['action']} "
                     f"({decision.get('reason')}, score={decision.get('score', 0):.2f})")
            
            # Handle routing decision
            if decision["action"] == "speak":
                # Trigger bot response through main realtime client
                asyncio.create_task(self._trigger_bot_response(decision))
            
        except Exception as e:
            log.error(f"Error processing transcript from {self.display_name}: {e}")
    
    async def _trigger_bot_response(self, decision: dict):
        """Trigger bot response through main realtime client"""
        try:
            # Bot responses go through the main realtime client to prevent conflicts
            await self.main_realtime_client.commit_audio_buffer()
            await self.main_realtime_client.create_response()
            
            thread_id = decision.get('thread_id')
            if thread_id:
                log.info(f"Triggered bot response for thread {thread_id} "
                        f"(reason: {decision.get('reason')})")
            
        except Exception as e:
            log.error(f"Error triggering bot response: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        now = time.time()
        return {
            'user_id': self.user_id,
            'display_name': self.display_name,
            'is_speaking': self.is_speaking,
            'total_audio_chunks': self.total_audio_chunks,
            'speech_segments': self.speech_segments,
            'last_transcript_seconds_ago': now - self.last_transcript_time if self.last_transcript_time > 0 else -1,
            'connected': bool(self.realtime_client and self.realtime_client.connected)
        }


class MultiUserVoiceCapture:
    """Manages per-user audio processing with centralized routing"""
    
    def __init__(self, voice_client: discord.VoiceClient, router: ConversationRouter, 
                 realtime_config: Dict[str, Any], main_realtime_client: RealtimeClient):
        self.voice_client = voice_client
        self.router = router
        self.realtime_config = realtime_config
        self.main_realtime_client = main_realtime_client
        
        # Per-user processing
        self.user_processors: Dict[str, UserAudioProcessor] = {}
        self.active_speakers: Set[str] = set()
        
        # Audio sink for receiving per-user audio
        self.audio_sink: Optional['UserAudioSink'] = None
        
        # Stats
        self.total_users_processed = 0
        self.start_time = time.time()
        
        log.info("Created MultiUserVoiceCapture")
    
    async def start(self):
        """Start multi-user voice capture"""
        try:
            # Try to use per-user audio processing
            voice_recv_available = False
            try:
                import discord.ext.voice_recv as voice_recv
                voice_recv_available = True
                log.info("voice_recv extension available - per-user audio enabled")
            except ImportError:
                try:
                    import voice_recv
                    voice_recv_available = True
                    log.info("voice_recv module available - per-user audio enabled")  
                except ImportError:
                    log.warning("voice_recv not available - falling back to mixed audio processing")
            
            if voice_recv_available:
                # Create audio sink for per-user processing
                self.audio_sink = UserAudioSink(self)
                
                # Start listening to per-user audio
                self.voice_client.listen(self.audio_sink)
                log.info("Started per-user voice capture")
            else:
                # Fallback: create a single processor for mixed audio
                await self._setup_fallback_capture()
                log.info("Started fallback mixed audio capture")
            
        except Exception as e:
            log.error(f"Failed to start multi-user voice capture: {e}")
            raise
    
    async def stop(self):
        """Stop voice capture and clean up all user processors"""
        try:
            # Stop listening
            if hasattr(self.voice_client, 'stop_listening'):
                self.voice_client.stop_listening()
            
            # Clean up all user processors
            cleanup_tasks = []
            for processor in self.user_processors.values():
                cleanup_tasks.append(processor.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self.user_processors.clear()
            self.active_speakers.clear()
            
            log.info("Stopped multi-user voice capture")
            
        except Exception as e:
            log.error(f"Error stopping multi-user voice capture: {e}")
    
    async def on_user_audio(self, user: discord.Member, audio_data: bytes):
        """Handle audio data from a specific user"""
        if not user or not audio_data:
            return
        
        user_id = str(user.id)
        display_name = user.display_name or user.name
        
        try:
            # Create processor for new users
            if user_id not in self.user_processors:
                await self._create_user_processor(user_id, display_name)
            
            # Update active speakers
            self.active_speakers.add(display_name)
            
            # Update router's active speakers list
            self.router.active_speakers = self.active_speakers.copy()
            
            # Process audio through user's processor
            processor = self.user_processors[user_id]
            await processor.process_audio_chunk(audio_data)
            
        except Exception as e:
            log.error(f"Error processing audio from user {display_name}: {e}")
    
    async def _create_user_processor(self, user_id: str, display_name: str):
        """Create and initialize a new user processor"""
        try:
            log.info(f"Creating processor for new user: {display_name} ({user_id})")
            
            processor = UserAudioProcessor(
                user_id=user_id,
                display_name=display_name,
                realtime_config=self.realtime_config,
                router=self.router,
                main_realtime_client=self.main_realtime_client
            )
            
            await processor.initialize()
            
            self.user_processors[user_id] = processor
            self.total_users_processed += 1
            
            log.info(f"Successfully created processor for {display_name}")
            
        except Exception as e:
            log.error(f"Failed to create processor for {display_name}: {e}")
            raise
    
    async def on_user_disconnect(self, user: discord.Member):
        """Handle user disconnection"""
        user_id = str(user.id)
        display_name = user.display_name or user.name
        
        if user_id in self.user_processors:
            try:
                processor = self.user_processors[user_id]
                await processor.cleanup()
                del self.user_processors[user_id]
                
                # Remove from active speakers
                self.active_speakers.discard(display_name)
                self.router.active_speakers = self.active_speakers.copy()
                
                log.info(f"Cleaned up processor for disconnected user: {display_name}")
                
            except Exception as e:
                log.error(f"Error cleaning up processor for {display_name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        now = time.time()
        uptime = now - self.start_time
        
        user_stats = {}
        for user_id, processor in self.user_processors.items():
            user_stats[user_id] = processor.get_stats()
        
        return {
            'uptime_seconds': uptime,
            'active_users': len(self.user_processors),
            'total_users_processed': self.total_users_processed,
            'active_speakers': list(self.active_speakers),
            'user_processors': user_stats
        }
    
    async def _setup_fallback_capture(self):
        """Setup fallback capture for when voice_recv is not available"""
        try:
            # Create a single processor for mixed audio with generic user
            await self._create_user_processor("mixed_audio", "MixedAudio")
            log.info("Fallback capture initialized with mixed audio processor")
        except Exception as e:
            log.error(f"Failed to setup fallback capture: {e}")
            raise


class UserAudioSink:
    """Audio sink that receives per-user audio from Discord"""
    
    def __init__(self, voice_capture: MultiUserVoiceCapture):
        self.voice_capture = voice_capture
        self.total_packets = 0
        self.users_seen: Set[str] = set()
    
    def write(self, user: discord.Member, data):
        """Called by discord.py for each user's audio packet"""
        if not user or not data:
            return
        
        try:
            self.total_packets += 1
            self.users_seen.add(str(user.id))
            
            # Extract PCM data from voice data
            if hasattr(data, 'packet'):
                audio_data = data.packet
            elif hasattr(data, 'pcm'):
                audio_data = data.pcm
            else:
                audio_data = bytes(data)
            
            # Forward to voice capture (async)
            asyncio.create_task(
                self.voice_capture.on_user_audio(user, audio_data)
            )
            
        except Exception as e:
            log.error(f"Error in UserAudioSink.write: {e}")
    
    def cleanup(self):
        """Clean up sink"""
        log.debug(f"UserAudioSink processed {self.total_packets} packets from {len(self.users_seen)} users")