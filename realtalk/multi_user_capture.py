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

# Import voice_recv for AudioSink base class
try:
    import discord.ext.voice_recv as voice_recv
    AudioSink = voice_recv.AudioSink
except ImportError:
    try:
        import voice_recv
        AudioSink = voice_recv.AudioSink
    except ImportError:
        # Fallback: create a basic AudioSink-compatible class
        class AudioSink:
            def wants_opus(self) -> bool:
                return False
                
            def write(self, user, data):
                pass
            
            def cleanup(self):
                pass

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
            # Override server_vad to disable response creation for per-user clients
            user_server_vad = self.realtime_config.get('server_vad', {}).copy()
            if user_server_vad:
                user_server_vad['create_response'] = False  # Per-user clients should not create responses
                log.debug(f"Per-user client config for {self.display_name}: create_response={user_server_vad.get('create_response')}")
            else:
                user_server_vad = {'create_response': False}
                log.debug(f"Per-user client config for {self.display_name}: no server_vad, using create_response=False")
            
            self.realtime_client = RealtimeClient(
                api_key=self.realtime_config['api_key'],
                model=self.realtime_config['model'],
                voice=self.realtime_config['voice'],
                transcribe=self.realtime_config['transcribe'],
                server_vad=user_server_vad,
                instructions="You are transcribing audio. Only provide transcriptions, no responses."
            )
            
            # Connect to OpenAI
            await self.realtime_client.connect()
            
            # Explicitly disable response creation after connection
            try:
                self.realtime_client.set_auto_create_response(False)
                log.debug(f"Disabled auto response creation for {self.display_name}")
            except Exception as e:
                log.warning(f"Could not disable auto response for {self.display_name}: {e}")
            
            # Set up callbacks for per-user client
            self.realtime_client.on_input_transcript = self._on_transcript_received
            
            # Set up server VAD callbacks to trigger manual commits for transcription
            self.realtime_client.on_input_audio_buffer_speech_stopped = self._on_server_vad_speech_stopped
            
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
        """Process incoming audio chunk - send to OpenAI for server VAD"""
        if not self.realtime_client or not audio_data:
            return
        
        self.total_audio_chunks += 1
        
        try:
            # Simply stream audio to OpenAI - let server VAD handle speech detection
            await self.realtime_client.send_audio(audio_data)
            
        except Exception as e:
            log.error(f"Error processing audio for {self.display_name}: {e}")
    
    def _on_server_vad_speech_stopped(self):
        """Called when server VAD detects end of speech - commit buffer for transcription"""
        try:
            log.debug(f"Server VAD speech stopped for {self.display_name} - committing audio buffer")
            # Create an async task to commit the audio buffer
            asyncio.create_task(self.realtime_client.commit_audio_buffer())
        except Exception as e:
            log.error(f"Error committing audio buffer for {self.display_name}: {e}")
    
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
        log.debug(f"Raw transcript received for {self.display_name}: '{transcript}'")
        transcript = transcript.strip()
        if not transcript:
            log.debug(f"Empty transcript from {self.display_name}, skipping")
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
                # Add turn to decision for response generation
                decision["turn"] = turn
                # Trigger bot response through main realtime client
                asyncio.create_task(self._trigger_bot_response(decision))
            
        except Exception as e:
            log.error(f"Error processing transcript from {self.display_name}: {e}")
    
    async def _trigger_bot_response(self, decision: dict):
        """Trigger bot response through main realtime client"""
        try:
            # Get the transcript text from the decision
            turn = decision.get('turn')
            if not turn or not turn.text:
                log.warning("No transcript text available for bot response")
                return
            
            # Send the transcript as a text message to the main realtime client
            await self.main_realtime_client.send_text_message(turn.text)
            
            # Create response from the text input
            await self.main_realtime_client.create_response()
            
            thread_id = decision.get('thread_id')
            if thread_id:
                log.info(f"Triggered bot response for thread {thread_id} "
                        f"(reason: {decision.get('reason')}) with text: '{turn.text}'")
            
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
        self.processor_creation_lock = asyncio.Lock()  # Prevent race conditions
        
        # Audio sink for receiving per-user audio
        self.audio_sink: Optional['UserAudioSink'] = None
        
        # Store main event loop for thread-safe audio processing
        self.main_loop = None
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        
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
            # Create processor for new users (with lock to prevent race conditions)
            if user_id not in self.user_processors:
                async with self.processor_creation_lock:
                    # Double-check inside the lock
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


class UserAudioSink(AudioSink):
    """Audio sink that receives per-user audio from Discord"""
    
    def __init__(self, voice_capture: MultiUserVoiceCapture):
        super().__init__()
        self.voice_capture = voice_capture
        self.total_packets = 0
        self.users_seen: Set[str] = set()
    
    def wants_opus(self) -> bool:
        """Return whether we want Opus-encoded audio (False = we want PCM)"""
        return False
    
    def write(self, user: discord.Member, data):
        """Called by discord.py for each user's audio packet"""
        if not user or not data:
            return
        
        try:
            self.total_packets += 1
            self.users_seen.add(str(user.id))
            
            # Extract PCM data from voice data
            audio_data = None
            
            # Debug: log data structure first time
            if self.total_packets == 1:
                log.debug(f"Audio data structure: type={type(data)}, attrs={[attr for attr in dir(data) if not attr.startswith('_')]}")
                if hasattr(data, 'packet'):
                    packet = data.packet
                    log.debug(f"Packet type={type(packet)}, attrs={[attr for attr in dir(packet) if not attr.startswith('_')]}")
            
            if hasattr(data, 'pcm') and data.pcm:
                # Direct PCM data
                audio_data = data.pcm
            elif hasattr(data, 'packet'):
                # RTPPacket object - extract payload
                packet = data.packet
                if hasattr(packet, 'payload'):
                    audio_data = packet.payload
                elif hasattr(packet, 'data'):
                    audio_data = packet.data
                elif hasattr(packet, 'pcm'):
                    audio_data = packet.pcm
                else:
                    # Try to convert packet directly
                    try:
                        audio_data = bytes(packet)
                    except (TypeError, ValueError):
                        log.warning(f"Unable to convert RTPPacket {type(packet)} to bytes")
                        return
            else:
                # Fallback - try direct conversion
                try:
                    audio_data = bytes(data)
                except (TypeError, ValueError):
                    log.warning(f"Unable to extract audio data from {type(data)}")
                    return
            
            if not audio_data:
                log.warning("No audio data extracted")
                return
            
            # Forward to voice capture (async) - thread-safe scheduling
            main_loop = self.voice_capture.main_loop
            if main_loop and not main_loop.is_closed():
                # Use the stored main loop for thread-safe scheduling
                asyncio.run_coroutine_threadsafe(
                    self.voice_capture.on_user_audio(user, audio_data), 
                    main_loop
                )
            else:
                log.warning("No main event loop available for audio processing")
            
        except Exception as e:
            log.error(f"Error in UserAudioSink.write: {e}")
    
    def cleanup(self):
        """Clean up sink"""
        log.debug(f"UserAudioSink processed {self.total_packets} packets from {len(self.users_seen)} users")