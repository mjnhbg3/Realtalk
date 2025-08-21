"""
OpenAI Realtime API Client for RealTalk

Enhanced client with improved audio streaming, VAD configuration,
better session management, and comprehensive error handling.
"""

import asyncio
import json
import logging
import time
import base64
from typing import Optional, Callable, Dict, Any

try:
    import websockets
    import websockets.exceptions
except ImportError:
    raise ImportError(
        "websockets package is required but not available. "
        "This should be installed automatically with discord.py. "
        "Try restarting your bot or reinstalling the cog."
    )

log = logging.getLogger("red.realtalk.realtime")


class RealtimeClient:
    """
    OpenAI Realtime API WebSocket client with enhanced features:
    - Improved audio streaming with backpressure handling
    - Reduced latency VAD (300ms silence threshold)
    - Better session configuration for voice chat
    - Enhanced event handling with user feedback
    - Connection management with timeouts and retries
    """

    def __init__(self, api_key: str, model: Optional[str] = None, voice: Optional[str] = None, transcribe: Optional[str] = None, server_vad: Optional[Dict[str, Any]] = None, instructions: Optional[str] = None):
        self.api_key = api_key
        self.model = model or "gpt-4o-realtime-preview-2024-10-01"
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.session_active = False
        
        # Event callbacks
        self.on_audio_output: Optional[Callable[[bytes], None]] = None
        self.on_conversation_item_created: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_response_audio_transcript_done: Optional[Callable[[str], None]] = None
        self.on_input_audio_buffer_speech_started: Optional[Callable[[], None]] = None
        self.on_input_audio_buffer_speech_stopped: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_response_started: Optional[Callable[[], None]] = None
        self.on_response_done: Optional[Callable[[], None]] = None
        
        # Connection management
        self.connection_timeout = 30.0
        self.reconnect_attempts = 3
        self.reconnect_delay = 2.0
        
        # Audio streaming with rate control
        self.audio_queue = asyncio.Queue(maxsize=50)  # Reduced for better rate control
        self.audio_task: Optional[asyncio.Task] = None
        self._last_audio_sent_ts: float = 0.0
        self._awaiting_commit: bool = False
        self._buffered_ms: float = 0.0
        self.input_sample_rate: int = 24000  # we downsample capture to 24k before sending
        self._response_active: bool = False
        self._last_output_ts: float = 0.0  # Track output timing for rate control
        
        # Session state
        self.session_config = {
            "modalities": ["text", "audio"],
            "instructions": instructions or "You are a helpful AI assistant having a voice conversation. Be conversational, concise, and natural. Respond as if you're talking to a friend.",
            "voice": (voice or "Alloy").lower(),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            # Note: Realtime API currently infers sample rates; we downsample input to 24k
            **({"input_audio_transcription": {"model": transcribe}} if transcribe and transcribe != "none" else {}),
            "turn_detection": {
                "type": "server_vad",
                "threshold": (server_vad or {}).get("threshold", 0.5),
                "prefix_padding_ms": 300,
                "silence_duration_ms": (server_vad or {}).get("silence_ms", 300),
                "create_response": True
            },
            "tools": [],
            "tool_choice": "auto",
            "temperature": 0.8,
            "max_response_output_tokens": "inf"
        }

    async def connect(self) -> bool:
        """Connect to OpenAI Realtime API with retry logic."""
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        for attempt in range(self.reconnect_attempts):
            try:
                log.info(f"Connecting to OpenAI Realtime API (attempt {attempt + 1}/{self.reconnect_attempts})")
                
                # Try different header parameter names for websockets compatibility
                try:
                    self.websocket = await asyncio.wait_for(
                        websockets.connect(url, extra_headers=headers),
                        timeout=self.connection_timeout
                    )
                except TypeError:
                    # Fallback for newer websockets versions
                    self.websocket = await asyncio.wait_for(
                        websockets.connect(url, additional_headers=headers),
                        timeout=self.connection_timeout
                    )
                
                self.connected = True
                log.info("✅ Successfully connected to OpenAI Realtime API")
                
                # Start background tasks
                self.audio_task = asyncio.create_task(self._audio_streaming_task())
                asyncio.create_task(self._message_handler())
                
                # Initialize session
                await self._initialize_session()
                log.info(f"🔧 Session config: {self.session_config}")
                
                return True
                
            except asyncio.TimeoutError:
                log.warning(f"Connection timeout (attempt {attempt + 1})")
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))
                    continue
                    
            except Exception as e:
                log.error(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.reconnect_attempts - 1:
                    await asyncio.sleep(self.reconnect_delay * (attempt + 1))
                    continue
                    
        log.error("Failed to connect to OpenAI Realtime API")
        return False

    async def disconnect(self):
        """Disconnect from OpenAI Realtime API."""
        self.connected = False
        self.session_active = False
        
        # Cancel audio streaming task
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                log.warning(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                
        log.info("Disconnected from OpenAI Realtime API")

    async def send_audio(self, audio_data: bytes):
        """Send audio data to the API with backpressure handling."""
        if not self.connected or not self.session_active:
            return
            
        try:
            # Add to queue with backpressure handling
            if self.audio_queue.full():
                # Drop oldest audio if queue is full to prevent memory buildup
                try:
                    self.audio_queue.get_nowait()
                    log.debug("Dropped oldest audio chunk - queue full")
                except asyncio.QueueEmpty:
                    pass
                    
            await self.audio_queue.put(audio_data)
            self._awaiting_commit = True
            log.debug(f"Queued audio data: {len(audio_data)} bytes, queue_size: {self.audio_queue.qsize()}")
            
        except Exception as e:
            log.error(f"Error queuing audio data: {e}")

    async def _audio_streaming_task(self):
        """Background task for streaming audio data."""
        while self.connected:
            try:
                # Get audio data with timeout to prevent blocking
                try:
                    audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    audio_data = None

                if audio_data and self.session_active:
                    await self._send_audio_chunk(audio_data)
                    # Update buffered duration in ms
                    try:
                        samples = len(audio_data) // 2
                        self._buffered_ms += (samples / self.input_sample_rate) * 1000.0
                    except Exception:
                        pass
                else:
                    # With server VAD and create_response: true, OpenAI handles everything
                    if self._awaiting_commit:
                        self._awaiting_commit = False
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error in audio streaming task: {e}")
                await asyncio.sleep(0.1)

    async def _send_audio_chunk(self, audio_data: bytes):
        """Send individual audio chunk to API."""
        try:
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            await self._send_message(message)
            try:
                log.info(f"Queued audio chunk to API: {len(audio_data)} bytes")
            except Exception:
                pass
            
        except Exception as e:
            log.error(f"Error sending audio chunk: {e}")

    async def commit_audio_buffer(self):
        """Commit the audio buffer to trigger response generation."""
        if not self.session_active:
            return
            
        try:
            await self._send_message({"type": "input_audio_buffer.commit"})
            self._buffered_ms = 0.0
        except Exception as e:
            log.error(f"Error committing audio buffer: {e}")

    async def create_response(self):
        """Request response generation for the committed input."""
        if not self.session_active:
            return
        try:
            if not self._response_active:
                await self._send_message({"type": "response.create"})
                self._response_active = True
        except Exception as e:
            log.error(f"Error creating response: {e}")

    async def cancel_response(self):
        """Cancel ongoing response generation."""
        if not self.session_active:
            return
            
        try:
            if self._response_active:
                await self._send_message({"type": "response.cancel"})
                self._response_active = False
                log.debug("Response cancellation sent to OpenAI")
        except Exception as e:
            log.error(f"Error canceling response: {e}")

    async def _initialize_session(self):
        """Initialize the session with optimized settings."""
        try:
            # Update session configuration
            await self._send_message({
                "type": "session.update",
                "session": self.session_config
            })
            
            self.session_active = True
            log.info("Session initialized with voice chat optimizations")
            
        except Exception as e:
            log.error(f"Error initializing session: {e}")
            raise

    async def _send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket."""
        if not self.websocket or not self.connected:
            return
            
        try:
            await self.websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            log.warning("WebSocket connection closed during send")
            self.connected = False
            self.session_active = False
        except Exception as e:
            log.error(f"Error sending message: {e}")

    async def _message_handler(self):
        """Handle incoming messages from WebSocket."""
        while self.connected and self.websocket:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )
                
                try:
                    data = json.loads(message)
                    await self._handle_event(data)
                except json.JSONDecodeError as e:
                    log.error(f"Invalid JSON received: {e}")
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                log.info("WebSocket connection closed")
                self.connected = False
                self.session_active = False
                break
            except Exception as e:
                log.error(f"Error in message handler: {e}")
                await asyncio.sleep(0.1)

    async def _handle_event(self, event: Dict[str, Any]):
        """Handle individual events from the API."""
        event_type = event.get("type")
        
        try:
            if event_type == "session.created":
                log.info("Session created successfully")
                
            elif event_type == "session.updated":
                log.info("Session updated successfully")
                
            elif event_type == "response.created":
                log.info("✅ OpenAI response generation started")
                if self.on_response_started:
                    try:
                        self.on_response_started()
                    except Exception:
                        pass

            elif event_type == "input_audio_buffer.speech_started":
                log.info("🎤 Server VAD detected speech START")
                if self.on_input_audio_buffer_speech_started:
                    self.on_input_audio_buffer_speech_started()
                # Immediately cancel any ongoing response for barge-in
                try:
                    if self._response_active:
                        await self.cancel_response()
                        log.info("Cancelled ongoing response due to user speech")
                except Exception as e:
                    log.error(f"Error cancelling response: {e}")
                    
            elif event_type == "input_audio_buffer.speech_stopped":
                log.info("🔇 Server VAD detected speech STOP - should trigger response")
                if self.on_input_audio_buffer_speech_stopped:
                    self.on_input_audio_buffer_speech_stopped()
                # With create_response: true, OpenAI automatically commits and generates response
                self._buffered_ms = 0.0
                    
            elif event_type == "conversation.item.created":
                log.debug("Conversation item created")
                if self.on_conversation_item_created:
                    self.on_conversation_item_created(event)
                    
            elif event_type == "response.audio.delta":
                # Handle audio output
                audio_b64 = event.get("delta")
                if audio_b64 and self.on_audio_output:
                    try:
                        audio_data = base64.b64decode(audio_b64)
                        log.info(f"🔊 Received audio delta: {len(audio_data)} bytes")
                        self.on_audio_output(audio_data)
                    except Exception as e:
                        log.error(f"Error processing audio delta: {e}")
                else:
                    if not audio_b64:
                        log.warning("❌ Received response.audio.delta with no audio data")
                    if not self.on_audio_output:
                        log.warning("❌ No audio output handler configured")
                        
            elif event_type == "response.audio_transcript.done":
                transcript = event.get("transcript", "")
                log.debug(f"Audio transcript: {transcript}")
                if self.on_response_audio_transcript_done:
                    self.on_response_audio_transcript_done(transcript)
                    
            elif event_type == "response.audio.done":
                log.debug("Response audio stream completed")
                # Audio streaming for this response is complete
                    
            elif event_type == "response.done":
                log.info("✅ Response completed")
                self._response_active = False
                if self.on_response_done:
                    try:
                        self.on_response_done()
                    except Exception:
                        pass
                
            elif event_type == "input_audio_buffer.committed":
                log.debug("Audio buffer committed by server")
                # Reset buffered time as audio has been committed
                self._buffered_ms = 0.0
                
            elif event_type == "input_audio_buffer.cleared":
                log.debug("Audio buffer cleared")
                self._buffered_ms = 0.0
                
            elif event_type == "error":
                error_info = event.get("error", {})
                log.error(f"API error: {error_info}")
                if self.on_error:
                    self.on_error(event)
                    
            else:
                log.debug(f"Unhandled event type: {event_type}")
                
        except Exception as e:
            log.error(f"Error handling event {event_type}: {e}")

    def update_instructions(self, instructions: str):
        """Update the AI instructions for the session."""
        self.session_config["instructions"] = instructions
        
        if self.session_active:
            # Send session update
            asyncio.create_task(self._send_message({
                "type": "session.update", 
                "session": {"instructions": instructions}
            }))

    def update_voice(self, voice: str):
        """Update the AI voice for the session."""
        if voice.lower() in ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]:
            self.session_config["voice"] = voice.lower()
            
            if self.session_active:
                asyncio.create_task(self._send_message({
                    "type": "session.update",
                    "session": {"voice": voice.lower()}
                }))
        else:
            log.warning(f"Invalid voice: {voice}. Valid voices: alloy, ash, ballad, coral, echo, sage, shimmer, verse")

    def update_vad_settings(self, threshold: float = 0.5, silence_duration_ms: int = 300):
        """Update Voice Activity Detection settings."""
        self.session_config["turn_detection"] = {
            "type": "server_vad",
            "threshold": threshold,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_duration_ms,
            "create_response": True
        }
        
        if self.session_active:
            asyncio.create_task(self._send_message({
                "type": "session.update",
                "session": {"turn_detection": self.session_config["turn_detection"]}
            }))

    @property
    def is_speaking(self) -> bool:
        """Check if AI is currently speaking (has audio in queue)."""
        return not self.audio_queue.empty()

    async def wait_for_response(self, timeout: float = 10.0) -> bool:
        """Wait for AI response to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_speaking:
                return True
            await asyncio.sleep(0.1)
            
        return False
