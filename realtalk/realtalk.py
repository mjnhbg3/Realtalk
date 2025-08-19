"""
RealTalk - OpenAI Realtime Voice Assistant for Red-DiscordBot

A Discord cog that enables real-time voice conversations with OpenAI's Realtime API.
Includes enhanced error handling, automatic retry logic, and comprehensive voice 
connection management to handle Discord's infrastructure issues.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

import discord
from discord.ext import commands
from redbot.core import Config, commands as red_commands, checks
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box, humanize_list

from .realtime import RealtimeClient
from .capture import VoiceCapture
from .audio import PCMQueueAudioSource

log = logging.getLogger("red.realtalk")


class RealTalk(red_commands.Cog):
    """
    Real-time voice conversation with OpenAI's Realtime API.
    
    Features:
    - Two-way voice conversation in Discord voice channels
    - Automatic error recovery for Discord voice issues (4006 errors)
    - Smart audio processing with speaker detection
    - Conversation interruption support
    - Comprehensive status monitoring
    """

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=280102050, force_registration=True)
        
        # Configuration schema
        default_global = {
            "openai_api_key": None,
            "max_retry_attempts": 5,
            "retry_base_delay": 2.0,
            "voice_timeout": 30.0,
            "audio_threshold": 0.01,
            "silence_threshold": 300,  # 300ms for reduced latency
        }
        
        default_guild = {
            "enabled": True,
            "voice_channel": None,
        }
        
        self.config.register_global(**default_global)
        self.config.register_guild(**default_guild)
        
        # Active sessions tracking
        self.sessions: Dict[int, Dict[str, Any]] = {}
        
        # Voice connection retry state
        self.retry_state: Dict[int, Dict[str, Any]] = {}

    async def cog_unload(self):
        """Clean up when cog is unloaded."""
        for guild_id in list(self.sessions.keys()):
            await self._cleanup_session(guild_id)

    @red_commands.group(name="realtalk", aliases=["rt"])
    async def realtalk(self, ctx: red_commands.Context):
        """RealTalk voice assistant commands."""
        pass

    @realtalk.command(name="setup")
    async def setup_command(self, ctx: red_commands.Context):
        """Show setup instructions for RealTalk."""
        setup_msg = """
**RealTalk Setup Instructions**

**1. Install Dependencies:**
```
[p]load pip
[p]pip install "PyNaCl>=1.6.0"
[p]pip install discord-ext-voice-recv
```

**2. Set OpenAI API Key:**
```
[p]set api openai api_key YOUR_API_KEY_HERE
```

**3. Restart your bot** to ensure dependencies are loaded.

**4. Join voice and start chatting:**
```
[p]realtalk join
```

**Requirements:**
- OpenAI API key with Realtime API access
- Bot needs "Connect" and "Speak" permissions in voice channels

**Troubleshooting:**
- 4006 errors: Discord infrastructure issue - bot will auto-retry
- No audio: Make sure voice dependencies are installed and bot is restarted
- Check `[p]realtalk status` for diagnostics
        """.replace("[p]", ctx.clean_prefix)
        
        await ctx.send(setup_msg)

    @realtalk.command(name="join")
    async def join_voice(self, ctx: red_commands.Context, *, channel: Optional[discord.VoiceChannel] = None):
        """Join voice channel and start AI conversation."""
        if not await self._check_setup(ctx):
            return
            
        guild_id = ctx.guild.id
        
        # Determine voice channel
        if channel is None:
            if ctx.author.voice and ctx.author.voice.channel:
                channel = ctx.author.voice.channel
            else:
                await ctx.send("You need to be in a voice channel or specify one.")
                return
        
        # Check if already in session
        if guild_id in self.sessions:
            await ctx.send("Already in a voice session. Use `leave` first.")
            return
            
        # Check permissions
        if not channel.permissions_for(ctx.guild.me).connect:
            await ctx.send("I don't have permission to connect to that voice channel.")
            return
            
        if not channel.permissions_for(ctx.guild.me).speak:
            await ctx.send("I don't have permission to speak in that voice channel.")
            return
        
        # Initialize retry state
        self.retry_state[guild_id] = {
            "attempts": 0,
            "last_attempt": 0,
            "total_attempts": 0
        }
        
        await ctx.send(f"Connecting to {channel.mention}...")
        
        # Attempt connection with retry logic
        success = await self._connect_with_retry(ctx, channel)
        if not success:
            await ctx.send("Failed to establish voice connection after multiple attempts. "
                         "This may be due to Discord infrastructure issues (error 4006). "
                         "Please try again in a few minutes.")
            return
            
        await ctx.send(f"Connected to {channel.mention}! I'm listening for voice input. "
                      f"Speak naturally to start a conversation.")

    @realtalk.command(name="leave")
    async def leave_voice(self, ctx: red_commands.Context):
        """Leave voice channel and stop session."""
        guild_id = ctx.guild.id
        
        if guild_id not in self.sessions:
            await ctx.send("Not currently in a voice session.")
            return
            
        await self._cleanup_session(guild_id)
        await ctx.send("Left voice channel and ended AI conversation session.")

    @realtalk.command(name="status")
    async def status_command(self, ctx: red_commands.Context):
        """Show connection and session status."""
        guild_id = ctx.guild.id
        
        # Basic status
        status_lines = ["**RealTalk Status**", ""]
        
        # API key status
        api_key = await self._get_openai_api_key()
        if api_key:
            status_lines.append("✅ OpenAI API Key: Configured")
        else:
            status_lines.append("❌ OpenAI API Key: Not configured")
            
        # Voice connection status
        voice_client = ctx.guild.voice_client
        if voice_client and voice_client.is_connected():
            status_lines.append(f"✅ Voice: Connected to {voice_client.channel.mention}")
        else:
            status_lines.append("❌ Voice: Not connected")
            
        # Session status
        if guild_id in self.sessions:
            session = self.sessions[guild_id]
            status_lines.append("✅ AI Session: Active")
            
            # Realtime client status
            realtime_client = session.get("realtime_client")
            if realtime_client and realtime_client.connected:
                status_lines.append("✅ OpenAI Realtime: Connected")
            else:
                status_lines.append("❌ OpenAI Realtime: Disconnected")
                
            # Voice capture status
            voice_capture = session.get("voice_capture")
            if voice_capture and voice_capture.is_recording:
                status_lines.append("✅ Voice Capture: Active")
            else:
                status_lines.append("❌ Voice Capture: Inactive")
        else:
            status_lines.append("❌ AI Session: Inactive")
            
        # Retry statistics
        if guild_id in self.retry_state:
            retry_info = self.retry_state[guild_id]
            if retry_info["total_attempts"] > 0:
                status_lines.append("")
                status_lines.append(f"Connection attempts: {retry_info['total_attempts']}")
                
        # Dependencies status
        status_lines.append("")
        status_lines.append("**Dependencies:**")
        
        try:
            import nacl
            status_lines.append(f"✅ PyNaCl: {nacl.__version__}")
        except ImportError:
            status_lines.append("❌ PyNaCl: Not installed")
            
        try:
            import discord.ext.voice_recv
            status_lines.append("✅ discord-ext-voice-recv: Available")
        except ImportError:
            status_lines.append("❌ discord-ext-voice-recv: Not installed")
            
        await ctx.send("\n".join(status_lines))

    @realtalk.group(name="sinks")
    async def sinks_command(self, ctx: red_commands.Context):
        """Voice capture sink management."""
        pass

    @sinks_command.command(name="status")
    async def sinks_status(self, ctx: red_commands.Context):
        """Check voice capture availability."""
        try:
            import discord.ext.voice_recv
            await ctx.send("✅ Voice capture is available (discord-ext-voice-recv installed)")
        except ImportError:
            await ctx.send("❌ Voice capture not available. Install with:\n"
                         f"`{ctx.clean_prefix}pip install discord-ext-voice-recv`")

    @realtalk.command(name="set")
    async def set_config(self, ctx: red_commands.Context, key: str, *, value: str):
        """Set configuration values."""
        if not await checks.is_owner(ctx):
            await ctx.send("Only the bot owner can modify configuration.")
            return
            
        if key == "key":
            await self.config.openai_api_key.set(value)
            await ctx.send("OpenAI API key updated.")
        else:
            await ctx.send(f"Unknown configuration key: {key}")

    async def _check_setup(self, ctx: red_commands.Context) -> bool:
        """Check if RealTalk is properly set up."""
        # Check API key
        api_key = await self._get_openai_api_key()
        if not api_key:
            await ctx.send("OpenAI API key not configured. Use:\n"
                         f"`{ctx.clean_prefix}set api openai api_key YOUR_KEY`\n"
                         f"or `{ctx.clean_prefix}realtalk set key YOUR_KEY`")
            return False
            
        # Check voice dependencies
        try:
            import nacl
            import discord.ext.voice_recv
        except ImportError as e:
            await ctx.send(f"Missing dependencies. Run setup command:\n"
                         f"`{ctx.clean_prefix}realtalk setup`")
            return False
            
        return True

    async def _get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from various sources."""
        # Try config first
        api_key = await self.config.openai_api_key()
        if api_key:
            return api_key
            
        # Try Red's API system
        try:
            api_tokens = await self.bot.get_shared_api_tokens("openai")
            return api_tokens.get("api_key")
        except Exception:
            pass
            
        # Try environment variable
        import os
        return os.getenv("OPENAI_API_KEY")

    async def _connect_with_retry(self, ctx: red_commands.Context, channel: discord.VoiceChannel) -> bool:
        """Connect to voice channel with exponential backoff retry logic."""
        guild_id = ctx.guild.id
        retry_state = self.retry_state[guild_id]
        max_attempts = await self.config.max_retry_attempts()
        base_delay = await self.config.retry_base_delay()
        
        for attempt in range(max_attempts):
            retry_state["attempts"] = attempt + 1
            retry_state["total_attempts"] += 1
            retry_state["last_attempt"] = time.time()
            
            try:
                log.info(f"Voice connection attempt {attempt + 1}/{max_attempts} for guild {guild_id}")
                
                # Try to connect
                voice_client = await channel.connect(
                    timeout=await self.config.voice_timeout(),
                    reconnect=True
                )
                
                if voice_client and voice_client.is_connected():
                    log.info(f"Voice connection successful for guild {guild_id}")
                    await self._start_session(ctx, voice_client)
                    return True
                    
            except discord.errors.ConnectionClosed as e:
                log.warning(f"Voice connection closed (attempt {attempt + 1}): {e}")
                if e.code == 4006:
                    # Session no longer valid - this is the common Discord infrastructure issue
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        log.info(f"Retrying in {delay}s due to 4006 error...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    log.error(f"Voice connection error {e.code}: {e}")
                    break
                    
            except Exception as e:
                log.error(f"Voice connection error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                break
        
        log.error(f"Failed to connect to voice after {max_attempts} attempts")
        return False

    async def _start_session(self, ctx: red_commands.Context, voice_client: discord.VoiceClient):
        """Start AI conversation session."""
        guild_id = ctx.guild.id
        api_key = await self._get_openai_api_key()
        
        try:
            # Initialize Realtime client
            realtime_client = RealtimeClient(api_key)
            
            # Initialize voice capture
            voice_capture = VoiceCapture(
                voice_client=voice_client,
                realtime_client=realtime_client,
                audio_threshold=await self.config.audio_threshold(),
                silence_threshold=await self.config.silence_threshold()
            )
            
            # Initialize audio source for bot speech
            audio_source = PCMQueueAudioSource()
            
            # Store session
            self.sessions[guild_id] = {
                "voice_client": voice_client,
                "realtime_client": realtime_client,
                "voice_capture": voice_capture,
                "audio_source": audio_source,
                "context": ctx,
                "start_time": time.time()
            }
            
            # Connect to OpenAI Realtime API
            await realtime_client.connect()
            
            # Set up audio output callback
            realtime_client.on_audio_output = lambda audio_data: audio_source.put_audio(audio_data)
            
            # Start voice capture
            await voice_capture.start()
            
            # Start playing audio source
            voice_client.play(audio_source)
            
            log.info(f"Started AI conversation session for guild {guild_id}")
            
        except Exception as e:
            log.error(f"Failed to start session: {e}")
            await self._cleanup_session(guild_id)
            await ctx.send(f"Failed to start AI session: {e}")

    async def _cleanup_session(self, guild_id: int):
        """Clean up session resources."""
        if guild_id not in self.sessions:
            return
            
        session = self.sessions[guild_id]
        
        try:
            # Stop voice capture
            voice_capture = session.get("voice_capture")
            if voice_capture:
                await voice_capture.stop()
                
            # Disconnect realtime client
            realtime_client = session.get("realtime_client")
            if realtime_client:
                await realtime_client.disconnect()
                
            # Stop audio source
            audio_source = session.get("audio_source")
            if audio_source:
                audio_source.stop()
                
            # Disconnect voice client
            voice_client = session.get("voice_client")
            if voice_client and voice_client.is_connected():
                await voice_client.disconnect()
                
        except Exception as e:
            log.error(f"Error during session cleanup: {e}")
        finally:
            # Remove session
            del self.sessions[guild_id]
            
            # Clean up retry state
            if guild_id in self.retry_state:
                del self.retry_state[guild_id]
                
        log.info(f"Cleaned up session for guild {guild_id}")

    @red_commands.Cog.listener()
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """Handle voice state updates for automatic cleanup."""
        if member != self.bot.user:
            return
            
        guild_id = member.guild.id
        
        # If bot was disconnected from voice
        if before.channel and not after.channel and guild_id in self.sessions:
            log.info(f"Bot disconnected from voice in guild {guild_id}, cleaning up session")
            await self._cleanup_session(guild_id)