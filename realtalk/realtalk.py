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
        
        # Version identifier for enhanced cog
        self.__version__ = "2.0.0-enhanced"

    async def cog_unload(self):
        """Clean up when cog is unloaded."""
        for guild_id in list(self.sessions.keys()):
            await self._cleanup_session(guild_id)

    @red_commands.group(name="realtalk", aliases=["rt"])
    async def realtalk(self, ctx: red_commands.Context):
        """RealTalk voice assistant commands."""
        pass
    
    @realtalk.command(name="version")
    async def version_command(self, ctx: red_commands.Context):
        """Show RealTalk version and discord.py compatibility info."""
        embed = discord.Embed(
            title=f"🤖 RealTalk v{self.__version__}",
            description="Enhanced Discord voice assistant with 4006 error fixes",
            color=0x00ff00
        )
        
        # Check discord.py version
        try:
            version = discord.__version__
            major, minor = map(int, version.split('.')[:2])
            
            if major < 2 or (major == 2 and minor < 4):
                embed.add_field(
                    name="⚠️ Discord.py Version Warning",
                    value=f"Current: {version}\n"
                          "Recommended: ≥2.4.0 for voice protocol v8\n"
                          f"**Fix**: `{ctx.clean_prefix}pipinstall -U discord.py`",
                    inline=False
                )
                embed.color = 0xff9900
            else:
                embed.add_field(
                    name="✅ Discord.py Compatible",
                    value=f"Version {version} supports voice protocol v8",
                    inline=False
                )
        except Exception:
            embed.add_field(
                name="❌ Discord.py Check Failed",
                value="Could not determine discord.py version",
                inline=False
            )
            
        # Check PyNaCl
        try:
            import nacl
            embed.add_field(
                name="✅ PyNaCl",
                value="Available for voice encryption",
                inline=True
            )
        except ImportError:
            embed.add_field(
                name="❌ PyNaCl Missing",
                value=f"Install: `{ctx.clean_prefix}pipinstall PyNaCl>=1.5.0`",
                inline=True
            )
            
        embed.set_footer(text="Use realtalk setup for installation guide")
        await ctx.send(embed=embed)

    @realtalk.command(name="setup")
    async def setup_command(self, ctx: red_commands.Context):
        """Show setup instructions for RealTalk."""
        setup_msg = """
**RealTalk Setup Instructions**

**1. Dependencies:** All dependencies are installed automatically when you install this cog.

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
- No audio: Make sure bot is restarted after installation
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
        except ImportError as e:
            await ctx.send(f"❌ Voice capture not available: {e}\n"
                         f"This dependency should have been installed automatically.\n"
                         f"Try: `{ctx.clean_prefix}cog update` and `{ctx.clean_prefix}cog reload realtalk`\n"
                         f"Or manual install: `{ctx.clean_prefix}pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git`")
                         
    @sinks_command.command(name="debug")
    async def sinks_debug(self, ctx: red_commands.Context):
        """Debug voice capture import issues."""
        debug_info = []
        
        # Check discord.py version
        import discord
        debug_info.append(f"discord.py version: {discord.__version__}")
        
        # Check if discord.ext exists
        try:
            import discord.ext
            debug_info.append("✅ discord.ext available")
        except ImportError:
            debug_info.append("❌ discord.ext not available")
            
        # Try multiple import patterns
        import_patterns = [
            ("from discord.ext import voice_recv", lambda: __import__('discord.ext.voice_recv', fromlist=['voice_recv'])),
            ("import discord_ext_voice_recv", lambda: __import__('discord_ext_voice_recv')),
            ("direct module import", lambda: __import__('voice_recv')),
        ]
        
        successful_import = None
        for pattern_name, import_func in import_patterns:
            try:
                module = import_func()
                debug_info.append(f"✅ {pattern_name}: SUCCESS")
                debug_info.append(f"  Module location: {getattr(module, '__file__', 'unknown')}")
                if hasattr(module, 'VoiceRecvClient'):
                    debug_info.append(f"  VoiceRecvClient available: ✅")
                else:
                    debug_info.append(f"  VoiceRecvClient available: ❌")
                successful_import = module
                break
            except ImportError as e:
                debug_info.append(f"❌ {pattern_name}: {e}")
            except Exception as e:
                debug_info.append(f"❌ {pattern_name}: Unexpected error: {e}")
                
        if not successful_import:
            debug_info.append("⚠️ No successful import patterns found")
            
        # Check if we can find the package and examine its structure
        try:
            import pkg_resources
            import os
            pkg = pkg_resources.get_distribution("discord-ext-voice-recv")
            debug_info.append(f"📦 Package info: {pkg.project_name} {pkg.version}")
            debug_info.append(f"📦 Package location: {pkg.location}")
            
            # Examine package directory structure
            package_path = pkg.location
            if os.path.exists(package_path):
                debug_info.append(f"📁 Package directory contents:")
                
                # Check for common patterns
                patterns_to_check = [
                    "discord",
                    "discord/ext",
                    "discord_ext_voice_recv", 
                    "voice_recv",
                    "*voice*",
                ]
                
                for pattern in patterns_to_check:
                    if "*" in pattern:
                        # Find files/dirs matching pattern
                        try:
                            import glob
                            matches = glob.glob(os.path.join(package_path, pattern))
                            if matches:
                                debug_info.append(f"  Pattern '{pattern}': {[os.path.basename(m) for m in matches[:3]]}")
                        except:
                            pass
                    else:
                        full_path = os.path.join(package_path, pattern)
                        if os.path.exists(full_path):
                            is_dir = "📁" if os.path.isdir(full_path) else "📄"
                            debug_info.append(f"  {is_dir} {pattern} - EXISTS")
                            
                            # If it's discord/ext, check what's inside
                            if pattern == "discord/ext" and os.path.isdir(full_path):
                                ext_contents = os.listdir(full_path)
                                debug_info.append(f"    Contents: {ext_contents[:5]}")
                        else:
                            debug_info.append(f"  ❌ {pattern} - NOT FOUND")
        except Exception as e:
            debug_info.append(f"📦 Package info unavailable: {e}")
            
        # Check sys.path for relevant paths
        import sys
        voice_paths = [path for path in sys.path if 'discord' in path.lower() or 'voice' in path.lower()]
        if voice_paths:
            debug_info.append(f"🔍 Relevant paths in sys.path:")
            for path in voice_paths[:3]:  # Show first 3 to avoid spam
                debug_info.append(f"  {path}")
            
        await ctx.send("**Voice Capture Debug Info:**\n```\n" + "\n".join(debug_info) + "\n```")

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
            await ctx.send(f"Missing dependencies. These should have been installed automatically.\n"
                         f"Try restarting your bot or use `{ctx.clean_prefix}cog update` and `{ctx.clean_prefix}cog reload realtalk`.\n"
                         f"For detailed setup: `{ctx.clean_prefix}realtalk setup`")
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