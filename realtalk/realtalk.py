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
from .audio import PCMQueueAudioSource, load_opus_lib

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
            "realtime_model": "gpt-4o-realtime-preview-2024-10-01",
            "max_retry_attempts": 5,
            "retry_base_delay": 2.0,
            "voice_timeout": 30.0,
            "audio_threshold": 0.001,
            "silence_threshold": 300,  # local silence hangover (ms) before user stops speaking
            # New configurable options
            "voice": "Alloy",
            "system_prompt": "You are a helpful AI assistant having a voice conversation. Be conversational, concise, and natural. Respond as if you're talking to a friend.",
            "server_vad_threshold": 0.5,
            "server_vad_silence_ms": 300,
            "transcribe_model": "gpt-4o-mini-transcribe",
            "noise_mode": "near",  # none|near|far
            "mix_multiple": True,
            "idle_timeout": 60,
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

        # Model in use
        try:
            status_lines.append(f"Model: `{await self.config.realtime_model()}`")
        except Exception:
            pass
        
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
        
        # Voice client state details
        try:
            if voice_client and hasattr(voice_client, 'is_listening'):
                status_lines.append(f"Listening: {'✅' if voice_client.is_listening() else '❌'}")
        except Exception:
            pass
        try:
            if voice_client:
                status_lines.append(f"Playing: {'✅' if voice_client.is_playing() else '❌'}")
        except Exception:
            pass
            
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
        
        # Opus loaded?
        try:
            status_lines.append(f"Opus loaded: {'✅' if discord.opus.is_loaded() else '❌'}")
        except Exception:
            status_lines.append("Opus loaded: ❓")
            
        try:
            # Use the working import method
            voice_recv_module = __import__('voice_recv')
            if hasattr(voice_recv_module, 'VoiceRecvClient'):
                status_lines.append("✅ discord-ext-voice-recv: Available")
            else:
                status_lines.append("❌ discord-ext-voice-recv: Missing components")
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
        # Use the same working import method as capture.py
        try:
            # Try the working import method
            voice_recv_module = __import__('voice_recv')
            if hasattr(voice_recv_module, 'VoiceRecvClient'):
                await ctx.send("✅ Voice capture is available (discord-ext-voice-recv installed)")
            else:
                await ctx.send("❌ Voice capture module imported but missing required components")
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

    @realtalk.group(name="set")
    @checks.is_owner()
    async def set_group(self, ctx: red_commands.Context):
        """Configure RealTalk options.

        Use subcommands to adjust specific settings. For example:
        - model: Set realtime model (e.g., gpt-4o-mini-realtime-preview)
        - voice: Set TTS voice (Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse)
        - vad: Configure server VAD (threshold, silence ms)
        - threshold: Set local audio activity threshold (0.0001–1.0)
        - noise: Set local noise reduction (none, near, far)
        - mix: Enable/disable multi-speaker mixing
        - transcribe: Set input transcription model (gpt-4o-mini-transcribe or none)
        """
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @set_group.command(name="model")
    async def set_model(self, ctx: red_commands.Context, *, model: str):
        """Set the realtime model (e.g., gpt-4o-mini-realtime-preview)."""
        await self.config.realtime_model.set(model)
        await ctx.send(f"Realtime model set to `{model}`. Rejoin voice to apply.")

    @set_group.command(name="voice")
    async def set_voice(self, ctx: red_commands.Context, *, voice: str):
        """Set the TTS voice: Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse."""
        allowed = {v.lower() for v in ["Alloy","Ash","Ballad","Coral","Echo","Sage","Shimmer","Verse"]}
        v = voice.strip().lower()
        if v not in allowed:
            await ctx.send("Invalid voice. Choose from: Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse")
            return
        await self.config.voice.set(voice.title())
        await ctx.send(f"Voice set to `{voice.title()}`. Rejoin voice to apply.")

    @set_group.command(name="prompt")
    async def set_prompt(self, ctx: red_commands.Context, *, prompt: str):
        """Set the system prompt (instructions) sent to the model."""
        text = prompt.strip()
        if not text:
            await ctx.send("Prompt cannot be empty.")
            return
        await self.config.system_prompt.set(text)
        await ctx.send("System prompt updated. Rejoin voice to apply.")

    @realtalk.group(name="show")
    async def show_group(self, ctx: red_commands.Context):
        """Show current RealTalk configuration values."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @show_group.command(name="prompt")
    async def show_prompt(self, ctx: red_commands.Context):
        """Show the current system prompt used for the session."""
        prompt = await self.config.system_prompt()
        if not prompt:
            prompt = "(empty)"
        await ctx.send(f"Current system prompt:\n{box(prompt)}")

    @show_group.command(name="config")
    async def show_config(self, ctx: red_commands.Context):
        """Show all RealTalk configuration values set via [p]realtalk set."""
        model = await self.config.realtime_model()
        voice = await self.config.voice()
        prompt = await self.config.system_prompt()
        sv_threshold = await self.config.server_vad_threshold()
        sv_silence = await self.config.server_vad_silence_ms()
        transcribe = await self.config.transcribe_model()
        noise = await self.config.noise_mode()
        mix = await self.config.mix_multiple()
        local_threshold = await self.config.audio_threshold()
        idle = await self.config.idle_timeout()
        local_silence = await self.config.silence_threshold()

        lines = [
            "**RealTalk Settings**",
            f"Model: `{model}`",
            f"Voice: `{voice}`",
            f"Server VAD: threshold={sv_threshold}, silence={sv_silence}ms",
            f"Transcription: `{transcribe}`",
            f"Noise reduction: `{noise}`",
            f"Mix multiple speakers: `{mix}`",
            f"Local audio threshold: `{local_threshold}`",
            f"Local silence hangover: `{local_silence}` ms",
            f"Idle timeout: `{idle}` seconds",
            "",
            "System prompt:",
            box(prompt or "(empty)")
        ]
        await ctx.send("\n".join(lines))

    @set_group.command(name="threshold")
    async def set_threshold(self, ctx: red_commands.Context, value: float):
        """Set local audio activity threshold (0.0001–1.0). Lower = more sensitive."""
        if not (0.0001 <= value <= 1.0):
            await ctx.send("Invalid threshold. Use a number between 0.0001 and 1.0, e.g. 0.001")
            return
        await self.config.audio_threshold.set(value)
        await ctx.send(f"Audio activity threshold set to {value}. Rejoin voice to apply.")

    @set_group.command(name="localsilence")
    async def set_local_silence(self, ctx: red_commands.Context, ms: int):
        """Set local silence hangover (ms) before we mark the user as stopped speaking (50–3000 ms)."""
        if ms < 50 or ms > 3000:
            await ctx.send("Invalid value. Choose 50..3000 ms")
            return
        await self.config.silence_threshold.set(ms)
        await ctx.send(f"Local silence hangover set to {ms} ms. Rejoin voice to apply.")

    @set_group.command(name="vad")
    async def set_vad(self, ctx: red_commands.Context, threshold: float, silence_ms: int = 300):
        """Set server VAD threshold (0–1) and silence duration (ms)."""
        if not (0.0 <= threshold <= 1.0) or silence_ms < 50 or silence_ms > 3000:
            await ctx.send("Invalid values. threshold 0..1, silence 50..3000 ms")
            return
        await self.config.server_vad_threshold.set(threshold)
        await self.config.server_vad_silence_ms.set(silence_ms)
        await ctx.send(f"Server VAD set: threshold={threshold}, silence={silence_ms}ms. Applied on next session.")

    @set_group.command(name="noise")
    async def set_noise(self, ctx: red_commands.Context, mode: str):
        """Set noise reduction: none, near, or far."""
        m = mode.strip().lower()
        if m not in {"none","near","far"}:
            await ctx.send("Invalid mode. Choose from: none, near, far")
            return
        await self.config.noise_mode.set(m)
        await ctx.send(f"Noise reduction set to `{m}`. Rejoin voice to apply.")

    @set_group.command(name="mix")
    async def set_mix(self, ctx: red_commands.Context, enabled: bool):
        """Enable or disable multi-speaker mixing (true/false)."""
        await self.config.mix_multiple.set(bool(enabled))
        await ctx.send(f"Multi-speaker mixing {'enabled' if enabled else 'disabled'}. Rejoin voice to apply.")

    @set_group.command(name="transcribe")
    async def set_transcribe(self, ctx: red_commands.Context, *, model: str):
        """Set input transcription model: gpt-4o-mini-transcribe or none."""
        m = model.strip().lower()
        if m not in {"gpt-4o-mini-transcribe","none"}:
            await ctx.send("Invalid model. Use 'gpt-4o-mini-transcribe' or 'none'")
            return
        await self.config.transcribe_model.set(m)
        await ctx.send(f"Transcription model set to `{m}`. Rejoin voice to apply.")

    @set_group.command(name="idle")
    async def set_idle(self, ctx: red_commands.Context, seconds: int):
        """Set idle timeout (seconds) before auto-leave on no voice activity."""
        if seconds < 10 or seconds > 3600:
            await ctx.send("Invalid value. Choose 10..3600 seconds")
            return
        await self.config.idle_timeout.set(seconds)
        await ctx.send(f"Idle timeout set to {seconds}s. Rejoin voice to apply.")

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
            # Use the working import method
            voice_recv_module = __import__('voice_recv')
            if not hasattr(voice_recv_module, 'VoiceRecvClient'):
                raise ImportError("voice_recv module missing VoiceRecvClient")
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

                # Ensure Opus is available for voice decoding/encoding
                try:
                    load_opus_lib()
                except Exception:
                    pass
                
                # For 4006 errors, we need to ensure any existing voice client is fully disconnected
                # before attempting a new connection
                existing_client = ctx.guild.voice_client
                if existing_client:
                    try:
                        await existing_client.disconnect(force=True)
                        # Give Discord time to clean up the connection
                        await asyncio.sleep(0.5)
                    except Exception as cleanup_error:
                        log.debug(f"Error cleaning up existing voice client: {cleanup_error}")
                
                # Import voice receive extension and use its client to enable capture
                try:
                    voice_recv_module = __import__('voice_recv')
                    recv_cls = getattr(voice_recv_module, 'VoiceRecvClient', None)
                except Exception:
                    recv_cls = None

                # Try to connect with a fresh connection, using VoiceRecvClient if available
                connect_kwargs = {
                    'timeout': await self.config.voice_timeout(),
                    'reconnect': False,
                }

                if recv_cls is not None:
                    voice_client = await channel.connect(cls=recv_cls, **connect_kwargs)
                else:
                    voice_client = await channel.connect(**connect_kwargs)
                
                if voice_client and voice_client.is_connected():
                    log.info(f"Voice connection successful for guild {guild_id}")
                    await self._start_session(ctx, voice_client)
                    return True
                    
            except discord.errors.ConnectionClosed as e:
                log.warning(f"Voice connection closed (attempt {attempt + 1}): {e}")
                if e.code == 4006:
                    # Session no longer valid - this is the common Discord infrastructure issue
                    # Force cleanup any existing connection state
                    existing_client = ctx.guild.voice_client
                    if existing_client:
                        try:
                            await existing_client.disconnect(force=True)
                        except Exception:
                            pass
                    
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        log.info(f"Retrying in {delay}s due to 4006 error (session invalidated)...")
                        await asyncio.sleep(delay)
                        continue
                else:
                    log.error(f"Voice connection error {e.code}: {e}")
                    break
                    
            except Exception as e:
                log.error(f"Voice connection error (attempt {attempt + 1}): {e}")
                # Clean up any partial connection state
                existing_client = ctx.guild.voice_client
                if existing_client:
                    try:
                        await existing_client.disconnect(force=True)
                    except Exception:
                        pass
                        
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
            # Initialize Realtime client with configured model and options
            model_name = await self.config.realtime_model()
            voice_name = await self.config.voice()
            transcribe_model = await self.config.transcribe_model()
            server_vad = {
                "threshold": await self.config.server_vad_threshold(),
                "silence_ms": await self.config.server_vad_silence_ms(),
            }

            realtime_client = RealtimeClient(
                api_key,
                model=model_name,
                voice=voice_name,
                transcribe=transcribe_model,
                server_vad=server_vad,
                instructions=await self.config.system_prompt(),
            )
            
            # Initialize voice capture
            voice_capture = VoiceCapture(
                voice_client=voice_client,
                realtime_client=realtime_client,
                audio_threshold=await self.config.audio_threshold(),
                silence_threshold=await self.config.silence_threshold()
            )
            # Apply noise mode and mixing
            try:
                noise_mode = await self.config.noise_mode()
                if noise_mode == "none":
                    voice_capture.set_noise_gate(False)
                elif noise_mode == "near":
                    voice_capture.set_noise_gate(True)
                    voice_capture.noise_floor_adaptation_rate = 0.002
                elif noise_mode == "far":
                    voice_capture.set_noise_gate(True)
                    voice_capture.noise_floor_adaptation_rate = 0.0005
            except Exception:
                pass
            try:
                mix_enabled = await self.config.mix_multiple()
                voice_capture.set_speaker_mixing(bool(mix_enabled))
            except Exception:
                pass
            
            # Initialize current audio source reference (will be created fresh per response)
            current_audio_source = None
            
            # Store session
            self.sessions[guild_id] = {
                "voice_client": voice_client,
                "realtime_client": realtime_client,
                "voice_capture": voice_capture,
                "current_audio_source": current_audio_source,
                "context": ctx,
                "start_time": time.time()
            }
            
            # Connect to OpenAI Realtime API
            await realtime_client.connect()
            
            # Audio playback completion callback
            def _audio_finished(error):
                nonlocal current_audio_source
                if error:
                    log.error(f"Audio playback error: {error}")
                else:
                    log.debug("Audio stream completed successfully")
                # Clear the current source so new audio can start fresh playback
                current_audio_source = None
                self.sessions[guild_id]["current_audio_source"] = None
            
            # Set up audio output callback: enqueue and start playback with proper timing
            def _handle_audio_output(audio_data: bytes):
                try:
                    nonlocal current_audio_source
                    
                    # Create fresh audio source if needed (new response)
                    if current_audio_source is None:
                        current_audio_source = PCMQueueAudioSource()
                        self.sessions[guild_id]["current_audio_source"] = current_audio_source
                        log.debug("Created fresh audio source for new response")
                    
                    # Always queue audio data first
                    current_audio_source.put_audio(audio_data)
                    
                    # Start playback only when we have a small buffer (3-5 frames = 60-100ms)
                    # This prevents Discord timing compensation while maintaining low latency
                    if (voice_client and not voice_client.is_playing() and 
                        current_audio_source.queue_size >= 3):
                        voice_client.play(current_audio_source, after=_audio_finished)
                        log.debug(f"Started Discord audio playback with {current_audio_source.queue_size} frames buffered")
                            
                except Exception as e:
                    log.error(f"Error in audio output handler: {e}")

            realtime_client.on_audio_output = _handle_audio_output
            
            # Handle user interruptions - immediate audio cutoff
            def _on_speech_started():
                try:
                    nonlocal current_audio_source
                    # Immediately stop Discord playback
                    if voice_client and voice_client.is_playing():
                        voice_client.stop()
                    # Clear audio buffers to prevent continuation
                    if current_audio_source:
                        current_audio_source.interrupt_playback()
                except Exception as e:
                    log.error(f"Error handling speech interruption: {e}")
            
            def _on_speech_stopped():
                # Speech ended - ready for bot response
                pass
            
            # Reset playback state on response boundaries
            def _on_resp_start():
                try:
                    nonlocal current_audio_source
                    # Reset the audio source state to clear any leftover data
                    if current_audio_source:
                        current_audio_source.reset_state()
                        log.debug("Reset audio source state for new response")
                except Exception:
                    pass
            
            def _on_resp_done():
                # Response completed - signal that the stream should end when queue empties
                try:
                    nonlocal current_audio_source
                    if current_audio_source:
                        # Signal that we're done sending audio - stream can end when queue empties
                        current_audio_source.finish_stream()
                        log.debug("Response completed - marked stream for completion")
                except Exception:
                    pass
            
            realtime_client.on_input_audio_buffer_speech_started = _on_speech_started
            realtime_client.on_input_audio_buffer_speech_stopped = _on_speech_stopped
            realtime_client.on_response_started = _on_resp_start
            realtime_client.on_response_done = _on_resp_done
            
            # Start voice capture
            await voice_capture.start()
            
            # Do not start playing until we actually have audio; the callback above will trigger playback

            # Start monitor for idle/no-people auto-leave
            idle_timeout = await self.config.idle_timeout()
            monitor_task = asyncio.create_task(self._monitor_session(guild_id, idle_timeout))
            self.sessions[guild_id]["monitor_task"] = monitor_task
            
            log.info(f"Started AI conversation session for guild {guild_id}")
            
        except Exception as e:
            log.error(f"Failed to start session: {e}")
            await self._cleanup_session(guild_id)
            await ctx.send(f"Failed to start AI session: {e}")

    async def _cleanup_session(self, guild_id: int):
        """Clean up session resources."""
        # Atomically remove the session to make cleanup idempotent
        session = self.sessions.pop(guild_id, None)
        if not session:
            return
        
        try:
            # Cancel monitor task
            monitor_task = session.get("monitor_task")
            if monitor_task:
                try:
                    monitor_task.cancel()
                except Exception:
                    pass
            # Stop voice capture
            voice_capture = session.get("voice_capture")
            if voice_capture:
                await voice_capture.stop()
                
            # Disconnect realtime client
            realtime_client = session.get("realtime_client")
            if realtime_client:
                await realtime_client.disconnect()
                
            # Stop current audio source
            current_audio_source = session.get("current_audio_source")
            if current_audio_source:
                current_audio_source.stop()
                
            # Disconnect voice client
            voice_client = session.get("voice_client")
            if voice_client and voice_client.is_connected():
                await voice_client.disconnect()
                
        except Exception as e:
            log.error(f"Error during session cleanup: {e}")
        finally:
            # Clean up retry state
            if guild_id in self.retry_state:
                del self.retry_state[guild_id]
                
        log.info(f"Cleaned up session for guild {guild_id}")

    @red_commands.Cog.listener()
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """Handle voice state updates for automatic cleanup."""
        guild_id = member.guild.id
        session = self.sessions.get(guild_id)
        if not session:
            return
        voice_client: discord.VoiceClient = session.get("voice_client")
        if not voice_client:
            return

        # If the bot disconnected, cleanup
        if member.id == self.bot.user.id:
            if before.channel and not after.channel:
                log.info(f"Bot disconnected from voice in guild {guild_id}, cleaning up session")
                await self._cleanup_session(guild_id)
            return

        # If any user leaves/swaps and bot is alone now, leave
        if voice_client.channel:
            others = [m for m in voice_client.channel.members if m.id != self.bot.user.id]
            if len(others) == 0:
                log.info(f"No other members in voice; leaving guild {guild_id}")
                await self._cleanup_session(guild_id)

    async def _monitor_session(self, guild_id: int, idle_timeout: int):
        """Monitor session for idle/no-people leave conditions."""
        while guild_id in self.sessions:
            try:
                session = self.sessions.get(guild_id)
                if not session:
                    break
                voice_client: discord.VoiceClient = session.get("voice_client")
                capture: VoiceCapture = session.get("voice_capture")
                if not voice_client or not capture:
                    break

                # Leave if alone in channel
                if voice_client.channel:
                    others = [m for m in voice_client.channel.members if m.id != self.bot.user.id]
                    if len(others) == 0:
                        log.info(f"Auto-leave: alone in channel for guild {guild_id}")
                        await self._cleanup_session(guild_id)
                        break

                # Leave if idle beyond threshold (no incoming voice captured)
                last_any = 0.0
                if capture.last_audio_time:
                    try:
                        last_any = max(capture.last_audio_time.values())
                    except Exception:
                        last_any = 0.0
                if last_any and (time.time() - last_any) > idle_timeout:
                    log.info(f"Auto-leave: idle for {idle_timeout}s in guild {guild_id}")
                    await self._cleanup_session(guild_id)
                    break

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(10)
