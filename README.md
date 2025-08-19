# RealTalk - OpenAI Realtime Voice Assistant for Red-DiscordBot

A Red-DiscordBot cog that enables real-time voice conversations with OpenAI's Realtime API in Discord voice channels.

## Features

🎤 **Two-way voice conversation** - Bot can hear users and respond naturally  
🔄 **Automatic error recovery** - Handles Discord voice infrastructure issues (4006 errors)  
🎯 **Smart audio processing** - Intelligent speaker detection and noise filtering  
⚡ **Low latency** - Real-time audio streaming with minimal delay  
🗣️ **Conversation interruption** - Users can interrupt bot mid-response  
📊 **Comprehensive monitoring** - Detailed status reporting and diagnostics  

## Quick Start

1. **Install dependencies** (see [INSTALLATION.md](INSTALLATION.md) for details):
   ```
   [p]load pip
   [p]pip install "PyNaCl>=1.6.0"
   [p]pip install "git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git"
   ```

2. **Set OpenAI API key**:
   ```
   [p]set api openai api_key YOUR_API_KEY_HERE
   ```

3. **Load the cog**:
   ```
   [p]load realtalk
   ```

4. **Join voice and start chatting**:
   ```
   [p]realtalk join
   ```

## Commands

- `[p]realtalk join` - Join voice channel and start AI conversation
- `[p]realtalk leave` - Leave voice channel and stop session  
- `[p]realtalk status` - Show connection and session status
- `[p]realtalk sinks status` - Check voice capture availability

## Requirements

- Red-DiscordBot v3.5.0+
- Python 3.8+
- OpenAI API key with Realtime API access
- PyNaCl >= 1.6.0 (for Discord voice protocol v8)
- For audio capture, choose one:
  - discord-ext-voice-recv: `pip install "git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git"`
  - OR Pycord (includes built-in sinks): `pip install -U py-cord`

## Technical Details

### Discord Voice Issues Fixed
- **WebSocket 4006 errors**: Implements exponential backoff and session recovery
- **Connection stability**: Enhanced retry logic with progressive timeouts
- **Audio pipeline**: Robust PCM processing with stereo-to-mono conversion

### OpenAI Realtime Integration
- **Optimized settings**: Configured for voice chat with appropriate VAD thresholds
- **Audio streaming**: Efficient PCM16 audio processing at 48kHz
- **Session management**: Automatic reconnection and error handling

### Audio Processing
- **Voice Activity Detection**: Dynamic thresholds for responsive conversation
- **Speaker Selection**: Intelligent audio mixing from multiple users
- **Noise Filtering**: Automatic noise gate to reduce background audio

## Troubleshooting

See [INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting guide.

Common issues:
- **4006 errors**: Discord infrastructure issue - bot will auto-retry
- **No audio capture**: Install voice dependencies and restart bot
- **Connection timeouts**: Check internet connection and try different voice channels

## Contributing

This cog handles Discord's evolving voice infrastructure and OpenAI's Realtime API. Contributions welcome for:
- Additional audio formats/codecs
- Enhanced speaker recognition  
- Performance optimizations
- Bug fixes and stability improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.