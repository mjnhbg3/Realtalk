# RealTalk Installation Guide

## Dependencies

This cog requires additional dependencies that need to be installed before loading the cog.

### Method 1: Using Red's PIP Cog (Recommended)

1. Load the pip cog:
   ```
   [p]load pip
   ```

2. Install required dependencies:
   ```
   [p]pip install "PyNaCl>=1.6.0"
   [p]pip install discord-ext-voice-recv
   ```

3. Restart your bot to ensure dependencies are loaded properly.

4. Load the RealTalk cog:
   ```
   [p]load realtalk
   ```

### Method 2: Manual Installation

If you have direct access to your bot's Python environment:

```bash
pip install "PyNaCl>=1.6.0" discord-ext-voice-recv
```

Then restart your bot and load the cog.

## Configuration

1. Set your OpenAI API key (choose one method):
   - **Recommended**: `[p]set api openai api_key YOUR_API_KEY_HERE`
   - **Alternative**: `[p]realtalk set key YOUR_API_KEY_HERE`
   - **Environment**: Set `OPENAI_API_KEY` environment variable

2. Join a voice channel and run:
   ```
   [p]realtalk join
   ```

## Troubleshooting

### Voice Connection Issues (Error 4006)
- This is a known Discord infrastructure issue
- The bot includes automatic retry logic with exponential backoff
- Try different voice channels or wait a few minutes if persistent

### Audio Capture Not Working
- Install voice capture dependencies:
  ```
  [p]pip install discord-ext-voice-recv
  ```
  OR
  ```
  [p]pip install -U git+https://github.com/Ext-Creators/discord-ext-sinks.git
  ```
- Restart your bot after installation

### Permission Issues
- Ensure bot has "Connect" and "Speak" permissions in voice channels
- Ensure bot has "Send Messages" permission in text channels for status updates

## Features

- ✅ Two-way voice conversation with OpenAI Realtime API
- ✅ Automatic error recovery for Discord voice issues
- ✅ Smart speaker detection and audio mixing
- ✅ Conversation interruption (barge-in) support
- ✅ Real-time audio streaming with low latency
- ✅ Comprehensive status monitoring and diagnostics