# RealTalk Update Guide

## Current Issues & Solutions

### Problem 1: Old Cog Version
You're running an older version of RealTalk that doesn't have the enhanced 4006 error handling.

**Solution:**
```bash
$cog uninstall realtalk realtalk
$cog install realtalk realtalk 
$load realtalk
```

### Problem 2: Discord.py Version & 4006 Errors
The 4006 "WebSocket closed" errors are a known Discord infrastructure issue affecting older discord.py versions.

**Check your discord.py version:**
```bash
$pipshow discord.py
```

**If you're on discord.py < 2.4.0, you have 3 options:**

#### Option A: Upgrade discord.py (Recommended)
```bash
$pipinstall -U "discord.py>=2.4.0"
$restart
```

#### Option B: Switch to Pycord (Alternative)
```bash
$pipuninstall discord.py
$pipinstall -U py-cord
$restart
```

#### Option C: Workaround for Current Version
If you can't upgrade, try:
1. **Different voice server region**: Join voice channels on non-US servers
2. **Wait and retry**: 4006 errors are often temporary
3. **Bot restart**: `$restart` sometimes clears the issue

### Problem 3: Voice Capture Dependencies

**For discord.py users:**
```bash
$pipinstall "git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git"
$restart
```

**For Pycord users:**
Pycord includes built-in voice capture - no additional packages needed!

## Complete Fresh Setup

If you want to start fresh with the latest version:

```bash
# 1. Remove old cog
$cog uninstall realtalk realtalk

# 2. Choose your Discord library
# Option A - discord.py (latest)
$pipinstall -U "discord.py>=2.4.0" "PyNaCl>=1.6.0"
$pipinstall "git+https://github.com/imayhaveborkedit/discord-ext-voice-recv.git"

# Option B - Pycord (includes voice capture)
$pipuninstall discord.py
$pipinstall -U py-cord "PyNaCl>=1.6.0"

# 3. Restart and reinstall cog
$restart
$cog install realtalk realtalk
$load realtalk

# 4. Set API key
$set api openai api_key YOUR_KEY_HERE

# 5. Test
$realtalk setup  # Shows setup instructions
$realtalk join   # Try connecting to voice
```

## Troubleshooting 4006 Errors

The 4006 errors are primarily a Discord infrastructure issue. Here's what helps:

1. **Library version**: Ensure discord.py >= 2.4.0 or use Pycord
2. **Voice server region**: Try different voice channels/servers
3. **PyNaCl version**: Must be >= 1.6.0 for voice protocol v8
4. **Timing**: Sometimes waiting 5-10 minutes helps
5. **Bot restart**: Fresh connection state

## Current Status Check

After updating, verify everything works:
```bash
$realtalk setup      # Shows installation guide
$realtalk status     # Shows current session status  
$realtalk sinks status  # Shows voice capture status
```