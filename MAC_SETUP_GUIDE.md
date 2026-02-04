# macOS Setup Guide - Crypto Market Agents

**Platform:** macOS (Mac)
**User:** animeshchattri@Animeshs-MacBook-Pro

---

## 🍎 Mac-Specific Notes

### Path Differences
- **Linux:** `/home/user/Test`
- **macOS:** `/Users/animeshchattri/Test` (or wherever you cloned it)

### Python Command
- Use `python3` (not `python`)
- macOS comes with Python 3 pre-installed

### Finding Your Project Directory

Run this to find where the project is:
```bash
mdfind -name "run_with_binance.py" 2>/dev/null | head -5
```

Or check common locations:
```bash
# Home directory
ls ~/Test 2>/dev/null && echo "✓ Found in home" || echo "✗ Not here"

# Documents
ls ~/Documents/Test 2>/dev/null && echo "✓ Found in Documents" || echo "✗ Not here"

# Desktop
ls ~/Desktop/Test 2>/dev/null && echo "✓ Found on Desktop" || echo "✗ Not here"
```

---

## 🚀 Mac Commands (Step-by-Step)

### Step 1: Find Your Project
```bash
# Find where the project is
mdfind -name "run_with_binance.py" | head -1
```

**Output will be something like:**
```
/Users/animeshchattri/Test/run_with_binance.py
```

### Step 2: Navigate to Project Directory
```bash
# Replace with your actual path from Step 1
cd ~/Test
# OR
cd ~/Documents/Test
# OR wherever mdfind found it
```

### Step 3: Verify Location
```bash
pwd
ls -la
```

**Should see:**
- `run_with_binance.py`
- `crypto_market_agents/`
- `requirements.txt`
- `.env` or `.env.example`

---

## 📦 Installing Dependencies on Mac

### Option 1: Using pip3 (Recommended)
```bash
pip3 install aiohttp python-dotenv
```

### Option 2: Using pip (if pip3 doesn't work)
```bash
python3 -m pip install aiohttp python-dotenv
```

### Option 3: Using Homebrew Python (if installed)
```bash
brew install python3
python3 -m pip install aiohttp python-dotenv
```

---

## 🔧 Mac-Specific Troubleshooting

### Issue: "command not found: pip3"

**Fix:** Install pip3 or use python3 -m pip
```bash
python3 -m pip install --upgrade pip
python3 -m pip install aiohttp python-dotenv
```

---

### Issue: "Permission denied"

**Fix:** Use `--user` flag
```bash
pip3 install --user aiohttp python-dotenv
```

---

### Issue: "SSL: CERTIFICATE_VERIFY_FAILED"

**Fix:** Install certificates for Python
```bash
# Find your Python installation
python3 --version

# Install certificates (Mac-specific)
open /Applications/Python\ 3.*/Install\ Certificates.command
```

Or:
```bash
pip3 install --upgrade certifi
```

---

### Issue: Multiple Python Versions

**Check which Python you have:**
```bash
which python3
python3 --version
```

**Use specific Python if needed:**
```bash
python3.9 -m pip install aiohttp python-dotenv
python3.9 run_with_binance.py
```

---

## 🎯 Complete Mac Workflow

### 1. Find Project Location
```bash
mdfind -name "run_with_binance.py" | head -1
```

### 2. Navigate There
```bash
cd ~/Test  # Or wherever it was found
```

### 3. Check Python
```bash
python3 --version
```

Should be Python 3.7+

### 4. Install Dependencies
```bash
pip3 install aiohttp python-dotenv
```

### 5. Check .env File
```bash
cat .env
```

If doesn't exist:
```bash
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=8357810702:AAGFWnF7OiTqeJ5KFYJxsDWNXCMFtjj02qs
TELEGRAM_CHAT_ID=1767762282
LOG_LEVEL=INFO
EOF
```

### 6. Test Installation
```bash
python3 test_edge_cases.py
```

### 7. Test Telegram
```bash
python3 test_telegram_integration.py
```

### 8. Run the System
```bash
python3 run_with_binance.py
```

---

## 🛑 Stopping on Mac

Press: **⌘ + C** (Command + C)

Or: **Control + C** (Ctrl + C)

---

## 📱 Mac Terminal App

**Recommended:** Use **iTerm2** or **Terminal.app**

**To open Terminal:**
1. Press `⌘ + Space` (Spotlight)
2. Type "Terminal"
3. Press Enter

---

## 🔐 Mac Security Notes

### If macOS blocks execution:

**"Python" cannot be opened because the developer cannot be verified**

**Fix:**
```bash
xattr -d com.apple.quarantine run_with_binance.py
```

Or go to: **System Preferences > Security & Privacy > Allow**

---

## 📊 Mac Activity Monitor

To monitor the running system:

1. Press `⌘ + Space`
2. Type "Activity Monitor"
3. Look for `Python` process

---

## 🍎 Mac-Specific Environment Variables

Mac uses `~/.zshrc` (if using zsh) or `~/.bash_profile` (if using bash)

**Check your shell:**
```bash
echo $SHELL
```

**To make env vars permanent (optional):**
```bash
# Add to ~/.zshrc or ~/.bash_profile
export TELEGRAM_BOT_TOKEN="8357810702:AAGFWnF7OiTqeJ5KFYJxsDWNXCMFtjj02qs"
export TELEGRAM_CHAT_ID="1767762282"
```

But the `.env` file is easier and more secure!

---

## ✅ Mac Quick Start (Copy-Paste)

```bash
# Find project
PROJECT_DIR=$(mdfind -name "run_with_binance.py" | head -1 | xargs dirname)
cd "$PROJECT_DIR"

# Install dependencies
pip3 install aiohttp python-dotenv

# Run tests
python3 test_edge_cases.py

# Run system
python3 run_with_binance.py
```

---

## 🎯 Next Steps

Once you tell me where your `Test` directory is located, I'll give you the exact commands with the correct path!

**Just run this and tell me the output:**
```bash
mdfind -name "run_with_binance.py" 2>/dev/null
```

This will show me exactly where your project is on your Mac! 🍎
