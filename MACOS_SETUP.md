# macOS Setup Guide - Crypto Market Agents

This guide helps you run the backtesting system on your Mac.

## Step 1: Find or Clone the Project

### Option A: If you already cloned the repository
Open Terminal and search for it:
```bash
# Search your home directory
find ~ -name "run_backtest.py" -type f 2>/dev/null

# Or check common locations
ls ~/Documents/
ls ~/Desktop/
ls ~/Projects/
```

Once found, navigate to that directory:
```bash
cd /path/to/your/project
```

### Option B: Clone the repository fresh
```bash
# Clone to your home directory
cd ~
git clone https://github.com/MelloMesh/Test.git crypto-trading
cd crypto-trading

# Or if you need SSH:
# git clone git@github.com:MelloMesh/Test.git crypto-trading
```

## Step 2: Verify You're in the Right Place
```bash
# You should see these files:
ls run_backtest.py run_optimization.py crypto_market_agents/

# Should show the project files
```

## Step 3: Install Python (if needed)
```bash
# Check if Python 3 is installed
python3 --version

# If not installed, install via Homebrew:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
```

## Step 4: Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv)
```

## Step 5: Install Dependencies
```bash
# Make sure you're in the project directory with requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 6: Set Up Configuration
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your exchange API credentials
nano .env  # or use: open -e .env
```

## Step 7: Run Backtest
```bash
# Make sure virtual environment is activated
# Your prompt should show (venv)

# Run the backtest
python3 run_backtest.py
```

## Quick Start Script
Save this as `start.sh` in the project directory:
```bash
#!/bin/bash
# Activate virtual environment
source venv/bin/activate

# Run backtest
python3 run_backtest.py
```

Make it executable:
```bash
chmod +x start.sh
./start.sh
```

## Troubleshooting

### "No module named 'crypto_market_agents'"
- Make sure you're in the project root directory
- Make sure virtual environment is activated
- Run: `pip install -e .`

### "ModuleNotFoundError: No module named 'ccxt'"
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### "command not found: python"
- On macOS, use `python3` instead of `python`
- All commands should use `python3`

## What You Need Before Running

1. **Exchange API Credentials** (in `.env` file):
   - `EXCHANGE_NAME=binance` (or your exchange)
   - `EXCHANGE_API_KEY=your_key`
   - `EXCHANGE_API_SECRET=your_secret`

2. **Internet Connection** - To fetch historical market data

3. **~15 minutes** - First run downloads historical data

## Expected Output

When successful, you'll see:
```
================================================================================
CRYPTO MARKET AGENTS - BACKTESTING ENGINE
================================================================================

Backtest Configuration:
  Symbols: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT
  Period: 2024-12-06 to 2025-02-04 (60 days)
  Timeframe: 1h
  Initial Capital: $10,000.00

Step 1: Fetching historical data...
...
```

## Need Help?

If you're still stuck, run these commands and share the output:
```bash
pwd                          # Where am I?
ls -la                       # What files are here?
python3 --version           # Is Python installed?
which python3               # Where is Python?
```
