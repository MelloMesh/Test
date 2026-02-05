# Quick Start Guide for macOS

## ⚠️ Important: Path Issue Explained

If you tried to run:
```bash
cd /home/user/Test
```

And got the error:
```
cd: no such file or directory: /home/user/Test
```

**This is normal!** The path `/home/user/Test` is from the development environment. On your Mac, the project is in a different location.

## 🚀 Step-by-Step Setup (5 minutes)

### Option A: If you already have the repository

1. **Find where you cloned it:**
   ```bash
   # Search for the project
   find ~ -name "run_backtest.py" -type f 2>/dev/null
   ```

2. **Navigate to that directory:**
   ```bash
   cd /path/shown/above
   # For example: cd ~/Documents/crypto-trading
   ```

3. **Run the automated setup:**
   ```bash
   ./setup_macos.sh
   ```

   This script will:
   - Check Python installation
   - Create virtual environment
   - Install all dependencies
   - Set up configuration files
   - Verify everything works

4. **Run your first backtest:**
   ```bash
   source venv/bin/activate
   python3 run_backtest.py
   ```

### Option B: Clone the repository fresh

1. **Clone to your home directory:**
   ```bash
   cd ~
   git clone https://github.com/MelloMesh/Test.git crypto-trading
   cd crypto-trading
   ```

2. **Run the automated setup:**
   ```bash
   ./setup_macos.sh
   ```

3. **Run your first backtest:**
   ```bash
   source venv/bin/activate
   python3 run_backtest.py
   ```

## 📊 What You Can Run

### 1. Backtesting (Validate Strategy on Historical Data)
Tests your strategy on past 60 days of data:
```bash
python3 run_backtest.py
```

Output shows:
- Total trades
- Win rate
- Profit factor
- Sharpe ratio
- Max drawdown

### 2. Parameter Optimization (Find Best Settings)
Tests different parameter combinations to find optimal configuration:
```bash
python3 run_optimization.py
```

This runs ~216 backtests with different settings and ranks them.

### 3. Live Trading (Paper Trading by Default)
Runs the system in real-time with paper trading:
```bash
python3 main.py
```

## ⚙️ Configuration (Optional)

### For Backtesting
**You don't need API keys!** Backtesting works with public market data.

### For Live Trading
Edit `.env` file:
```bash
open .env  # Opens in TextEdit
```

Required for live trading:
- `TELEGRAM_BOT_TOKEN` - Get from @BotFather (optional but recommended)
- `TELEGRAM_CHAT_ID` - Get from @userinfobot (optional but recommended)

Optional (public data works fine):
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret

## 🔍 Troubleshooting

### "command not found: python"
On macOS, use `python3` instead:
```bash
python3 --version  # Should show Python 3.x
```

### "No module named 'crypto_market_agents'"
Make sure you're in the project directory and virtual environment is activated:
```bash
pwd  # Should show the project directory
source venv/bin/activate  # Activate venv
pip install -r requirements.txt  # Reinstall if needed
```

### "No module named 'aiohttp'"
Virtual environment not activated:
```bash
source venv/bin/activate
```

### "Permission denied: ./setup_macos.sh"
Make it executable:
```bash
chmod +x setup_macos.sh
./setup_macos.sh
```

## 📈 Expected First Run

When you run `python3 run_backtest.py`, you'll see:

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
--------------------------------------------------------------------------------
  BTC/USDT: 1440 candles
  ETH/USDT: 1440 candles
  ...

Step 2: Running backtest simulation...
--------------------------------------------------------------------------------
Processing 1440 time steps...
[Progress updates...]

Step 3: Analysis Results
================================================================================

📊 PERFORMANCE SUMMARY
================================================================================
  Total Trades:      42
  Win Rate:          54.8%
  Profit Factor:     1.82
  Total Return:      +8.45%
  Sharpe Ratio:      1.23
  Max Drawdown:      -4.56%
  ...
```

## 🎯 What's Next?

1. **Review backtest results** in `backtest_results/` directory
2. **Run optimization** to find better parameters
3. **Read CODE_REVIEW.md** for production recommendations
4. **Set up Telegram notifications** for live trading
5. **Start paper trading** to test in real-time

## 📚 More Documentation

- `README.md` - Full documentation
- `MACOS_SETUP.md` - Detailed macOS setup guide
- `CODE_REVIEW.md` - Senior engineer review and recommendations
- `.env.example` - Configuration options

## 💡 Pro Tips

1. **First time?** Just run the backtest - no configuration needed
2. **Want notifications?** Set up Telegram (5 minutes, totally worth it)
3. **Optimizing parameters?** Run on weekends (takes 15-30 minutes)
4. **Going live?** Start with paper trading to build confidence

## ❓ Still Stuck?

Run these diagnostic commands and check the output:
```bash
pwd                    # Where am I?
ls -la                 # What files are here?
python3 --version     # Python installed?
which python3         # Where is Python?
source venv/bin/activate && python3 -c "import crypto_market_agents"  # Can import?
```

If all else fails, try the nuclear option (fresh start):
```bash
rm -rf venv
./setup_macos.sh
```
