#!/bin/bash
# Automated setup script for Crypto Market Agents on macOS

set -e  # Exit on error

echo "================================================================================"
echo "  Crypto Market Agents - macOS Setup"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check Python installation
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 is not installed!"
    echo ""
    echo "Please install Python 3 first:"
    echo "  Option 1: brew install python"
    echo "  Option 2: Download from https://www.python.org/downloads/"
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}!${NC} Virtual environment already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Virtual environment recreated"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Step 3: Activate virtual environment and install dependencies
echo ""
echo "Step 3: Installing dependencies..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip --quiet

# Install requirements
pip install -r requirements.txt

echo -e "${GREEN}✓${NC} Dependencies installed:"
pip list | grep -E "(aiohttp|python-dotenv)"

# Step 4: Set up environment file
echo ""
echo "Step 4: Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file from template"
    echo ""
    echo -e "${YELLOW}⚠${NC}  IMPORTANT: Edit .env file with your credentials:"
    echo "   - TELEGRAM_BOT_TOKEN (optional - for notifications)"
    echo "   - TELEGRAM_CHAT_ID (optional - for notifications)"
    echo "   - BINANCE_API_KEY (optional - public data works without keys)"
    echo "   - BINANCE_API_SECRET (optional - public data works without keys)"
    echo ""
    echo "   To edit: open .env    (or use: nano .env)"
else
    echo -e "${GREEN}✓${NC} .env file already exists"
fi

# Step 5: Create data directories
echo ""
echo "Step 5: Creating data directories..."
mkdir -p data backtest_data backtest_results optimization_results output
echo -e "${GREEN}✓${NC} Data directories created"

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python3 << 'EOF'
try:
    import aiohttp
    import dotenv
    from crypto_market_agents.config import SystemConfig
    from crypto_market_agents.backtesting import BacktestEngine
    print("\033[0;32m✓\033[0m All imports successful")
except ImportError as e:
    print(f"\033[0;31m✗\033[0m Import error: {e}")
    exit(1)
EOF

# Success message
echo ""
echo "================================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. (Optional) Edit .env file with your credentials:"
echo "     open .env"
echo ""
echo "  3. Run a backtest:"
echo "     python3 run_backtest.py"
echo ""
echo "  4. Or run parameter optimization:"
echo "     python3 run_optimization.py"
echo ""
echo "  5. For live trading (paper trading by default):"
echo "     python3 main.py"
echo ""
echo "================================================================================"
echo ""
echo "Tip: You don't need API keys to run backtests with historical data."
echo "     The system can fetch public market data without authentication."
echo ""
