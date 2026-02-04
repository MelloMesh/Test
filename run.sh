#!/bin/bash
# Convenience script to run the trading system with python3

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Crypto Perpetual Futures Trading System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check Python version
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Run the main system
echo -e "\n${GREEN}Running trading system...${NC}\n"
python3 main.py "$@"
