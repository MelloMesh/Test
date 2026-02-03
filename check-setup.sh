#!/bin/bash
# Quick setup verification script

echo "========================================="
echo "Trading System Setup Verification"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Not in the Test directory"
    echo "💡 Solution: Run 'cd ~/Test' first"
    exit 1
fi

echo "✅ Location: Correct directory"

# Check Python 3
if command -v python3 &> /dev/null; then
    echo "✅ Python 3: $(python3 --version)"
else
    echo "❌ Python 3: Not found"
    exit 1
fi

# Check if main.py exists
if [ -f "main.py" ]; then
    echo "✅ main.py: Found"
else
    echo "❌ main.py: Not found"
    exit 1
fi

# Check if run.sh exists and is executable
if [ -x "run.sh" ]; then
    echo "✅ run.sh: Found and executable"
else
    echo "❌ run.sh: Not executable"
    chmod +x run.sh
    echo "✅ Fixed: Made run.sh executable"
fi

# Check config.py
if [ -f "config.py" ]; then
    echo "✅ config.py: Found"
else
    echo "❌ config.py: Not found"
    exit 1
fi

# Check src directory
if [ -d "src" ]; then
    echo "✅ src/ directory: Found"
else
    echo "❌ src/ directory: Not found"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ Setup verified! You're ready to run:"
echo ""
echo "  ./run.sh --mode discovery"
echo "  ./run.sh --mode full --quick"
echo ""
echo "Or directly:"
echo "  python3 main.py --mode discovery"
echo "========================================="
