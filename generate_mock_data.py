#!/usr/bin/env python3
"""
Generate mock historical data for testing the trading system
This creates realistic-looking OHLC data when Kraken API is unavailable
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def generate_ohlc_data(
    start_date: datetime,
    end_date: datetime,
    timeframe_minutes: int,
    initial_price: float = 45000.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Generate realistic-looking OHLC data"""

    # Calculate number of candles
    total_minutes = int((end_date - start_date).total_seconds() / 60)
    num_candles = total_minutes // timeframe_minutes

    # Generate timestamps
    timestamps = [start_date + timedelta(minutes=i * timeframe_minutes) for i in range(num_candles)]

    # Generate realistic price movement (random walk with drift)
    np.random.seed(42)  # Reproducible results

    closes = [initial_price]
    for _ in range(num_candles - 1):
        change = np.random.normal(0.0001, volatility)  # Small drift, realistic volatility
        new_price = closes[-1] * (1 + change)
        closes.append(new_price)

    # Generate OHLC from closes
    data = []
    for i, timestamp in enumerate(timestamps):
        close = closes[i]
        # High and low within reasonable range of close
        high = close * (1 + abs(np.random.normal(0, volatility/2)))
        low = close * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = np.random.uniform(low, high)

        # Ensure OHLC logic is valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Volume (realistic trading volume)
        volume = np.random.uniform(100, 1000) * (timeframe_minutes / 5)

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })

    return pd.DataFrame(data)

def main():
    """Generate mock data for all timeframes"""

    # Configuration
    instrument = "BTCUSDT.P"
    timeframes = {
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30
    }

    # 6 months of data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    print("=" * 60)
    print("GENERATING MOCK HISTORICAL DATA")
    print("=" * 60)
    print(f"\nInstrument: {instrument}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Timeframes: {', '.join(timeframes.keys())}")
    print()

    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data for each timeframe
    for tf_name, tf_minutes in timeframes.items():
        print(f"Generating {tf_name} data...", end=" ")

        df = generate_ohlc_data(
            start_date=start_date,
            end_date=end_date,
            timeframe_minutes=tf_minutes,
            initial_price=45000.0,
            volatility=0.02
        )

        # Save to CSV
        filename = f"{instrument}_{tf_name}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)

        print(f"✓ {len(df)} candles → {filepath}")

    # Generate funding rates
    print("\nGenerating funding rates...", end=" ")
    funding_dir = Path("data/funding_rates")
    funding_dir.mkdir(parents=True, exist_ok=True)

    # Funding rates every 8 hours
    funding_timestamps = []
    current = start_date
    while current <= end_date:
        funding_timestamps.append(current)
        current += timedelta(hours=8)

    funding_data = pd.DataFrame({
        'timestamp': funding_timestamps,
        'funding_rate': np.random.normal(0.0001, 0.0001, len(funding_timestamps))  # Realistic funding rates
    })

    funding_filepath = funding_dir / f"{instrument}_funding.csv"
    funding_data.to_csv(funding_filepath, index=False)
    print(f"✓ {len(funding_data)} records → {funding_filepath}")

    print("\n" + "=" * 60)
    print("✓ MOCK DATA GENERATION COMPLETE")
    print("=" * 60)
    print("\nYou can now run backtesting:")
    print("  ./run.sh --mode backtest --no-fetch")
    print()

if __name__ == "__main__":
    main()
