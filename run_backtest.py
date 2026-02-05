"""
Run backtests on historical data to validate and optimize trading strategy.

This script allows you to test the trading system on past data to:
1. Validate strategy performance
2. Optimize parameters
3. Identify issues before live trading
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from crypto_market_agents.config import SystemConfig
from crypto_market_agents.backtesting import BacktestEngine, HistoricalDataFetcher
from crypto_market_agents.exchange.adapter import ExchangeFactory


async def main():
    """Run backtest."""

    # Load environment variables
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("="  * 80)
    print("CRYPTO MARKET AGENTS - BACKTESTING ENGINE")
    print("=" * 80)
    print()

    # Load configuration
    config = SystemConfig.from_env()

    # Configure backtest parameters
    symbols = [
        'BTC/USDT',
        'ETH/USDT',
        'SOL/USDT',
        'BNB/USDT',
        'XRP/USDT'
    ]

    # Date range (last 60 days)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=60)

    timeframe = '1h'  # 1-hour candles
    initial_capital = 10000.0  # $10,000 starting capital

    print(f"Backtest Configuration:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_date.date()} to {end_date.date()} ({(end_date - start_date).days} days)")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print()

    # Step 1: Download historical data
    print("Step 1: Fetching historical data...")
    print("-" * 80)

    # Create exchange adapter for data fetching
    exchange = ExchangeFactory.create(config.exchange)
    await exchange.connect()

    data_fetcher = HistoricalDataFetcher(exchange, cache_dir="backtest_data")

    # Fetch data for all symbols
    historical_data = await data_fetcher.fetch_multiple_symbols(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=True  # Use cache to avoid re-downloading
    )

    await exchange.disconnect()

    # Show data summary
    for symbol, candles in historical_data.items():
        print(f"  {symbol}: {len(candles)} candles")
    print()

    # Step 2: Run backtest
    print("Step 2: Running backtest simulation...")
    print("-" * 80)

    backtest_engine = BacktestEngine(config, data_dir="backtest_results")

    # Load historical data into backtest engine
    backtest_engine.set_historical_data(historical_data)

    results = await backtest_engine.run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_capital=initial_capital
    )

    # Step 3: Analyze results
    print()
    print("Step 3: Analysis Results")
    print("=" * 80)

    analysis = backtest_engine.analyze_results(results)

    if 'error' not in analysis:
        summary = analysis['summary']
        print(f"\n📊 PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"  Total Trades:      {summary['total_trades']}")
        print(f"  Win Rate:          {summary['win_rate']:.1f}%")
        print(f"  Profit Factor:     {summary['profit_factor']:.2f}")
        print(f"  Total Return:      {summary['total_return']:+.2f}%")
        print(f"  Sharpe Ratio:      {summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:      {summary['max_drawdown']:.2f}%")

        risk = analysis['risk_metrics']
        print(f"\n📈 RISK METRICS")
        print(f"{'='*80}")
        print(f"  Avg Win:           +{risk['avg_win']:.2f}%")
        print(f"  Avg Loss:          {risk['avg_loss']:.2f}%")
        print(f"  Largest Win:       +{risk['largest_win']:.2f}%")
        print(f"  Largest Loss:      {risk['largest_loss']:.2f}%")
        print(f"  Avg R:R Ratio:     {risk['avg_rr']:.2f}:1")

        streaks = analysis['streak_analysis']
        print(f"\n🔥 STREAK ANALYSIS")
        print(f"{'='*80}")
        print(f"  Max Consecutive Wins:   {streaks['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses: {streaks['max_consecutive_losses']}")
        print(f"  Current Streak:         {streaks['current_streak']:+d}")

        # Performance verdict
        print(f"\n🎯 VERDICT")
        print(f"{'='*80}")
        pf = summary['profit_factor']
        wr = summary['win_rate']

        if pf > 2.0 and wr > 50:
            verdict = "🚀 EXCELLENT - Strategy shows strong edge"
        elif pf > 1.5 and wr > 40:
            verdict = "✅ GOOD - Strategy is profitable"
        elif pf > 1.0:
            verdict = "🟡 MARGINAL - Strategy barely profitable"
        else:
            verdict = "❌ POOR - Strategy needs significant improvement"

        print(f"  {verdict}")
        print(f"{'='*80}\n")

        # Save detailed results
        print(f"✅ Detailed results saved to backtest_results/")
        print(f"   - Full trade history")
        print(f"   - Performance metrics")
        print(f"   - Configuration used")

    else:
        print(f"❌ Error: {analysis['error']}")

    print()
    print("=" * 80)
    print("Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
