"""
Run parameter optimization to find best strategy configuration.

This script tests different parameter combinations to find the optimal
configuration for your trading strategy.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from crypto_market_agents.config import SystemConfig
from crypto_market_agents.backtesting import HistoricalDataFetcher, BacktestEngine
from crypto_market_agents.backtesting.optimizer import ParameterOptimizer
from crypto_market_agents.exchange.adapter import ExchangeFactory


async def main():
    """Run parameter optimization."""

    # Load environment variables
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 80)
    print("CRYPTO MARKET AGENTS - PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()

    # Load base configuration
    config = SystemConfig.from_env()

    # Optimization settings
    symbols = [
        'BTC/USDT',
        'ETH/USDT',
        'SOL/USDT',
        'BNB/USDT',
        'XRP/USDT'
    ]

    # Optimization period (60 days)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=60)

    timeframe = '1h'
    initial_capital = 10000.0

    print(f"Optimization Settings:")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Period: {start_date.date()} to {end_date.date()} ({(end_date - start_date).days} days)")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print()

    # Step 1: Fetch historical data once (reused for all backtests)
    print("Step 1: Fetching historical data...")
    print("-" * 80)

    exchange = ExchangeFactory.create(config.exchange)
    await exchange.connect()

    data_fetcher = HistoricalDataFetcher(exchange, cache_dir="backtest_data")

    historical_data = await data_fetcher.fetch_multiple_symbols(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )

    await exchange.disconnect()

    for symbol, candles in historical_data.items():
        print(f"  {symbol}: {len(candles)} candles")
    print()

    # Step 2: Generate parameter grid
    print("Step 2: Generating parameter combinations...")
    print("-" * 80)

    optimizer = ParameterOptimizer(config, output_dir="optimization_results")

    # Define parameter ranges to test
    # (Smaller grid for demonstration - full optimization would test more values)
    parameter_sets = optimizer.generate_parameter_grid(
        min_confidence_values=[0.35, 0.40, 0.45, 0.50],      # Confidence threshold
        max_stop_loss_values=[2.5, 3.0, 4.0],                # Max stop loss %
        reward_risk_values=[1.5, 2.0, 2.5],                  # R:R ratio
        kelly_fraction_values=[0.1, 0.25, 0.5],              # Kelly fraction
        max_portfolio_risk_values=[8.0, 10.0, 12.0],         # Max portfolio risk %
        max_positions_values=[3, 5, 7]                       # Max concurrent positions
    )

    print(f"  Generated {len(parameter_sets)} parameter combinations")
    print(f"  Estimated runtime: {len(parameter_sets) * 2 // 60} - {len(parameter_sets) * 3 // 60} minutes")
    print()

    # Step 3: Run optimization
    print("Step 3: Running optimization (this may take a while)...")
    print("-" * 80)
    print()

    results = await optimizer.run_optimization(
        parameter_sets=parameter_sets,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_capital=initial_capital,
        historical_data=historical_data,
        max_parallel=3  # Run 3 backtests in parallel
    )

    # Step 4: Display results
    print()
    print("=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print()

    if results:
        # Top 5 configurations
        print("TOP 5 CONFIGURATIONS")
        print("-" * 80)
        print()

        for i, result in enumerate(results[:5], 1):
            print(f"#{i} - {result.parameters.name}")
            print(f"  Optimization Score:  {result.optimization_score:.2f}/100")
            print(f"  Parameters:")
            print(f"    • Min Confidence:      {result.parameters.min_confidence:.2f}")
            print(f"    • Max Stop Loss:       {result.parameters.max_stop_loss_pct:.1f}%")
            print(f"    • R:R Ratio:           {result.parameters.reward_risk_ratio:.1f}:1")
            print(f"    • Kelly Fraction:      {result.parameters.kelly_fraction:.2f}")
            print(f"    • Max Portfolio Risk:  {result.parameters.max_portfolio_risk_pct:.1f}%")
            print(f"    • Max Positions:       {result.parameters.max_concurrent_positions}")
            print(f"  Performance:")
            print(f"    • Win Rate:            {result.win_rate:.1f}%")
            print(f"    • Profit Factor:       {result.profit_factor:.2f}")
            print(f"    • Total Return:        {result.total_return_pct:+.2f}%")
            print(f"    • Sharpe Ratio:        {result.sharpe_ratio:.2f}")
            print(f"    • Max Drawdown:        {result.max_drawdown_pct:.2f}%")
            print(f"    • Total Trades:        {result.total_trades}")
            print()

        # Best configuration recommendation
        best = results[0]
        print("=" * 80)
        print("RECOMMENDED CONFIGURATION")
        print("=" * 80)
        print()
        print(f"Use configuration: {best.parameters.name}")
        print(f"Optimization Score: {best.optimization_score:.2f}/100")
        print()
        print("To apply this configuration, update your config.py with:")
        print()
        print(f"  config.signal_synthesis.min_confidence = {best.parameters.min_confidence}")
        print(f"  config.signal_synthesis.max_stop_loss_pct = {best.parameters.max_stop_loss_pct}")
        print(f"  config.signal_synthesis.reward_risk_ratio = {best.parameters.reward_risk_ratio}")
        print(f"  config.risk_management.kelly_fraction = {best.parameters.kelly_fraction}")
        print(f"  config.risk_management.max_portfolio_risk_pct = {best.parameters.max_portfolio_risk_pct}")
        print(f"  config.risk_management.max_concurrent_positions = {best.parameters.max_concurrent_positions}")
        print()

        # Performance expectations
        print("Expected Performance:")
        print(f"  • {best.total_trades} trades over 60 days")
        print(f"  • {best.win_rate:.1f}% win rate")
        print(f"  • {best.total_return_pct:+.2f}% total return")
        print(f"  • {best.max_drawdown_pct:.2f}% maximum drawdown")
        print()

        # Warnings
        if best.total_trades < 20:
            print("⚠️  WARNING: Low sample size (<20 trades). Results may not be statistically significant.")
            print("   Consider running optimization on longer time period.")
            print()

        if best.max_drawdown_pct > 15:
            print("⚠️  WARNING: High maximum drawdown (>15%). Strategy may be too risky.")
            print("   Consider more conservative parameters.")
            print()

    else:
        print("❌ No valid results found. All configurations failed minimum requirements.")
        print("   Try adjusting parameter ranges or check data quality.")

    print()
    print("=" * 80)
    print(f"✅ Optimization complete! Results saved to optimization_results/")
    print("   - Full results JSON")
    print("   - Human-readable summary report")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
