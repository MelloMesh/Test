"""
Demo mode - Simulates the crypto market agents system with mock data.

This demonstrates how the system works without requiring actual API access.
"""

import asyncio
import random
from datetime import datetime, timezone
from crypto_market_agents.schemas import (
    PriceActionSignal,
    MomentumSignal,
    VolumeSignal,
    TradingSignal,
    SystemReport,
    AgentStatus
)


def generate_mock_price_signals(count=10):
    """Generate mock price action signals."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
               "DOGEUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT"]

    signals = []
    for i in range(min(count, len(symbols))):
        symbol = symbols[i]
        base_price = random.uniform(0.5, 50000)
        price_change = random.uniform(-8, 8)
        volatility = random.uniform(0.5, 3.5)

        signal = PriceActionSignal(
            symbol=symbol,
            price=base_price,
            price_change_pct=price_change,
            intraday_range_pct=abs(price_change) * 1.5,
            volatility_ratio=volatility,
            breakout_detected=abs(price_change) > 3.0 or volatility > 2.0,
            timeframe="24h",
            timestamp=datetime.now(timezone.utc)
        )
        signals.append(signal)

    return signals


def generate_mock_momentum_signals(count=10):
    """Generate mock momentum signals."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
               "DOGEUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT"]

    signals = []
    for i in range(min(count, len(symbols))):
        symbol = symbols[i]
        rsi = random.uniform(20, 80)
        obv_change = random.uniform(-30, 30)

        if rsi > 70:
            status = "overbought"
            strength = (rsi - 50) * 2
        elif rsi < 30:
            status = "oversold"
            strength = (50 - rsi) * 2
        else:
            status = "neutral"
            strength = abs(rsi - 50) * 2

        signal = MomentumSignal(
            symbol=symbol,
            rsi=rsi,
            obv=random.uniform(1000000, 10000000),
            obv_change_pct=obv_change,
            status=status,
            strength_score=strength,
            timeframe="15m",
            timestamp=datetime.now(timezone.utc)
        )
        signals.append(signal)

    return signals


def generate_mock_volume_signals(count=10):
    """Generate mock volume signals."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
               "DOGEUSDT", "XRPUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT"]

    signals = []
    for i in range(min(count, len(symbols))):
        symbol = symbols[i]
        volume = random.uniform(5000000, 50000000)
        baseline = volume * random.uniform(0.5, 0.9)
        volume_change = ((volume - baseline) / baseline) * 100
        zscore = random.uniform(-1, 4)

        signal = VolumeSignal(
            symbol=symbol,
            volume_24h=volume,
            volume_change_pct=volume_change,
            volume_zscore=zscore,
            spike_detected=zscore > 2.0,
            baseline_volume=baseline,
            liquidity_usd=volume * random.uniform(20, 50000),
            timestamp=datetime.now(timezone.utc)
        )
        signals.append(signal)

    return signals


def synthesize_trading_signal(price_signal, momentum_signal, volume_signal):
    """Synthesize a trading signal from component signals."""

    # Determine direction and confidence
    score = 0
    rationale_parts = []

    # Price analysis
    if price_signal.breakout_detected:
        if price_signal.price_change_pct > 0:
            score += 0.3
            rationale_parts.append(f"Bullish breakout (+{price_signal.price_change_pct:.1f}%)")
        else:
            score -= 0.3
            rationale_parts.append(f"Bearish breakout ({price_signal.price_change_pct:.1f}%)")

    if price_signal.volatility_ratio > 2.0:
        rationale_parts.append(f"High volatility ({price_signal.volatility_ratio:.1f}x)")

    # Momentum analysis
    if momentum_signal.status == "oversold":
        score += 0.3
        rationale_parts.append(f"Oversold RSI {momentum_signal.rsi:.1f}")
    elif momentum_signal.status == "overbought":
        score -= 0.3
        rationale_parts.append(f"Overbought RSI {momentum_signal.rsi:.1f}")

    if momentum_signal.obv_change_pct > 10:
        score += 0.1
        rationale_parts.append(f"OBV rising (+{momentum_signal.obv_change_pct:.1f}%)")
    elif momentum_signal.obv_change_pct < -10:
        score -= 0.1
        rationale_parts.append(f"OBV falling ({momentum_signal.obv_change_pct:.1f}%)")

    # Volume analysis
    if volume_signal.spike_detected:
        score += 0.2
        rationale_parts.append(f"Volume spike (z-score: {volume_signal.volume_zscore:.1f})")

    # Determine direction
    if score > 0.3:
        direction = "LONG"
        confidence = min(1.0, abs(score))
    elif score < -0.3:
        direction = "SHORT"
        confidence = min(1.0, abs(score))
    else:
        return None

    # Calculate levels
    entry = price_signal.price
    stop_pct = min(price_signal.intraday_range_pct / 2, 2.0)
    stop_pct = max(stop_pct, 0.5)

    if direction == "LONG":
        stop = entry * (1 - stop_pct / 100)
        target = entry * (1 + (stop_pct * 2) / 100)
    else:
        stop = entry * (1 + stop_pct / 100)
        target = entry * (1 - (stop_pct * 2) / 100)

    return TradingSignal(
        asset=price_signal.symbol,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        confidence=confidence,
        rationale=" | ".join(rationale_parts),
        timestamp=datetime.now(timezone.utc),
        price_signal=price_signal.to_dict(),
        momentum_signal=momentum_signal.to_dict(),
        volume_signal=volume_signal.to_dict()
    )


async def simulate_agent_cycle():
    """Simulate one cycle of all agents."""

    print("\n" + "="*80)
    print("SIMULATION CYCLE STARTED")
    print("="*80)

    # Simulate Price Action Agent
    print("\n[Price Action Agent] Analyzing price movements...")
    await asyncio.sleep(0.5)
    price_signals = generate_mock_price_signals(10)
    breakouts = [s for s in price_signals if s.breakout_detected]
    print(f"✓ Generated {len(price_signals)} price signals, {len(breakouts)} breakouts detected")

    # Simulate Momentum Agent
    print("\n[Momentum Agent] Computing RSI and OBV...")
    await asyncio.sleep(0.5)
    momentum_signals = generate_mock_momentum_signals(10)
    overbought = [s for s in momentum_signals if s.status == "overbought"]
    oversold = [s for s in momentum_signals if s.status == "oversold"]
    print(f"✓ Generated {len(momentum_signals)} momentum signals")
    print(f"  - Overbought: {len(overbought)}, Oversold: {len(oversold)}")

    # Simulate Volume Spike Agent
    print("\n[Volume Spike Agent] Detecting volume anomalies...")
    await asyncio.sleep(0.5)
    volume_signals = generate_mock_volume_signals(10)
    spikes = [s for s in volume_signals if s.spike_detected]
    print(f"✓ Generated {len(volume_signals)} volume signals, {len(spikes)} spikes detected")

    # Simulate Signal Synthesis
    print("\n[Signal Synthesis Agent] Integrating all signals...")
    await asyncio.sleep(0.5)

    trading_signals = []
    for i in range(len(price_signals)):
        signal = synthesize_trading_signal(
            price_signals[i],
            momentum_signals[i],
            volume_signals[i]
        )
        if signal and signal.confidence >= 0.6:
            trading_signals.append(signal)

    trading_signals.sort(key=lambda x: x.confidence, reverse=True)

    print(f"✓ Generated {len(trading_signals)} actionable trading signals")

    return trading_signals


async def main():
    """Run the demo simulation."""

    print("="*80)
    print("CRYPTO MARKET AGENTS - DEMO MODE")
    print("="*80)
    print("\nThis demo simulates the multi-agent system with mock market data.")
    print("In production, this connects to real exchange APIs.\n")

    # Simulate 3 cycles
    for cycle in range(1, 4):
        print(f"\n{'='*80}")
        print(f"CYCLE {cycle}/3")
        print(f"{'='*80}")

        trading_signals = await simulate_agent_cycle()

        # Display top signals
        print(f"\n{'='*80}")
        print("TOP TRADING SIGNALS")
        print(f"{'='*80}\n")

        for i, signal in enumerate(trading_signals[:5], 1):
            print(f"{i}. {signal.asset} - {signal.direction}")
            print(f"   Entry:      ${signal.entry:,.2f}")
            print(f"   Stop Loss:  ${signal.stop:,.2f}")
            print(f"   Target:     ${signal.target:,.2f}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Risk/Reward: {abs((signal.target - signal.entry) / (signal.entry - signal.stop)):.2f}:1")
            print(f"   Rationale:  {signal.rationale}")
            print()

        if cycle < 3:
            print("\nWaiting 10 seconds before next cycle...")
            await asyncio.sleep(10)

    # Generate final report
    print("\n" + "="*80)
    print("FINAL SYSTEM REPORT")
    print("="*80 + "\n")

    agent_statuses = [
        AgentStatus(
            agent_name="PriceAction",
            status="running",
            last_update=datetime.now(timezone.utc),
            signals_generated=30,
            errors=0
        ),
        AgentStatus(
            agent_name="Momentum",
            status="running",
            last_update=datetime.now(timezone.utc),
            signals_generated=30,
            errors=0
        ),
        AgentStatus(
            agent_name="VolumeSpike",
            status="running",
            last_update=datetime.now(timezone.utc),
            signals_generated=30,
            errors=0
        ),
        AgentStatus(
            agent_name="SignalSynthesis",
            status="running",
            last_update=datetime.now(timezone.utc),
            signals_generated=len(trading_signals),
            errors=0
        )
    ]

    report = SystemReport(
        timestamp=datetime.now(timezone.utc),
        active_agents=4,
        trading_signals=trading_signals,
        agent_statuses=agent_statuses
    )

    print(f"Active Agents: {report.active_agents}/4")
    print(f"Total Trading Signals: {len(trading_signals)}")
    print(f"\nAgent Status:")
    for status in agent_statuses:
        print(f"  • {status.agent_name}: {status.status.upper()} "
              f"({status.signals_generated} signals, {status.errors} errors)")

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nIn production mode, this data would be:")
    print("  • Saved to output/latest_report.json")
    print("  • Updated every 5 minutes")
    print("  • Based on real-time market data from the exchange")
    print("\nTo run with real data:")
    print("  1. Ensure you have network access to exchange APIs")
    print("  2. For US users: Implement a US-compliant exchange adapter")
    print("  3. Run: python -m crypto_market_agents.main")


if __name__ == "__main__":
    asyncio.run(main())
