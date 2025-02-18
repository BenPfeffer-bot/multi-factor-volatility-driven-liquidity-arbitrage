# Multi-Factor Volatility-Driven Liquidity Arbitrage (MVLA) Strategy

This directory contains the implementation of the MVLA strategy, which leverages multiple factors to identify and exploit volatility-driven arbitrage opportunities.

## Directory Structure

```
strategies/
├── mvla_strategy.py           # Main strategy class
├── signals/                   # Signal generators
│   ├── volatility_signals.py  # Volatility signal generation
│   ├── liquidity_signals.py   # Liquidity signal generation
│   ├── options_signals.py     # Options signal generation
│   ├── macro_signals.py       # Macro signal generation
│   ├── sentiment_signals.py   # Sentiment signal generation
│   └── signal_combiner.py     # Signal combination logic
└── README.md                  # This file
```

## Components

### Main Strategy (`mvla_strategy.py`)
- Orchestrates the overall strategy execution
- Manages data loading and validation
- Coordinates signal generation and combination
- Handles position sizing and risk management

### Signal Generators

#### Volatility Signals (`volatility_signals.py`)
- Calculates multi-timeframe realized volatility
- Detects volatility clusters using GARCH models
- Analyzes IV-RV spreads and term structure
- Generates volatility arbitrage signals

#### Liquidity Signals (`liquidity_signals.py`)
- Calculates Kyle's Lambda for price impact
- Analyzes order flow toxicity and imbalances
- Monitors liquidity regimes and transitions
- Generates liquidity arbitrage signals

#### Options Signals (`options_signals.py`)
- Analyzes volatility surface dynamics
- Monitors skew and term structure
- Calculates volatility risk premium
- Generates options arbitrage signals

#### Macro Signals (`macro_signals.py`)
- Analyzes yield curve dynamics
- Monitors monetary policy impact
- Tracks macro volatility regimes
- Generates macro-driven signals

#### Sentiment Signals (`sentiment_signals.py`)
- Analyzes multi-timeframe sentiment momentum
- Monitors sentiment-price correlations
- Detects sentiment regime shifts
- Generates sentiment-driven signals

### Signal Combination (`signal_combiner.py`)
- Combines signals with dynamic weights
- Calculates signal strength and conviction
- Manages signal conflicts and confirmation
- Provides final trading signals

## Usage

```python
from src.strategies.mvla_strategy import MVLAStrategy

# Initialize strategy
strategy = MVLAStrategy(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    lookback_days=20,
    vol_window=60,
    min_liquidity_threshold=100000,
    sentiment_impact_threshold=0.5,
    iv_rank_threshold=0.7,
    vol_signal_threshold=2.0,
    liquidity_signal_threshold=2.5,
    macro_impact_threshold=1.5
)

# Load data
strategy.load_data(start_date='2024-01-01', end_date='2024-02-17')

# Generate signals
signals = strategy.generate_signals()
```

## Signal Weights

Default signal weights are:
- Volatility: 1.5
- Liquidity: 1.2
- Options: 1.0
- Macro: 0.8
- Sentiment: 0.5

These weights can be customized during strategy initialization. 