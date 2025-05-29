# Cryptocurrency Trading Bot with PPO Deep Reinforcement Learning

This project implements a cryptocurrency trading bot influenced by the research paper "Automated Cryptocurrency Trading Bot Implementing DRL" by Aisha Peng, Sau Loong Ang, and Chia Yean Lim. The bot uses Proximal Policy Optimization (PPO) with a CNN-LSTM neural network architecture to learn optimal trading strategies.

## Features

- Automated cryptocurrency trading using deep reinforcement learning
- CNN-LSTM architecture for feature extraction and time series analysis
- Technical indicators including RSI, ATR, and Chaikin Money Flow
- Enhanced data preprocessing with differencing for stationarity and normalization
- PPO agent implementation with actor-critic framework
- Live trading simulation with Binance API integration
- Performance metrics and visualization
- Comprehensive backtesting functionality with detailed reports

## Key Formulations

The bot implements the following key formulations from the research paper:

1. **PPO-CLIP Objective Function**:
   
   L<sup>CLIP</sup>(θ) = Ê<sub>t</sub>[min(r<sub>t</sub>(θ)Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1 − ε, 1 + ε)Â<sub>t</sub>)]

2. **Probability Ratio Calculation**:
   
   r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ old</sub>(a<sub>t</sub>|s<sub>t</sub>)

3. **Trading Mechanism**:
   - Buy calculation: Amount bought = Current net worth / Current crypto closing price
   - Sell calculation: Amount sold = Current crypto amount held × Current crypto closing price
   - Reward function: r_t = (v_t - v_{t-1}) / v_{t-1}

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── cnn_lstm_model.py  # CNN-LSTM neural network architecture
│   │   └── ppo_agent.py       # PPO reinforcement learning agent
│   ├── data/
│   │   └── ...                # Data storage (will be created)
│   ├── env/
│   │   └── crypto_env.py      # Trading environment
│   ├── utils/
│   │   └── data_processor.py  # Data processing utilities
│   ├── train.py               # Training script
│   ├── backtest.py            # Backtesting implementation
│   └── live_trading.py        # Live trading implementation
├── models/                    # Saved model weights (will be created)
├── results/                   # Training results and logs (will be created)
├── train_bot.py               # Training wrapper script
├── backtest_bot.py            # Backtesting wrapper script
├── live_trade.py              # Live trading wrapper script
├── demo_data_processing.py    # Script to demonstrate data preprocessing
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection and Preprocessing

The bot automatically downloads historical data from Binance and applies sophisticated preprocessing steps including:

#### Technical Indicators
- Relative Strength Index (RSI): Measures the magnitude of recent price changes to evaluate overbought or oversold conditions
- Average True Range (ATR): Measures market volatility
- Chaikin Money Flow (CMF): Combines price and volume to indicate buying/selling pressure

#### Making Data Stationary
> "The first stage in any analysis should be to see if there is any indication of a trend or seasonal impacts, and if so, remove them. Therefore, the data fed to the stationary model are a realisation of a stationary process"

The bot implements differencing to make price data stationary:
- First-order differencing is applied to price data to remove trends
- Technical indicators are also differenced where appropriate
- RSI is inherently stationary and doesn't require differencing

#### Normalization]

This ensures that all features contribute equally to model learning and speeds up training.

You can view a demonstration of these preprocessing steps by running:
```bash
python demo_data_processing.py
```
This will generate visualizations of each preprocessing step in the `data_plots/` directory.

### Training the Model

To train the model, use the provided wrapper script:

```bash
python train_bot.py
```

You can specify additional parameters:

```bash
python train_bot.py --symbol ETHUSDT --interval 1h --start-date 2021-01-01 --end-date 2022-01-01 --episodes 200 --initial-balance 5000
```

This will:
1. Download historical data for the specified trading pair
2. Preprocess the data with differencing and normalization
3. Train the PPO agent for the specified number of episodes
4. Save the trained model
5. Evaluate the model on test data
6. Compare against a buy-and-hold strategy

### Backtesting

The bot includes a comprehensive backtesting module to evaluate the trained model's performance on historical data:

```bash
python backtest_bot.py
```

You can customize the backtesting parameters:

```bash
python backtest_bot.py --symbol BTCUSDT --interval 1h --start-date 2021-01-01 --end-date 2022-01-01 --commission 0.001 --initial-balance 10000
```

The backtester will:
1. Load the trained model for the specified symbol
2. Simulate trading on the historical data
3. Generate detailed performance metrics including:
   - Total return compared to buy-and-hold
   - Sharpe ratio and maximum drawdown
   - Trade analysis (win rate, profit factor)
4. Create visualizations such as:
   - Equity curve with trade markers
   - Drawdown analysis
   - Action distribution
   - Trade positions over time
5. Save all results and metrics to the specified output directory

### Live Trading

To run the bot in live trading mode, use the provided wrapper script:

```bash
python live_trade.py
```

You can specify additional parameters:

```bash
python live_trade.py --symbol BTCUSDT --interval 1h --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET --test-mode --max-iterations 24 --interval-seconds 3600
```

By default, the bot runs in test mode (no real trades). To enable real trading:

1. Provide your Binance API key and secret as command-line arguments
2. Remove the `--test-mode` flag

**Warning:** Trading cryptocurrencies involves significant risk. Always start with small amounts and use the test mode first.

## Model Architecture

The trading bot uses a hybrid CNN-LSTM architecture:

1. **CNN Layers**: Extract features from historical price data and technical indicators
2. **LSTM Layers**: Model temporal dependencies in the time series data
3. **PPO Agent**: Actor-critic framework that learns optimal trading policy

Benefits of this architecture:
- CNN extracts relevant patterns from price data
- LSTM captures long-term dependencies
- PPO provides stable training with clipped objective function

## Performance Evaluation

The model's performance is evaluated based on:
- Total return compared to buy-and-hold strategy
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown (maximum loss from peak to trough)
- Win rate (percentage of profitable trades)
- Profit factor (ratio of gross profits to gross losses)
