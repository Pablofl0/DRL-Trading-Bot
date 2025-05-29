# Cryptocurrency Trading Bot with PPO Deep Reinforcement Learning

This project implements a cryptocurrency trading bot based on the research paper "Automated Cryptocurrency Trading Bot Implementing DRL" by Aisha Peng, Sau Loong Ang, and Chia Yean Lim. The bot uses Proximal Policy Optimization (PPO) with a CNN-LSTM neural network architecture to learn optimal trading strategies.

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

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

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
As recommended in the research paper:
> "The first stage in any analysis should be to see if there is any indication of a trend or seasonal impacts, and if so, remove them. Therefore, the data fed to the stationary model are a realisation of a stationary process (Cowpertwait & Metcalfe, 2009)."

The bot implements differencing to make price data stationary:
- First-order differencing is applied to price data to remove trends
- Technical indicators are also differenced where appropriate
- RSI is inherently stationary and doesn't require differencing

#### Normalization
Input data is normalized to the range [-1, 1] as mentioned in the paper:
> "Research has found that input data normalisation prior to training is crucial to obtaining good results and significantly fastening the calculations (Sola & Sevilla, 1997)"

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

## References

Peng, A., Ang, S. L., & Lim, C. Y. (2022). Automated Cryptocurrency Trading Bot Implementing DRL. Pertanika J. Sci. & Technol., 30(4), 2683-2705.

Bishop, C. M. (1995). Neural networks for pattern recognition. Oxford University Press.

Cowpertwait, P. S., & Metcalfe, A. V. (2009). Introductory time series with R. New York: Springer. 

Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.

Sola, J., & Sevilla, J. (1997). Importance of input data normalization for the application of neural networks to complex industrial problems. IEEE Transactions on Nuclear Science, 44(3), 1464-1468.

## License

This project is licensed under the MIT License.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

## Data Caching Feature

A major enhancement to this project is the data caching system that allows you to:

1. Download data from Binance once and reuse it for multiple training sessions
2. Cache both raw data and processed data (with technical indicators, differencing, etc.)
3. Significantly speed up training by avoiding redundant downloads and processing

## Key Files

- `src/utils/cached_data_processor.py`: Enhanced data processor with caching functionality
- `src/train_cached.py`: Modified training function that uses cached data
- `cached_train_bot.py`: Command-line script to train the bot with cached data
- `use_cached_data.py`: Example script demonstrating the data caching functionality

## How to Use

### Basic Data Caching Demo

Run the demonstration script to see how data caching works:

```bash
python use_cached_data.py
```

This will show:
1. The first run downloading data from Binance and saving it to cache
2. The second run loading from cache (much faster)

### Training with Cached Data

```bash
python cached_train_bot.py --symbol BTCUSDT --interval 1h --start-date 2020-01-01 --end-date 2021-07-20
```

### Command-line Options for Cached Training

```
--symbol BTCUSDT         # Trading pair symbol
--interval 1h            # Timeframe interval (1h, 15m, 1d, etc.)
--start-date 2020-01-01  # Start date for training data
--end-date 2021-07-20    # End date for training data
--episodes 4000          # Number of training episodes
--fast-train             # Use fewer episodes (100) for quick testing
--no-gpu                 # Disable GPU training
--no-cache               # Disable data caching (always download fresh)
--cache-dir data_cache   # Directory for cached data files
```

## Cache Directory Structure

Data is cached in the `data_cache` directory (or custom directory if specified):

- `BTCUSDT_1h_2020-01-01_to_2021-07-20.csv`: Raw data from Binance
- `BTCUSDT_2020-01-01_to_2021-07-20_processed.csv`: Processed data with indicators

## Benefits

- **Speed**: Training sessions after the first one start much faster
- **Bandwidth**: Reduces API calls to Binance
- **Reproducibility**: Ensures the same data is used across training runs
- **Offline Work**: Train the model without internet access using cached data

## Notes

- Cache files include timestamps in their names, so different date ranges are cached separately
- You can clear the cache at any time by deleting files from the cache directory
- Use the `--no-cache` flag if you want to force a fresh download 