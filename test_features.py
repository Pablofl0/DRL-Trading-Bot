import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_processor import DataProcessor

def test_feature_generation():
    """Test if the data processor correctly generates the four required features"""
    # Create a larger synthetic price dataset
    # We need at least 30 data points for ATR (14) and CMF (20) calculations
    n_samples = 50
    
    # Generate synthetic price data with some trend and volatility
    np.random.seed(42)
    
    # Start with a base price
    base_price = 100
    
    # Generate price movements with some randomness
    price_changes = np.random.normal(0.001, 0.02, n_samples)
    
    # Create cumulative returns
    cumulative_returns = np.cumprod(1 + price_changes)
    
    # Generate prices
    close_prices = base_price * cumulative_returns
    
    # Generate other OHLC data
    sample_data = {
        'close': close_prices,
        'open': close_prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': close_prices * (1 + abs(np.random.normal(0, 0.01, n_samples))),
        'low': close_prices * (1 - abs(np.random.normal(0, 0.01, n_samples))),
        'volume': np.random.normal(10000, 2000, n_samples)
    }
    
    data_cache_file = f"data_cache\BTCUSDT_1h_2020-01-01_to_2021-07-20.csv"
    # Create dataframe
    df = pd.read_csv(data_cache_file)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Process the data
    processed_df = processor.prepare_data(df)
    
    # Print the processed dataframe columns
    print("Processed DataFrame columns:", processed_df.columns.tolist())
    
    # Check if the required features are present
    required_features = ['close_diff', 'rsi', 'atr', 'cmf', 'close_orig']
    missing_features = [col for col in required_features if col not in processed_df.columns]
    
    if missing_features:
        print(f"WARNING: Missing required features: {missing_features}")
    else:
        print("SUCCESS: All required features are present")
        
    # Print sample of the processed data
    print("\nSample of processed data:")
    print(processed_df)
    
    # Plot the features
    plt.figure(figsize=(12, 12))
    
    # Plot original close price
    plt.subplot(5, 1, 1)
    plt.plot(processed_df['close_orig'], label='Original Close')
    plt.title('Original Close Price')
    plt.legend()
    
    # Plot differenced closing price
    plt.subplot(5, 1, 2)
    plt.plot(processed_df['close_diff'], label='Differenced Close')
    plt.title('Differenced Close Price')
    plt.legend()
    
    # Plot RSI
    plt.subplot(5, 1, 3)
    plt.plot(processed_df['rsi'], label='RSI', color='orange')
    plt.title('RSI')
    plt.legend()
    
    # Plot normalized ATR
    plt.subplot(5, 1, 4)
    plt.plot(processed_df['atr'], label='Normalized ATR', color='green')
    plt.title('Normalized ATR')
    plt.legend()
    
    # Plot CMF
    plt.subplot(5, 1, 5)
    plt.plot(processed_df['cmf'], label='CMF', color='purple')
    plt.title('Chaikin Money Flow')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_test.png')
    print("\nFeature visualization saved as 'feature_test.png'")

    
    
if __name__ == "__main__":
    test_feature_generation() 