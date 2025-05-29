import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.cached_data_processor import CachedDataProcessor

def demonstrate_caching():
    """
    Demonstrate data caching functionality with before/after download time comparison
    """
    # Initialize the cached data processor
    data_processor = CachedDataProcessor(cache_dir='data_cache')
    
    # Define parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '2020-01-01'
    end_date = '2021-07-20'
    
    print("=== Data Caching Demonstration ===")
    print(f"Symbol: {symbol}, Interval: {interval}")
    print(f"Date Range: {start_date} to {end_date}")
    
    # First run - should download and cache
    print("\n1. First run (downloading data):")
    # Clear the cache directory for demonstration purposes
    cache_file = f"data_cache/{symbol}_{interval}_{start_date}_to_{end_date}.csv"
    processed_cache_file = f"data_cache/{symbol}_{start_date}_to_{end_date}_processed.csv"
    
    if os.path.exists(cache_file):
        print(f"Removing existing cache file for demonstration: {cache_file}")
        os.remove(cache_file)
    
    if os.path.exists(processed_cache_file):
        print(f"Removing existing processed cache file for demonstration: {processed_cache_file}")
        os.remove(processed_cache_file)
    
    # Time the download and processing
    import time
    start_time = time.time()
    
    # Get data (will download from Binance)
    data = data_processor.get_data(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date,
        use_cache=True,
        use_processed_cache=True,
        save_processed=True
    )
    
    first_run_time = time.time() - start_time
    print(f"First run took {first_run_time:.2f} seconds")
    print(f"Data shape: {data.shape}")
    
    # Second run - should load from cache
    print("\n2. Second run (loading from cache):")
    start_time = time.time()
    
    # Get data again (should load from cache)
    cached_data = data_processor.get_data(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date,
        use_cache=True,
        use_processed_cache=True,
        save_processed=True
    )
    
    second_run_time = time.time() - start_time
    print(f"Second run took {second_run_time:.2f} seconds")
    print(f"Data shape: {cached_data.shape}")
    
    # Show the speedup
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"\nSpeedup from caching: {speedup:.1f}x faster")
    
    # Plot some data to verify
    plt.figure(figsize=(12, 6))
    if 'close_orig' in data.columns:
        plt.plot(data.index, data['close_orig'], label='Bitcoin Price')
    else:
        plt.plot(data.index, data['close'], label='Bitcoin Price')
    plt.title(f'{symbol} Price from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'data_cache/{symbol}_price_chart.png')
    print(f"Saved price chart to data_cache/{symbol}_price_chart.png")

if __name__ == "__main__":
    demonstrate_caching() 