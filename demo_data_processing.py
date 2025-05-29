import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tabulate import tabulate

# Add current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the DataProcessor
from src.utils.data_processor import DataProcessor

def main():
    """
    Demonstrate the data preprocessing techniques from the research paper.
    Shows how input states are created with 100 hours of market information containing:
    - Closing price
    - Relative strength index indicator (RSI)
    - Normalized average true range indicator (ATR)
    - On-balance volume indicator (OBV)
    """
    # Initialize the data processor
    data_processor = DataProcessor()
    
    # Set parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    
    # Calculate dates (last 3 months of data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"Downloading historical data for {symbol} from {start_date} to {end_date}...")
    
    try:
        # Download historical data
        df = data_processor.download_data(symbol, interval, start_date, end_date)
        print(f"Downloaded {len(df)} data points.")
        
        # Save the original data
        original_df = df.copy()
        
        # Step 1: Add technical indicators (as specified in Table 1)
        print("\nStep 1: Adding technical indicators...")
        print("- Relative strength index indicator (RSI)")
        print("- Normalized average true range indicator (ATR)")
        print("- On-balance volume indicator (OBV)")
        
        df_with_indicators = data_processor.add_technical_indicators(df)
        print(f"Data shape after adding indicators: {df_with_indicators.shape}")
        print(f"Added indicators: {set(df_with_indicators.columns) - set(df.columns)}")
        
        # Plot the technical indicators
        print("\nGenerating plots of the technical indicators...")
        data_processor.plot_indicators(df_with_indicators)
        print("Technical indicators plot saved as 'technical_indicators.png'")
        
        # Step 2: Apply differencing
        print("\nStep 2: Applying differencing to make data stationary...")
        df_differenced = data_processor.apply_difference(df_with_indicators)
        print(f"Data shape after differencing: {df_differenced.shape}")
        diff_columns = [col for col in df_differenced.columns if col.endswith('_diff')]
        print(f"Differenced columns: {diff_columns}")
        
        # Step 3: Normalize data
        print("\nStep 3: Normalizing data...")
        df_normalized = data_processor.normalize_data(df_differenced)
        print(f"Data shape after normalization: {df_normalized.shape}")
        
        # Create directory for plots
        os.makedirs('data_plots', exist_ok=True)
        
        # Plot original vs. differenced close price
        print("\nGenerating comparison plots...")
        plt.figure(figsize=(15, 10))
        
        # Original close price
        plt.subplot(3, 1, 1)
        plt.plot(original_df.index, original_df['close'], color='blue')
        plt.title('Original Close Price')
        plt.grid(True, alpha=0.3)
        
        # Differenced close price
        plt.subplot(3, 1, 2)
        plt.plot(df_differenced.index, df_differenced['close_diff'], color='green')
        plt.title('Differenced Close Price (Stationary)')
        plt.grid(True, alpha=0.3)
        
        # Normalized differenced close price
        plt.subplot(3, 1, 3)
        plt.plot(df_normalized.index, df_normalized['close_diff'], color='red')
        plt.title('Normalized Differenced Close Price (Range [-1, 1])')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_plots/close_price_processing.png')
        plt.close()
        
        # Create a table like Table 1 in the paper showing the first 5 rows of input state
        print("\nExample of input states to model at each step (similar to Table 1):")
        print("Each state has 100 hours of market information containing the closing price and three technical indicators")
        
        # Select columns for display (matching Table 1)
        table_cols = ['close', 'rsi', 'norm_atr', 'norm_obv']
        
        # Get a sample from processed data
        sample_data = df_with_indicators[table_cols].head(5)
        sample_data = sample_data.reset_index()
        
        # Rename columns to match Table 1
        sample_data.columns = ['Timestamp', 'Closing price', 'Relative strength index indicator', 
                              'Normalised average true range indicator', 'On-balance volume indicator']
        
        # Convert timestamp to UNIX timestamp like in Table 1
        sample_data['Timestamp'] = sample_data['Timestamp'].apply(lambda x: int(x.timestamp()))
        
        # Print as table
        print(tabulate(sample_data, headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        # Visualize the 100-hour lookback window
        print("\nVisualizing the 100-hour lookback window (state representation)...")
        
        # Create a plot showing the sliding window concept
        lookback_window_size = 100
        if len(df_with_indicators) > lookback_window_size + 10:
            # Get a section of data for visualization
            window_end = lookback_window_size + 10
            window_data = df_with_indicators.iloc[:window_end]
            
            # Plot the closing prices
            plt.figure(figsize=(15, 8))
            plt.plot(window_data.index, window_data['close'], color='blue', label='Closing Price')
            
            # Highlight the lookback window
            window_start_idx = 10  # Start at index 10
            window_end_idx = window_start_idx + lookback_window_size
            
            lookback_data = window_data.iloc[window_start_idx:window_end_idx]
            plt.plot(lookback_data.index, lookback_data['close'], color='red', linewidth=3, 
                    label=f'Current State (Lookback Window: {lookback_window_size} hours)')
            
            # Add vertical lines to show window boundaries
            plt.axvline(x=lookback_data.index[0], color='green', linestyle='--', 
                       label='Lookback Window Start')
            plt.axvline(x=lookback_data.index[-1], color='purple', linestyle='--', 
                       label='Current Timestep')
            
            plt.title(f'Visualization of {lookback_window_size}-hour Lookback Window (State Representation)')
            plt.xlabel('Timestamp')
            plt.ylabel('Closing Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('data_plots/lookback_window_visualization.png')
            plt.close()
            print("Lookback window visualization saved to 'data_plots/lookback_window_visualization.png'")
        
        print("\nData preprocessing demonstration completed.")
        print(f"The processed data contains a lookback window of {lookback_window_size} hours with:")
        print("- Closing price")
        print("- Relative strength index indicator (RSI)")
        print("- Normalized average true range indicator (ATR)")
        print("- On-balance volume indicator (OBV)")
        print("\nThese match the input state structure shown in Table 1 of the research paper.")
        
    except Exception as e:
        print(f"Error during data processing demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 