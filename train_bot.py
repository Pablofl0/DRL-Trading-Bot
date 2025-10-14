#!/usr/bin/env python3
import os
import sys
from datetime import date
import tensorflow as tf

# Asegurar que src esté en path (igual que tu estructura)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importa la función train_agent (tu train.py debe exponerla)
from src.train import train_agent

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train the cryptocurrency trading bot (multi-activo)')
    # Mantengo --symbol para retrocompatibilidad; --assets permite lista multi-activo
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Single trading pair symbol (retrocompatible).')
    parser.add_argument('--assets', nargs='+', type=str, default=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
                        help='List of assets for multi-head training. Ej: --assets BTCUSDT ETHUSDT SOLUSDT (default: None -> uses --symbol only)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='Timeframe interval (default: 1h)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=date.today().strftime('%Y-%m-%d'),
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of training episodes')
    parser.add_argument('--trajectory-size', type=int, default=1000,
                        help='Number of steps in each trajectory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs per update')
    parser.add_argument('--initial-balance', type=float, default=1000,
                        help='Initial balance for trading')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Trading commission rate (default 0.001 -> 0.1%)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Portion of data to reserve for testing (0.0-1.0)')
    parser.add_argument('--lookback-window', type=int, default=100,
                        help='Lookback window size for observations')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save frequency (in episodes)')
    parser.add_argument('--fast-train', action='store_true',
                        help='Use a short run for debugging (100 episodes)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU (use CPU)')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision (if GPU available)')
    parser.add_argument('--resume-from', type=int, default=0,
                        help='Resume training from given episode (default 0)')
    parser.add_argument('--use-lr-schedule', action='store_true',
                        help='Use LR schedules in the agent (if supported)')

    args = parser.parse_args()

    # Decide episodes (fast mode)
    if args.fast_train:
        episodes = 2
        print("⚠️ FAST TRAIN MODE: episodes set to 100 for quick testing")
    else:
        episodes = args.episodes

    # GPU / mixed precision setup (informational only; actual config handled in train_agent)
    use_gpu = not args.no_gpu

    if not use_gpu:
        # Oculta las GPUs para forzar uso de CPU
        try:
            tf.config.set_visible_devices([], 'GPU')
        except Exception:
            pass
        print("❌ GPU disabled by user. Running on CPU.")
    else:
        print("🔍 GPU preferred (train_agent will try to configure it).")

    # Solo activa mixed precision si hay GPU real
    if args.mixed_precision and use_gpu and tf.config.list_physical_devices('GPU'):
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("🚀 Mixed precision enabled (global policy set to mixed_float16)")
        except Exception as e:
            print(f"⚠️ Could not enable mixed precision: {e}")
    else:
        tf.keras.mixed_precision.set_global_policy('float32')

    # Directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    # Print configuration summary
    print(f"\n📈 Launching training")
    print(f"  - symbol (single): {args.symbol}")
    print(f"  - assets (multi): {args.assets}")
    print(f"  - interval: {args.interval}, start: {args.start_date}, end: {args.end_date}")
    print(f"  - episodes: {episodes}, traj size: {args.trajectory_size}, batch: {args.batch_size}, epochs: {args.epochs}")
    print(f"  - lookback window: {args.lookback_window}, initial balance: {args.initial_balance}")
    print(f"  - commission: {args.commission}, save_freq: {args.save_freq}")
    print(f"  - resume_from: {args.resume_from}, use_lr_schedule: {args.use_lr_schedule}")
    print("")

    # Call train_agent forwarding both 'symbol' and 'assets' (train_agent handles None assets)
    train_agent(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        test_split=args.test_split,
        lookback_window_size=args.lookback_window,
        episodes=episodes,
        trajectory_size=args.trajectory_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_balance=args.initial_balance,
        save_freq=args.save_freq,
        commission=args.commission,
        use_gpu=use_gpu,
        start_episode=args.resume_from,
        use_lr_schedule=args.use_lr_schedule,
        assets=args.assets  # <-- pasamos lista de assets (o None si no se indicó)
    )
