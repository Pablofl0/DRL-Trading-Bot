# Test the trained agent
print("Starting testing...")
test_env = CryptoTradingEnv(test_df, lookback_window_size, initial_balance, commission)

# Create a test agent with the same risk parameters
test_agent = PPOAgent(
    input_shape=input_shape, 
    action_space=action_space,
    use_lr_schedule=False  # No need for scheduling during testing
)

# Load the best model from training
test_agent.load_models(
    f'models/{symbol}_actor_best.keras',
    f'models/{symbol}_critic_best.keras'
)

# Initialize risk metrics for testing
#test_agent.reset_risk_metrics(initial_capital=initial_balance)

# Test loop with risk management
test_state = test_env.reset()
done = False
test_rewards = []
position_sizes = []
test_trades = []

# Initialize price history for volatility calculation
if len(test_df) > 100:
    # Get the appropriate price column name
    price_column = 'close'
    if 'close_orig' in test_df.columns:
        price_column = 'close_orig'
    elif 'close' not in test_df.columns and 'close_diff' in test_df.columns:
        print("Warning: Using 'close_diff' for volatility calculation as 'close' is not available")
        price_column = 'close_diff'
        
    #price_history = np.array(test_df[price_column].values[:100])
    #test_agent.update_volatility(price_history)

print("Running test with risk management...")
pbar = tqdm(total=len(test_df) - lookback_window_size, desc="Testing")

while not done:
    # Get action with position sizing
    action, action_probs = test_agent.get_action(test_state, training=False)
    position_prob = action_probs[action]  # Extrae la probabilidad de la acción elegida
    position_sizes.append(position_prob)
    
    # Take action in environment with position sizing
    next_state, reward, done, info = test_env.step(action)
    
    # Calculate PnL for risk tracking
    pnl = info['net_worth'] - test_env.prev_net_worth
    
    # Track trade information if this was a buy or sell action
    if action in [0, 2]:
        test_trades.append({
            'action': 'buy' if action == 0 else 'sell',
            'price': info['current_price'],
            'position_size': position_prob,
            'net_worth': info['net_worth']
        })
        
        # Update risk metrics in the agent
        test_agent.remember(test_state, action, reward, next_state, done, None)
    
    # Update volatility estimate periodically (every 20 steps)
    if test_env.current_step % 20 == 0 and test_env.current_step > lookback_window_size:
        # Get the appropriate price column name
        price_column = 'close'
        if 'close_orig' in test_df.columns:
            price_column = 'close_orig'
        elif 'close' not in test_df.columns and 'close_diff' in test_df.columns:
            price_column = 'close_diff'
            
        #recent_prices = test_df[price_column].values[max(0, test_env.current_step-100):test_env.current_step+1]
        #test_agent.update_volatility(recent_prices)
    
    # Update state and record reward
    test_state = next_state
    test_rewards.append(reward)
    
    # Update progress bar
    pbar.update(1)
    if test_env.current_step % 100 == 0:
        pbar.set_postfix({
            'net_worth': f'${info["net_worth"]:.2f}',
            'position': f'{float(position_prob):.2f}',
            'drawdown': f'{test_agent.current_drawdown:.2%}'
        })

pbar.close()

# Calculate test metrics
test_return = test_env.net_worth - initial_balance
test_return_pct = (test_return / initial_balance) * 100

print(f"\nTest Results for {symbol} with Risk Management:")
print(f"Initial Balance: ${initial_balance:.2f}")
print(f"Final Balance: ${test_env.net_worth:.2f}")
print(f"Return: ${test_return:.2f} ({test_return_pct:.2f}%)")
print(f"Avg Position Size: {np.mean(position_sizes):.2f}")
print(f"Max Drawdown: {test_agent.current_drawdown:.2%}")
print(f"Win Rate: {test_agent.win_count}/{test_agent.total_trades} = {test_agent.win_count/max(1, test_agent.total_trades):.2%}")

# Compare to buy and hold strategy
price_column = 'close_orig' if 'close_orig' in test_df.columns else 'close'
first_price = test_df.iloc[0][price_column]
last_price = test_df.iloc[-1][price_column]
buy_hold_return = (last_price - first_price) / first_price * initial_balance
buy_hold_return_pct = (buy_hold_return / initial_balance) * 100

print(f"\nBuy & Hold Strategy:")
print(f"Return: ${buy_hold_return:.2f} ({buy_hold_return_pct:.2f}%)")

# Save test results with risk metrics
test_results = {
    'symbol': symbol,
    'start_date': start_date,
    'end_date': end_date,
    'initial_balance': initial_balance,
    'final_balance': test_env.net_worth,
    'return': test_return,
    'return_pct': test_return_pct,
    'buy_hold_return': buy_hold_return,
    'buy_hold_return_pct': buy_hold_return_pct,
}

# Save test results and trades to CSV
pd.DataFrame([test_results]).to_csv(f'results/{symbol}_test_results_with_risk.csv', index=False)
pd.DataFrame(test_trades).to_csv(f'results/{symbol}_test_trades.csv', index=False)

return train_history, test_results