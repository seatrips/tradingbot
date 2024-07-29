import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
import sqlite3
import os
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add a file handler
file_handler = logging.FileHandler(f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class TradingStats:
    def __init__(self):
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.start_time = datetime.now()
        self.init_db()

    def init_db(self):
        conn = None
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS trades
                         (entry_price real, exit_price real, entry_time text, exit_time text, profit real, btc_amount real, usdt_amount real)''')
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()

    def add_trade(self, entry_price, exit_price, entry_time, exit_time, btc_amount, usdt_amount):
        profit = (exit_price - entry_price) / entry_price
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'profit': profit,
            'btc_amount': btc_amount,
            'usdt_amount': usdt_amount
        }
        self.trades.append(trade)
        self.total_trades += 1
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        self.total_profit += profit
        
        try:
            self.save_trade_sqlite(trade)
        except sqlite3.Error as e:
            logger.error(f"Error saving trade to SQLite: {e}")

    def save_trade_sqlite(self, trade):
        conn = None
        try:
            conn = sqlite3.connect('trades.db')
            c = conn.cursor()
            c.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?)", 
                      (trade['entry_price'], trade['exit_price'], trade['entry_time'], 
                       trade['exit_time'], trade['profit'], trade['btc_amount'], trade['usdt_amount']))
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_stats(self):
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_profit = self.total_profit / self.total_trades if self.total_trades > 0 else 0
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'avg_profit': avg_profit,
            'runtime': str(datetime.now() - self.start_time)
        }

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_horizontal_support(df, lookback=100, threshold=0.02):
    lows = df['low'].rolling(window=lookback).min()
    support_levels = []
    for i in range(len(df) - lookback):
        if abs(df['low'].iloc[i] - lows.iloc[i]) / df['low'].iloc[i] < threshold:
            support_levels.append(df['low'].iloc[i])
    return support_levels

def is_bounce_from_support(current_price, support_levels, threshold=0.01):
    for level in support_levels:
        if abs(current_price - level) / level < threshold:
            return True
    return False

def generate_price_chart(df, support_levels):
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], label='Price')
    plt.plot(df['timestamp'], df['ema24'], label='EMA24')
    plt.plot(df['timestamp'], df['sma67'], label='SMA67')
    
    for level in support_levels:
        plt.axhline(y=level, color='r', linestyle='--', alpha=0.5)
    
    plt.title('BTC/USDT Price Chart with Indicators')
    plt.xlabel('Time')
    plt.ylabel('Price (USDT)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return image_base64

def load_trades_sqlite():
    conn = None
    try:
        conn = sqlite3.connect('trades.db')
        c = conn.cursor()
        c.execute("SELECT * FROM trades")
        trades = [dict(zip(['entry_price', 'exit_price', 'entry_time', 'exit_time', 'profit', 'btc_amount', 'usdt_amount'], row)) for row in c.fetchall()]
        return trades
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

def save_trade_data(data):
    with open('trade_data.json', 'w') as f:
        json.dump(data, f)

def load_trade_data():
    try:
        with open('trade_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def initialize_exchange():
    while True:
        api_key = input("Enter your Phemex API key: ")
        secret_key = input("Enter your Phemex secret key: ")
        
        exchange = ccxt.phemex({
            'apiKey': api_key,
            'secret': secret_key,
        })
        
        logger.info("Checking API connection...")
        try:
            exchange.fetch_balance()
            logger.info("API connection successful!")
            return exchange
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            retry = input("Connection failed. Do you want to try again? (y/n): ")
            if retry.lower() != 'y':
                logger.info("Exiting program.")
                exit()

def get_usdt_balance(exchange):
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        logger.info(f"Available USDT balance: {usdt_balance}")
        return usdt_balance
    except Exception as e:
        logger.error(f"Error fetching USDT balance: {e}")
        return None

def get_btc_balance(exchange):
    try:
        balance = exchange.fetch_balance()
        btc_balance = balance['BTC']['free']
        logger.info(f"Available BTC balance: {btc_balance}")
        return btc_balance
    except Exception as e:
        logger.error(f"Error fetching BTC balance: {e}")
        return None

def calculate_buy_amount(exchange, current_price):
    usdt_balance = get_usdt_balance(exchange)
    if usdt_balance is None:
        return None

    buy_amount_usdt = usdt_balance * 0.9  # 90% of USDT balance
    buy_amount_btc = buy_amount_usdt / current_price

    logger.info(f"Calculated buy amount: {buy_amount_btc:.8f} BTC (90% of {usdt_balance:.2f} USDT)")
    return buy_amount_btc

def fetch_ohlcv(exchange, symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['sma67'] = df['close'].rolling(window=67).mean()
    df['rsi'] = calculate_rsi(df)
    return df

def check_trading_conditions(df, support_levels):
    current_price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    ema_cross = df['ema24'].iloc[-1] > df['sma67'].iloc[-1] and df['ema24'].iloc[-2] <= df['sma67'].iloc[-2]
    oversold = rsi < 20
    bounce = is_bounce_from_support(current_price, support_levels)
    
    if ema_cross:
        return 'buy_ema_cross'
    elif oversold and bounce:
        return 'buy_oversold_bounce'
    elif df['ema24'].iloc[-1] < df['sma67'].iloc[-1] and df['ema24'].iloc[-2] >= df['sma67'].iloc[-2]:
        return 'sell'
    else:
        return None

def place_order(exchange, symbol, side, amount, price):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        logger.info(f"{side.capitalize()} order placed: {order}")
        return order
    except Exception as e:
        logger.error(f"Error placing {side} order: {e}")
        return None

# Global variable to control the main loop
running = True

def signal_handler(sig, frame):
    global running
    print('You pressed Ctrl+C!')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def cleanup(exchange, stats):
    logger.info("Performing cleanup before exit...")
    
    # Close any open positions
    trade_data = load_trade_data()
    if trade_data and trade_data['position'] == 'long':
        current_amount = trade_data['current_amount']
        current_price = exchange.fetch_ticker('BTC/USDT')['last']
        logger.info(f"Closing open position: selling {current_amount:.8f} BTC at {current_price}")
        place_order(exchange, 'BTC/USDT', 'sell', current_amount, current_price)
    
    # Save final statistics
    final_stats = stats.get_stats()
    logger.info("Final trading statistics:")
    for key, value in final_stats.items():
        logger.info(f"{key}: {value}")
    
    # You might want to save these stats to a file or database here
    
    logger.info("Cleanup complete. Exiting.")

def main():
    global running
    
    # Initialize the Phemex exchange
    exchange = initialize_exchange()

    symbol = 'BTC/USDT'
    timeframe = '1m'
    initial_take_profit = 0.0018  # 0.18% initial take profit
    initial_stop_loss = 0.0006  # 0.06% initial stop loss
    trailing_stop = 0.0003  # 0.03% trailing stop

    stats = TradingStats()
    last_log_time = datetime.now()

    logger.info("Trading bot is running...")
    logger.info(f"Trading {symbol} on {timeframe} timeframe")
    logger.info(f"Using EMA24/SMA67 crossover, RSI oversold, and horizontal support bounce strategy")
    logger.info(f"Initial take profit set at {initial_take_profit*100}%")
    logger.info(f"Initial stop loss set at {initial_stop_loss*100}%")
    logger.info(f"Trailing stop set at {trailing_stop*100}%")
    logger.info(f"Buy amount: 90% of available USDT balance")

    while running:
        try:
            # Load trade data at the start of each iteration
            trade_data = load_trade_data()
            if trade_data:
                position = trade_data['position']
                entry_price = trade_data['entry_price']
                entry_time = datetime.fromisoformat(trade_data['entry_time'])
                current_amount = trade_data['current_amount']
                highest_price = trade_data.get('highest_price', entry_price)
                trailing_active = trade_data.get('trailing_active', False)
                logger.info(f"Loaded trade data. Position: {position}, Entry price: {entry_price}, Amount: {current_amount}, Highest price: {highest_price}, Trailing active: {trailing_active}")
            else:
                position = None
                entry_price = 0
                entry_time = None
                current_amount = 0
                highest_price = 0
                trailing_active = False

            # Fetch latest data
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=100)
            df = calculate_indicators(df)
            support_levels = detect_horizontal_support(df)

            current_price = df['close'].iloc[-1]
            signal = check_trading_conditions(df, support_levels)

            # Log current market state and trading signals
            logger.info(f"Current state: Price={current_price}, Signal={signal}, Position={position}, Trailing active={trailing_active}")

            if position is None and (signal == 'buy_ema_cross' or signal == 'buy_oversold_bounce'):
                logger.info(f"Potential buy signal detected: {signal}")
                
                buy_amount = calculate_buy_amount(exchange, current_price)
                if buy_amount is not None and buy_amount > 0:
                    logger.info(f"Buy signal confirmed. Attempting to place buy order for {buy_amount:.8f} BTC...")
                    
                    order = place_order(exchange, symbol, 'buy', buy_amount, current_price)
                    if order:
                        position = 'long'
                        entry_price = current_price
                        entry_time = datetime.now()
                        current_amount = buy_amount
                        usdt_amount = current_amount * current_price
                        highest_price = current_price
                        trailing_active = False
                        logger.info(f"Entered long position at {entry_price} for {current_amount:.8f} BTC ({usdt_amount:.2f} USDT)")
                        # Save trade data
                        save_trade_data({
                            'position': position,
                            'entry_price': entry_price,
                            'entry_time': entry_time.isoformat(),
                            'current_amount': current_amount,
                            'usdt_amount': usdt_amount,
                            'highest_price': highest_price,
                            'trailing_active': trailing_active
                        })
                    else:
                        logger.error("Failed to place buy order. Continuing to next iteration.")
                else:
                    logger.warning("Unable to calculate buy amount or insufficient funds.")

            elif position == 'long':
                # Update highest price if current price is higher
                if current_price > highest_price:
                    highest_price = current_price
                    logger.info(f"New highest price reached: {highest_price}")
                    save_trade_data({
                        'position': position,
                        'entry_price': entry_price,
                        'entry_time': entry_time.isoformat(),
                        'current_amount': current_amount,
                        'usdt_amount': current_amount * current_price,
                        'highest_price': highest_price,
                        'trailing_active': trailing_active
                    })

                # Check if take profit is reached and activate trailing stop
                if current_price >= entry_price * (1 + initial_take_profit) and not trailing_active:
                    trailing_active = True
                    logger.info(f"Take profit reached. Activating trailing stop at {trailing_stop*100}%")
                    save_trade_data({
                        'position': position,
                        'entry_price': entry_price,
                        'entry_time': entry_time.isoformat(),
                        'current_amount': current_amount,
                        'usdt_amount': current_amount * current_price,
                        'highest_price': highest_price,
                        'trailing_active': trailing_active
                    })

                # Check conditions for closing the position
                if ((signal == 'sell' and not trailing_active) or
                    (not trailing_active and current_price <= entry_price * (1 - initial_stop_loss)) or
                    (trailing_active and current_price <= highest_price * (1 - trailing_stop))):
                    
                    reason = "Sell signal" if signal == 'sell' and not trailing_active else "Stop loss" if not trailing_active else "Trailing stop"
                    logger.info(f"{reason} triggered. Checking actual BTC balance...")
                    
                    actual_btc_balance = get_btc_balance(exchange)
                    if actual_btc_balance is None:
                        logger.error("Unable to fetch BTC balance. Skipping sell order.")
                        continue
                    
                    if actual_btc_balance < current_amount:
                        logger.warning(f"Actual BTC balance ({actual_btc_balance}) is less than expected ({current_amount}). Adjusting sell amount.")
                        current_amount = actual_btc_balance
                    if current_amount > 0:
                        logger.info(f"Attempting to close position of {current_amount:.8f} BTC...")
                        order = place_order(exchange, symbol, 'sell', current_amount, current_price)
                        if order:
                            exit_time = datetime.now()
                            usdt_amount = current_amount * current_price
                            stats.add_trade(entry_price, current_price, entry_time.isoformat(), exit_time.isoformat(), current_amount, usdt_amount)
                            position = None
                            logger.info(f"Closed long position at {current_price} due to {reason.lower()}")
                            # Clear trade data
                            save_trade_data(None)
                        else:
                            logger.error("Failed to place sell order. Will retry in next iteration.")
                    else:
                        logger.warning("No BTC balance to sell. Resetting position.")
                        position = None
                        save_trade_data(None)

            # Log end of iteration
            logger.info("Completed iteration. Waiting for next cycle.")

            time.sleep(60)  # Wait for 1 minute before next iteration

        except ccxt.NetworkError as e:
            logger.error(f"Network error: {e}")
            logger.info("Waiting for 60 seconds before retrying...")
            time.sleep(60)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            logger.info("Waiting for 60 seconds before retrying...")
            time.sleep(60)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            logger.info("Waiting for 60 seconds before retrying...")
            time.sleep(60)

    # After the main loop ends, perform cleanup
    cleanup(exchange, stats)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main. Exiting gracefully.")
        # Note: cleanup is called inside main() when the loop exits
    sys.exit(0)
