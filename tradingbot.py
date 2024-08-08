import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
import json
import sqlite3
import os
import signal
import sys
import pytz

# Load configuration from JSON file
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Configuration file 'config.json' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error parsing 'config.json'. Please check the file format.")
        sys.exit(1)

# Global configuration
config = load_config()

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

def calculate_indicators(df):
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['sma67'] = df['close'].rolling(window=67).mean()
    df['rsi'] = calculate_rsi(df)
    return df

def fetch_multi_timeframe_data(exchange, symbol, timeframes):
    data = {}
    for tf in timeframes:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        data[tf] = calculate_indicators(df)
    return data

def check_multi_timeframe_conditions(data):
    timeframes = list(data.keys())
    timeframes.sort(key=lambda x: pd.Timedelta(x))  # Sort timeframes
    shortest_tf, middle_tf, longest_tf = timeframes

    conditions = {tf: {} for tf in timeframes}
    
    for tf in timeframes:
        df = data[tf]
        conditions[tf]['ema_above_sma'] = df['ema24'].iloc[-1] > df['sma67'].iloc[-1]
        conditions[tf]['ema_cross_up'] = df['ema24'].iloc[-1] > df['sma67'].iloc[-1] and df['ema24'].iloc[-2] <= df['sma67'].iloc[-2]
        conditions[tf]['ema_cross_down'] = df['ema24'].iloc[-1] < df['sma67'].iloc[-1] and df['ema24'].iloc[-2] >= df['sma67'].iloc[-2]
        conditions[tf]['rsi_above_50'] = df['rsi'].iloc[-1] > 50

    # Buy signal
    buy_condition = (
        conditions[longest_tf]['ema_above_sma'] and
        conditions[middle_tf]['ema_above_sma'] and
        conditions[shortest_tf]['ema_cross_up'] and
        all(conditions[tf]['rsi_above_50'] for tf in timeframes)
    )

    if buy_condition:
        return 'buy'
    
    # Sell signal
    elif conditions[longest_tf]['ema_cross_down']:
        return 'sell'
    
    return None

def calculate_stop_loss(df):
    return df['low'].rolling(window=100).min().iloc[-1]

def calculate_take_profit(entry_price, stop_loss):
    sl_distance = entry_price - stop_loss
    return entry_price + (3 * sl_distance)

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
    exchange = ccxt.phemex({
        'apiKey': config['api_key'],
        'secret': config['secret_key'],
    })
    
    logger.info("Checking API connection...")
    try:
        exchange.fetch_balance()
        logger.info("API connection successful!")
        return exchange
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        sys.exit(1)

def get_balance(exchange, asset):
    try:
        balance = exchange.fetch_balance()
        return balance[asset]['free']
    except Exception as e:
        logger.error(f"Error fetching {asset} balance: {e}")
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
    print('Ctrl+C pressed. Initiating graceful shutdown...')
    running = False

signal.signal(signal.SIGINT, signal_handler)

def cleanup(exchange, stats):
    logger.info("Performing cleanup before exit...")
    
    trade_data = load_trade_data()
    if trade_data and trade_data['position'] == 'long':
        current_amount = trade_data['current_amount']
        current_price = exchange.fetch_ticker(config['trading_pair'])['last']
        logger.info(f"Closing open position: selling {current_amount:.8f} {config['base_currency']} at {current_price}")
        place_order(exchange, config['trading_pair'], 'sell', current_amount, current_price)
    
    final_stats = stats.get_stats()
    logger.info("Final trading statistics:")
    for key, value in final_stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("Cleanup complete. Exiting.")

def is_trading_allowed():
    tz = pytz.timezone(config['timezone'])
    current_time = datetime.now(tz)
    pause_start = current_time.replace(hour=int(config['pause_start'].split(':')[0]), 
                                       minute=int(config['pause_start'].split(':')[1]), 
                                       second=0, microsecond=0)
    pause_end = current_time.replace(hour=int(config['pause_end'].split(':')[0]), 
                                     minute=int(config['pause_end'].split(':')[1]), 
                                     second=0, microsecond=0)
    
    return not (pause_start <= current_time < pause_end)

def main():
    global running
    
    exchange = initialize_exchange()
    symbol = config['trading_pair']
    timeframes = config['timeframes']
    timeframes.sort(key=lambda x: pd.Timedelta(x))
    shortest_tf, middle_tf, longest_tf = timeframes

    stats = TradingStats()
    last_log_time = datetime.now()

    logger.info("Trading bot is running...")
    logger.info(f"Trading {symbol} on multiple timeframes: {', '.join(timeframes)}")
    logger.info("Press Ctrl+C to stop the bot safely.")
    logger.info(f"Trading will be paused between {config['pause_start']} and {config['pause_end']} {config['timezone']} time.")

    try:
        while running:
            try:
                if not is_trading_allowed():
                    logger.info("Trading is currently paused due to time restrictions. Waiting...")
                    time.sleep(60)
                    continue

                # Load trade data at the start of each iteration
                trade_data = load_trade_data()
                if trade_data:
                    position = trade_data['position']
                    entry_price = trade_data['entry_price']
                    entry_time = datetime.fromisoformat(trade_data['entry_time'])
                    current_amount = trade_data['current_amount']
                    stop_loss = trade_data['stop_loss']
                    take_profit = trade_data['take_profit']
                    logger.info(f"Loaded trade data. Position: {position}, Entry price: {entry_price}, Amount: {current_amount}, SL: {stop_loss}, TP: {take_profit}")
                else:
                    position = None
                    entry_price = 0
                    entry_time = None
                    current_amount = 0
                    stop_loss = 0
                    take_profit = 0

                # Fetch latest data for all timeframes
                multi_tf_data = fetch_multi_timeframe_data(exchange, symbol, timeframes)
                current_price = multi_tf_data[shortest_tf]['close'].iloc[-1]
                signal = check_multi_timeframe_conditions(multi_tf_data)

                # Log current market state and trading signals
                logger.info(f"Current state: Price={current_price}, Signal={signal}, Position={position}")

                if position is None and signal == 'buy':
                    logger.info("Buy signal detected across multiple timeframes.")
                    
                    available_funds = get_balance(exchange, config['quote_currency'])
                    if available_funds is None or available_funds == 0:
                        logger.warning("No funds available for trading.")
                        continue

                    entry_amount = available_funds * config['position_size']
                    buy_amount = entry_amount / current_price

                    logger.info(f"Attempting to place buy order for {buy_amount:.8f} {config['base_currency']}...")
                    
                    order = place_order(exchange, symbol, 'buy', buy_amount, current_price)
                    if order:
                        stop_loss = calculate_stop_loss(multi_tf_data[shortest_tf])
                        take_profit = calculate_take_profit(current_price, stop_loss)
                        position = 'long'
                        entry_price = current_price
                        entry_time = datetime.now()
                        current_amount = buy_amount
                        logger.info(f"Entered long position at {entry_price} for {current_amount:.8f} {config['base_currency']}")
                        logger.info(f"Stop Loss set at: {stop_loss}")
                        logger.info(f"Take Profit set at: {take_profit}")
                        # Save trade data
                        save_trade_data({
                            'position': position,
                            'entry_price': entry_price,
                            'entry_time': entry_time.isoformat(),
                            'current_amount': current_amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                    else:
                        logger.error("Failed to place buy order. Continuing to next iteration.")

                elif position == 'long':
                    # Check conditions for closing the position
                    if current_price <= stop_loss or current_price >= take_profit or signal == 'sell':
                        reason = "Stop loss" if current_price <= stop_loss else "Take profit" if current_price >= take_profit else "Sell signal"
                        logger.info(f"{reason} triggered. Checking actual {config['base_currency']} balance...")
                        
                        actual_balance = get_balance(exchange, config['base_currency'])
                        if actual_balance is None:
                            logger.error(f"Unable to fetch {config['base_currency']} balance. Skipping sell order.")
                            continue
                        
                        if actual_balance < current_amount:
                            logger.warning(f"Actual {config['base_currency']} balance ({actual_balance}) is less than expected ({current_amount}). Adjusting sell amount.")
                            current_amount = actual_balance
                        if current_amount > 0:
                            logger.info(f"Attempting to close position of {current_amount:.8f} {config['base_currency']}...")
                            order = place_order(exchange, symbol, 'sell', current_amount, current_price)
                            if order:
                                exit_time = datetime.now()
                                quote_amount = current_amount * current_price
                                stats.add_trade(entry_price, current_price, entry_time.isoformat(), exit_time.isoformat(), current_amount, quote_amount)
                                position = None
                                logger.info(f"Closed long position at {current_price} due to {reason.lower()}")
                                # Clear trade data
                                save_trade_data(None)
                            else:
                                logger.error("Failed to place sell order. Will retry in next iteration.")
                        else:
                            logger.warning(f"No {config['base_currency']} balance to sell. Resetting position.")
                            position = None
                            save_trade_data(None)
                    else:
                        # Update trade data
                        save_trade_data({
                            'position': position,
                            'entry_price': entry_price,
                            'entry_time': entry_time.isoformat(),
                            'current_amount': current_amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })

                # Log trading statistics periodically
                if datetime.now() - last_log_time > timedelta(hours=1):
                    logger.info("Hourly trading statistics:")
                    for key, value in stats.get_stats().items():
                        logger.info(f"{key}: {value}")
                    last_log_time = datetime.now()

                # Log end of iteration
                logger.info("Completed iteration. Waiting for next cycle.")

                time.sleep(config['sleep_time'])  # Wait before next iteration

            except ccxt.NetworkError as e:
                logger.error(f"Network error: {e}")
                logger.info(f"Waiting for {config['error_sleep_time']} seconds before retrying...")
                time.sleep(config['error_sleep_time'])
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                logger.info(f"Waiting for {config['error_sleep_time']} seconds before retrying...")
                time.sleep(config['error_sleep_time'])
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                logger.info(f"Waiting for {config['error_sleep_time']} seconds before retrying...")
                time.sleep(config['error_sleep_time'])

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught. Initiating graceful shutdown...")
    finally:
        logger.info("Exiting main loop. Performing cleanup...")
        cleanup(exchange, stats)

if __name__ == "__main__":
    main()
    sys.exit(0)
