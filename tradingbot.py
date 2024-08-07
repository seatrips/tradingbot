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
from scipy.stats import linregress
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

def detect_horizontal_support(df, lookback=100, threshold=0.02):
    lows = df['low'].rolling(window=lookback).min()
    support_levels = []
    for i in range(len(df) - lookback):
        if abs(df['low'].iloc[i] - lows.iloc[i]) / df['low'].iloc[i] < threshold:
            support_levels.append(df['low'].iloc[i])
    return support_levels

def detect_horizontal_resistance(df, lookback=100, threshold=0.02):
    highs = df['high'].rolling(window=lookback).max()
    resistance_levels = []
    for i in range(len(df) - lookback):
        if abs(df['high'].iloc[i] - highs.iloc[i]) / df['high'].iloc[i] < threshold:
            resistance_levels.append(df['high'].iloc[i])
    return resistance_levels

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

def fetch_multi_timeframe_data(exchange, symbol, timeframes):
    data = {}
    for tf in timeframes:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        data[tf] = df
    return data

def calculate_indicators(df):
    df['ema24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['sma67'] = df['close'].rolling(window=67).mean()
    df['rsi'] = calculate_rsi(df)
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(window=period).mean()

def calculate_dynamic_sl_tp(df, entry_price, atr_multiple=2, rr_ratio=3):
    atr = calculate_atr(df).iloc[-1]
    
    sl_distance = atr * atr_multiple
    
    support_levels = detect_horizontal_support(df)
    resistance_levels = detect_horizontal_resistance(df)
    
    nearest_support = min((level for level in support_levels if level < entry_price), default=entry_price - sl_distance)
    nearest_resistance = min((level for level in resistance_levels if level > entry_price), default=entry_price + sl_distance)
    
    adjusted_sl = max(entry_price - sl_distance, nearest_support)
    if entry_price - adjusted_sl > 1.5 * sl_distance:
        adjusted_sl = entry_price - 1.5 * sl_distance
    
    sl_amount = entry_price - adjusted_sl
    tp_amount = sl_amount * rr_ratio
    take_profit = entry_price + tp_amount
    
    if abs(take_profit - nearest_resistance) < atr:
        take_profit = nearest_resistance - atr
    
    return adjusted_sl, take_profit

def calculate_dynamic_position_size(df, available_funds, min_percentage=0.5, max_percentage=0.9):
    trend_strength = calculate_trend_strength(df)
    
    atr = calculate_atr(df).iloc[-1]
    volatility_factor = 1 - (atr / df['close'].iloc[-1])
    
    volume_sma = df['volume'].rolling(window=20).mean().iloc[-1]
    volume_factor = min(df['volume'].iloc[-1] / volume_sma, 2)
    
    rsi = calculate_rsi(df).iloc[-1]
    rsi_factor = 1 - abs(50 - rsi) / 50
    
    current_price = df['close'].iloc[-1]
    support_levels = detect_horizontal_support(df)
    resistance_levels = detect_horizontal_resistance(df)
    nearest_support = max([level for level in support_levels if level < current_price], default=current_price)
    nearest_resistance = min([level for level in resistance_levels if level > current_price], default=current_price)
    
    support_distance = (current_price - nearest_support) / current_price
    resistance_distance = (nearest_resistance - current_price) / current_price
    sr_factor = min(support_distance, resistance_distance)

    combined_factor = (
        abs(trend_strength) * 0.3 +
        volatility_factor * 0.2 +
        volume_factor * 0.2 +
        rsi_factor * 0.15 +
        sr_factor * 0.15
    )

    position_percentage = min_percentage + (max_percentage - min_percentage) * combined_factor

    return available_funds * position_percentage

def calculate_trend_strength(df, period=14):
    close = df['close'].values
    x = np.arange(len(close))
    slope, _, r_value, _, _ = linregress(x[-period:], close[-period:])
    return abs(r_value) * (1 if slope > 0 else -1)

def check_multi_timeframe_conditions(data):
    timeframes = list(data.keys())
    if len(timeframes) < 2:
        logger.error("At least two timeframes are required for analysis")
        return None
    
    # Sort timeframes from shortest to longest
    sorted_timeframes = sorted(timeframes, key=lambda x: pd.Timedelta(x))
    short_tf = sorted_timeframes[0]
    long_tf = sorted_timeframes[-1]
    
    short_tf_data = data[short_tf]
    long_tf_data = data[long_tf]
    
    # Short timeframe conditions
    short_ema_cross = short_tf_data['ema24'].iloc[-1] > short_tf_data['sma67'].iloc[-1] and short_tf_data['ema24'].iloc[-2] <= short_tf_data['sma67'].iloc[-2]
    rsi_bullish = short_tf_data['rsi'].iloc[-1] > 50 and short_tf_data['rsi'].iloc[-1] > short_tf_data['rsi'].iloc[-2]  # RSI above 50 and increasing
    
    # Long timeframe trend
    long_uptrend = long_tf_data['close'].iloc[-1] > long_tf_data['sma67'].iloc[-1]
    
    if short_ema_cross and rsi_bullish and long_uptrend:
        return 'buy'
    elif short_tf_data['ema24'].iloc[-1] < short_tf_data['sma67'].iloc[-1] and short_tf_data['ema24'].iloc[-2] >= short_tf_data['sma67'].iloc[-2]:
        return 'sell'
    else:
        return None

def update_trailing_take_profit(current_price, highest_price, take_profit, trailing_percentage):
    if current_price > highest_price:
        new_take_profit = current_price * (1 - trailing_percentage)
        if new_take_profit > take_profit:
            return new_take_profit, current_price
    return take_profit, highest_price

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
    trailing_percentage = config['trailing_percentage']

    stats = TradingStats()
    last_log_time = datetime.now()

    logger.info("Trading bot is running...")
    logger.info(f"Trading {symbol} on multiple timeframes: {', '.join(timeframes)}")
    logger.info(f"Trailing Take Profit set at {trailing_percentage*100}%")
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
                    highest_price = trade_data.get('highest_price', entry_price)
                    logger.info(f"Loaded trade data. Position: {position}, Entry price: {entry_price}, Amount: {current_amount}, SL: {stop_loss}, TP: {take_profit}, Highest: {highest_price}")
                else:
                    position = None
                    entry_price = 0
                    entry_time = None
                    current_amount = 0
                    stop_loss = 0
                    take_profit = 0
                    highest_price = 0

                # Fetch latest data for all timeframes
                multi_tf_data = fetch_multi_timeframe_data(exchange, symbol, timeframes)
                for tf, df in multi_tf_data.items():
                    multi_tf_data[tf] = calculate_indicators(df)

                current_price = multi_tf_data[timeframes[0]]['close'].iloc[-1]
                signal = check_multi_timeframe_conditions(multi_tf_data)
                trend_strength = calculate_trend_strength(multi_tf_data[timeframes[-1]])

                # Log current market state and trading signals
                logger.info(f"Current state: Price={current_price}, Signal={signal}, Position={position}, Trend Strength={trend_strength:.2f}")

                if position is None and signal == 'buy' and trend_strength > 0:
                    logger.info("Buy signal detected across multiple timeframes.")
                    
                    available_funds = get_balance(exchange, config['quote_currency'])
                    if available_funds is None or available_funds == 0:
                        logger.warning("No funds available for trading.")
                        continue

                    entry_amount = calculate_dynamic_position_size(multi_tf_data[timeframes[0]], available_funds)
                    buy_amount = entry_amount / current_price

                    logger.info(f"Attempting to place buy order for {buy_amount:.8f} {config['base_currency']}...")
                    
                    order = place_order(exchange, symbol, 'buy', buy_amount, current_price)
                    if order:
                        stop_loss, take_profit = calculate_dynamic_sl_tp(multi_tf_data[timeframes[0]], current_price)
                        position = 'long'
                        entry_price = current_price
                        entry_time = datetime.now()
                        current_amount = buy_amount
                        highest_price = current_price
                        logger.info(f"Entered long position at {entry_price} for {current_amount:.8f} {config['base_currency']}")
                        logger.info(f"Stop Loss set at: {stop_loss}")
                        logger.info(f"Initial Take Profit set at: {take_profit}")
                        # Save trade data
                        save_trade_data({
                            'position': position,
                            'entry_price': entry_price,
                            'entry_time': entry_time.isoformat(),
                            'current_amount': current_amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'highest_price': highest_price
                        })
                    else:
                        logger.error("Failed to place buy order. Continuing to next iteration.")

                elif position == 'long':
                    # Update trailing take profit
                    take_profit, highest_price = update_trailing_take_profit(current_price, highest_price, take_profit, trailing_percentage)
                    
                    # Check conditions for closing the position
                    if (signal == 'sell' or
                        current_price <= stop_loss or
                        current_price >= take_profit or
                        trend_strength < -0.5):
                        
                        reason = "Sell signal" if signal == 'sell' else "Stop loss" if current_price <= stop_loss else "Take profit" if current_price >= take_profit else "Trend reversal"
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
                        # Update trade data with new take profit and highest price
                        save_trade_data({
                            'position': position,
                            'entry_price': entry_price,
                            'entry_time': entry_time.isoformat(),
                            'current_amount': current_amount,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'highest_price': highest_price
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
