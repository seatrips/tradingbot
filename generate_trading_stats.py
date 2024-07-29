import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import logging
from datetime import datetime, timedelta
import time
import os
import traceback
import signal

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Add a file handler for logging
file_handler = logging.FileHandler(f"stats_generator_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Global flag for graceful shutdown
keep_running = True

def signal_handler(signum, frame):
    global keep_running
    logger.info("Shutdown signal received. Gracefully shutting down...")
    keep_running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_trades_sqlite():
    conn = None
    try:
        logger.debug("Attempting to connect to the database...")
        conn = sqlite3.connect('trades.db')
        logger.debug("Connected to the database. Executing SQL query...")
        df = pd.read_sql_query("SELECT * FROM trades", conn)
        logger.debug(f"SQL query executed. Retrieved {len(df)} rows.")
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['profit_usd'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'] * df['btc_amount'] * df['exit_price'])
        logger.debug("Data processing completed.")
        return df
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")

def get_trade_count():
    conn = None
    try:
        logger.debug("Attempting to get trade count...")
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        logger.debug(f"Trade count retrieved: {count}")
        return count
    except sqlite3.Error as e:
        logger.error(f"Database error while getting trade count: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return 0
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")

def generate_profit_trend_chart(df):
    try:
        logger.debug("Generating profit trend chart...")
        plt.figure(figsize=(10, 5))
        cumulative_profit = df['profit_usd'].cumsum()
        plt.plot(df['exit_time'], cumulative_profit, label='Cumulative Profit/Loss')
        plt.title('Cumulative Profit/Loss Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Profit/Loss (USD)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        logger.debug("Profit trend chart generated successfully.")
        return image_base64
    except Exception as e:
        logger.error(f"Error generating profit trend chart: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return ""

def save_stats_html(df, trend_chart):
    try:
        logger.debug("Preparing HTML content...")
        total_profit = df['profit_usd'].sum()
        total_trades = len(df)
        winning_trades = len(df[df['profit_usd'] > 0])
        losing_trades = len(df[df['profit_usd'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_btc_traded = df['btc_amount'].sum()
        total_usdt_traded = df['usdt_amount'].sum()

        html_content = f"""
        <html>
        <head>
            <title>Trading Bot Statistics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }}
                .container {{ background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1, h2 {{ color: #333; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin-top: 20px; }}
                .summary {{ margin-bottom: 20px; }}
            </style>
            <meta http-equiv="refresh" content="60">
        </head>
        <body>
            <div class="container">
                <h1>Trading Bot Statistics</h1>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total Trades: {total_trades}</p>
                    <p>Winning Trades: {winning_trades}</p>
                    <p>Losing Trades: {losing_trades}</p>
                    <p>Win Rate: {win_rate:.2%}</p>
                    <p>Total Profit/Loss: ${total_profit:.2f}</p>
                    <p>Total BTC Traded: {total_btc_traded:.8f} BTC</p>
                    <p>Total USDT Traded: ${total_usdt_traded:.2f}</p>
                    <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Cumulative Profit/Loss Trend</h2>
                <img src="data:image/png;base64,{trend_chart}" alt="Cumulative Profit/Loss Trend">
                
                <h2>Trade History</h2>
                <table>
                    <tr>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>BTC Amount</th>
                        <th>USDT Amount</th>
                        <th>Profit/Loss (USD)</th>
                    </tr>
                    {''.join(f"<tr><td>{row['entry_time']}</td><td>{row['exit_time']}</td><td>${row['entry_price']:.2f}</td><td>${row['exit_price']:.2f}</td><td>{row['btc_amount']:.8f}</td><td>${row['usdt_amount']:.2f}</td><td>${row['profit_usd']:.2f}</td></tr>" for _, row in df.iterrows())}
                </table>
            </div>
        </body>
        </html>
        """
        
        logger.debug("Writing HTML content to file...")
        with open('data.html', 'w') as f:
            f.write(html_content)
        logger.info("Statistics and trend chart saved to data.html")
    except Exception as e:
        logger.error(f"Error saving stats HTML: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")

def update_stats():
    try:
        logger.debug("Updating stats...")
        df = load_trades_sqlite()
        if not df.empty:
            logger.info(f"Loaded {len(df)} trades from the database.")
            trend_chart = generate_profit_trend_chart(df)
            logger.info("Generated profit trend chart.")
            save_stats_html(df, trend_chart)
            logger.info("Generated and saved HTML report.")
        else:
            logger.warning("No trade data found in the database.")
    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")

def main():
    logger.info("Starting the Auto-updating Trading Stats Generator...")
    
    last_trade_count = 0
    update_interval = 60  # Check for updates every 60 seconds
    heartbeat_interval = 300  # Log heartbeat every 5 minutes
    last_heartbeat = datetime.now()

    while keep_running:
        try:
            current_time = datetime.now()
            
            # Heartbeat
            if (current_time - last_heartbeat).total_seconds() >= heartbeat_interval:
                logger.info("Heartbeat: Script is still running.")
                last_heartbeat = current_time

            current_trade_count = get_trade_count()
            
            if current_trade_count != last_trade_count:
                logger.info(f"New trades detected. Updating stats...")
                update_stats()
                last_trade_count = current_trade_count
            else:
                logger.info("No new trades. Skipping update.")
            
            # Sleep in small intervals to allow for quicker shutdown
            for _ in range(update_interval):
                if not keep_running:
                    break
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            time.sleep(update_interval)  # Wait before retrying

    logger.info("Script has been shut down gracefully.")

if __name__ == "__main__":
    main()
