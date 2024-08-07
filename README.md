# Cryptocurrency Trading Bot

## Description
This is an advanced cryptocurrency trading bot designed to automate trading on the Phemex exchange, focusing on the BTC/USDT pair. The bot uses multi-timeframe analysis, technical indicators, support/resistance levels, and dynamic position sizing to make trading decisions. It also features a Trailing Take Profit mechanism for optimized profit capture.

## Features
- Automated trading on Phemex exchange
- Multi-timeframe analysis (configurable, default: 5m, 1h, 4h)
- Technical analysis using EMA, SMA, and RSI indicators
- Support and resistance level detection
- Dynamic stop-loss and take-profit calculations
- Trailing Take Profit mechanism
- Trend strength analysis
- Adaptive position sizing based on market conditions
- Real-time logging and error handling
- Persistent trade data storage using SQLite and JSON
- Graceful shutdown and position management
- Configurable trading pause during specified hours

## Requirements
- Python 3.7+
- ccxt
- pandas
- numpy
- matplotlib
- scipy
- pytz

## Installation

### Windows
1. Install Python 3.7+ from [python.org](https://www.python.org/downloads/windows/)
2. Open Command Prompt and run:
   ```
   pip install ccxt pandas numpy matplotlib scipy pytz
   ```

### Raspberry Pi (Raspbian)
1. Open terminal and run:
   ```
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install ccxt pandas numpy matplotlib scipy pytz
   ```

### Ubuntu
1. Open terminal and run:
   ```
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install ccxt pandas numpy matplotlib scipy pytz
   ```

### macOS
1. Install Homebrew if not already installed:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python and required packages:
   ```
   brew install python
   pip3 install ccxt pandas numpy matplotlib scipy pytz
   ```

## Usage
1. Clone this repository:
   ```
   git clone https://github.com/seatrips/tradingbot.git
   cd tradingbot
   ```
2. Create a `config.json` file in the same directory as the script with the following structure:
   ```json
   {
     "api_key": "your_phemex_api_key",
     "secret_key": "your_phemex_secret_key",
     "trading_pair": "BTC/USDT",
     "base_currency": "BTC",
     "quote_currency": "USDT",
     "timeframes": ["5m", "1h", "4h"],
     "trailing_percentage": 0.01,
     "sleep_time": 60,
     "error_sleep_time": 300,
     "pause_start": "02:00",
     "pause_end": "04:00",
     "timezone": "Europe/Amsterdam"
   }
   ```
3. Run the bot using the following command:
   ```
   python tradingbot.py
   ```
   or
   ```
   python3 tradingbot.py
   ```

## Configuration
The bot uses a `config.json` file for configuration. You can adjust the following parameters:
- `api_key`: Your Phemex API key
- `secret_key`: Your Phemex secret key
- `trading_pair`: The trading pair to use (e.g., "BTC/USDT")
- `base_currency`: The base currency of the trading pair (e.g., "BTC")
- `quote_currency`: The quote currency of the trading pair (e.g., "USDT")
- `timeframes`: Array of timeframes to use for analysis (e.g., ["5m", "1h", "4h"])
- `trailing_percentage`: The trailing take profit percentage (e.g., 0.01 for 1%)
- `sleep_time`: Time to wait between iterations in seconds
- `error_sleep_time`: Time to wait after an error occurs before retrying
- `pause_start`: Time to start the trading pause (24-hour format)
- `pause_end`: Time to end the trading pause (24-hour format)
- `timezone`: The timezone for the trading pause times

## Trading Strategy

The bot implements a sophisticated trading strategy that combines multiple technical indicators and market analysis techniques:

1. Multi-timeframe Analysis
2. EMA and SMA Crossovers
3. RSI (Relative Strength Index) Analysis
4. Support and Resistance Level Detection
5. Trend Strength Calculation
6. Dynamic Position Sizing
7. Adaptive Stop-Loss and Take-Profit Levels
8. Trailing Take Profit Mechanism
9. Entry Conditions (EMA/SMA crossover, RSI, and trend strength)
10. Exit Conditions (Signal change, stop-loss, take-profit, trend reversal)
11. Continuous Market Analysis
12. Risk Management (dynamic position sizing and stop-loss)

## Logging
The bot logs all activities, including trades, errors, and periodic statistics. Logs are saved in a file named `trading_log_YYYYMMDD_HHMMSS.log`.

## Data Storage
- Trade data is stored in an SQLite database (`trades.db`) for persistence and analysis.
- Current trade information is stored in a JSON file (`trade_data.json`) for quick access and recovery in case of bot restart.

## Safety Features
- Graceful shutdown on Ctrl+C
- Error handling for network and exchange issues
- Periodic balance checks and position verification
- Configurable trading pause during potentially volatile hours
- Trailing Take Profit to lock in gains during strong trends

## Disclaimer
This bot is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/crypto-trading-bot/issues) if you want to contribute.

## License
[MIT](https://choosealicense.com/licenses/mit/)
