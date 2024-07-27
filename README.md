# BTC/USDT Trading Bot

## Overview

This Python-based trading bot is designed to automatically trade Bitcoin (BTC) against USDT on the Phemex cryptocurrency exchange. It uses a combination of technical indicators and market analysis to make trading decisions.

## Features

- Connects to the Phemex exchange using the CCXT library
- Implements a trading strategy based on:
  - EMA24/SMA67 crossover
  - RSI oversold conditions
  - Horizontal support level bounces
- Manages trades with:
  - Take profit at 0.18%
  - Stop loss at 0.06%
- Logs all activities and trades
- Stores trade data in a SQLite database
- Generates price charts with indicators

## Requirements

- Python 3.7+
- ccxt
- pandas
- numpy
- matplotlib
- sqlite3

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install ccxt pandas numpy matplotlib
   ```
3. Set up your Phemex API key and secret

## Configuration

Before running the bot, you need to configure the following:

- Phemex API key and secret
- Trading pair (default is BTC/USDT)
- Timeframe (default is 1 minute)
- Take profit and stop loss percentages

## Usage

Run the bot using:

```
python trading_bot.py
```

The bot will prompt you to enter your Phemex API key and secret when it starts.

## How it Works

1. **Initialization**: 
   - Connects to the Phemex exchange
   - Checks initial balances

2. **Main Loop**:
   - Fetches latest market data
   - Calculates technical indicators (EMA24, SMA67, RSI)
   - Detects horizontal support levels
   - Checks for trading signals

3. **Buy Signal**:
   - Triggered by EMA24 crossing above SMA67 or
   - RSI oversold condition with price bouncing from support
   - Places a market buy order for 90% of available USDT balance

4. **Sell Signal**:
   - Triggered by EMA24 crossing below SMA67 or
   - Take profit or stop loss conditions met
   - Places a market sell order for the entire position

5. **Logging and Data Storage**:
   - All activities and trades are logged
   - Trade data is stored in a SQLite database

## Risk Management

- The bot uses only 90% of available USDT for each trade
- A stop loss is set at 0.06% below the entry price
- A take profit is set at 0.18% above the entry price

## Stats Generator

The project includes a separate script for generating and updating trading statistics:

### Features

- Automatically updates trading statistics
- Generates an HTML report with:
  - Summary of trading performance
  - Cumulative profit/loss trend chart
  - Detailed trade history table
- Refreshes data periodically

### Usage

Run the stats generator using:

```
python stats_generator.py
```

The script will:
1. Connect to the SQLite database
2. Load trade data
3. Calculate performance metrics
4. Generate a profit trend chart
5. Create an HTML report (data.html)
6. Automatically update when new trades are detected

The HTML report includes:
- Total number of trades
- Number of winning and losing trades
- Win rate
- Total profit/loss
- A chart showing cumulative profit/loss over time
- A table with detailed trade history

The report auto-refreshes every 60 seconds in the browser.

## Limitations

- The bot currently only supports trading on the Phemex exchange
- It's designed to trade a single pair (BTC/USDT by default)
- The strategy is not guaranteed to be profitable in all market conditions

## Disclaimer

This bot is for educational purposes only. Use it at your own risk. The authors are not responsible for any financial losses incurred from using this bot.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page] if you want to contribute.

## License

[MIT License](https://choosealicense.com/licenses/mit/)
