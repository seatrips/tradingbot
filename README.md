# Cryptocurrency Trading Bot

## Description
This is an advanced cryptocurrency trading bot designed to automate trading on the Phemex exchange. The bot uses multi-timeframe analysis, EMA/SMA crossovers, and RSI indicators to make trading decisions. It features dynamic position sizing, stop-loss and take-profit mechanisms, and flexible timeframe configuration.

## Features
- Automated trading on Phemex exchange
- Multi-timeframe analysis (configurable, default: 1m, 5m, 15m)
- Technical analysis using EMA, SMA, and RSI indicators
- Dynamic position sizing based on available USDT
- Stop-loss and take-profit mechanisms
- Flexible timeframe configuration
- Real-time logging and error handling
- Persistent trade data storage using SQLite
- Configurable trading pause during specified hours

## Requirements
- Python 3.7+
- ccxt
- pandas
- numpy
- pytz

## Installation

### Clone the repository
```
git clone https://github.com/seatrips/tradingbot.git
cd tradingbot
```

### Set up a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install required packages
```
pip install ccxt pandas numpy pytz
```

## Configuration
1. Edit `config.json` with your Phemex API credentials and desired settings:

```json
{
  "api_key": "your_api_key_here",
  "secret_key": "your_secret_key_here",
  "trading_pair": "BTC/USDT",
  "base_currency": "BTC",
  "quote_currency": "USDT",
  "timeframes": ["1m", "5m", "15m"],
  "position_size": 0.8,
  "sleep_time": 60,
  "error_sleep_time": 30,
  "pause_start": "02:00",
  "pause_end": "04:00",
  "timezone": "Europe/Amsterdam"
}
```

## Usage
Run the bot using the following command:
```
python(3) tradingbot.py
```

## Trading Strategy
The bot implements a strategy based on EMA/SMA crossovers and RSI indicators:

1. **Buy Signal**:
   - EMA24 > SMA67 on longest and middle timeframes
   - EMA24 crosses above SMA67 on shortest timeframe
   - RSI > 50 on all timeframes

2. **Sell Signal**:
   - EMA24 crosses below SMA67 on longest timeframe
   - Take-profit or stop-loss hit

3. **Position Sizing**: Uses a specified percentage (default 80%) of available USDT for each trade.

4. **Stop-Loss**: Set at the lowest low of the last 100 candles.

5. **Take-Profit**: Set at 3 times the distance to the stop-loss.

## Risk Management
- Dynamic position sizing (configurable percentage of available USDT)
- Stop-loss and take-profit mechanisms
- Configurable trading hours to avoid high-volatility periods

## Logging
The bot logs all activities, including trades, errors, and periodic statistics. Logs are saved in a file named `trading_log_YYYYMMDD_HHMMSS.log`.

## Data Storage
- Trade data is stored in an SQLite database (`trades.db`) for persistence and analysis.
- Current trade information is stored in a JSON file (`trade_data.json`) for quick access and recovery in case of bot restart.

## Customization
- Adjust timeframes in `config.json` to change the analysis periods.
- Modify `position_size` in `config.json` to change the percentage of USDT used per trade.
- Edit `pause_start` and `pause_end` in `config.json` to set trading pause hours.

## Safety Features
- Graceful shutdown on Ctrl+C
- Error handling for network and exchange issues
- Configurable trading pause during potentially volatile hours

## Disclaimer
This bot is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. The developers of this bot are not responsible for any financial losses incurred through its use.

## Contributing
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/crypto-trading-bot/issues) if you want to contribute.

## License
[MIT](https://choosealicense.com/licenses/mit/)
