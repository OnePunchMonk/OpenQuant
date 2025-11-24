# Paper Trading Feature Guide

## Overview

The Paper Trading feature provides a complete simulated trading environment where you can:
- Track a virtual portfolio starting with $100,000
- Execute buy/sell orders at real-time prices
- Monitor positions, P&L, and trade history
- View live price charts with candlesticks and volume
- Test strategies without risking real money

## Features

### 💼 Portfolio Management
- **Total Value**: Real-time portfolio valuation including cash and positions
- **Cash Balance**: Available cash for new trades
- **Positions Value**: Current market value of all holdings
- **Total P&L**: Profit/Loss since portfolio inception

### 📊 Position Tracking
- View all current positions with:
  - Number of shares/units held
  - Average purchase price
  - Current market price
  - Position value
  - Unrealized P&L ($ and %)

### ⚡ Trade Execution
- **Buy Orders**: Purchase assets at current market price
  - Validates sufficient cash balance
  - Updates position with new average cost
- **Sell Orders**: Liquidate positions at current price
  - Validates sufficient shares held
  - Automatically closes position if fully sold

### 📈 Live Charts
- Candlestick price chart (last 120 periods)
- Volume bars with color-coded price direction
- Interactive Plotly charts with zoom/pan

### 📜 Trade History
- Complete record of all executed trades
- Displays last 20 trades with:
  - Timestamp
  - Symbol
  - Side (Buy/Sell)
  - Shares executed
  - Execution price
  - Trade value

### 🔄 Portfolio Reset
- Reset portfolio to initial $100,000 cash
- Clears all positions and trade history
- Useful for starting fresh experiments

## How to Use

1. **Select View Mode**: Choose "Paper Trading" from the sidebar
2. **Choose Asset Mode**: Toggle between Stocks or Crypto
3. **Monitor Prices**: All universe assets are fetched automatically
4. **Execute Trades**:
   - Select an asset from the dropdown
   - Choose Buy or Sell
   - Enter number of shares/units
   - Click "Execute Order"
5. **Track Performance**: View real-time P&L in portfolio overview
6. **Review History**: Check trade history at the bottom

## Data Persistence

- Portfolio state is saved to `data_store/paper_portfolio.json`
- Persists across browser sessions
- Survives app restarts
- Use "Reset Portfolio" button to start over

## Asset Support

### Stocks (via yfinance)
- SPY, QQQ, IWM, TLT, GLD (configurable in `config.yaml`)
- Historical data cached locally
- Real-time prices on refresh

### Crypto (via CCXT)
- BTC/USDT, ETH/USDT, DOGE/USDT, PEPE/USDT
- Direct exchange API access
- Configurable exchange (default: Binance)

## Configuration

Edit `config.yaml` to customize:

```yaml
paper_trading:
  refresh_seconds: 60        # Auto-refresh interval
  exchange: "binance"        # Crypto exchange
  timeframe: "1h"           # Chart timeframe

universe:
  - "SPY"
  - "QQQ"
  # Add more stock tickers

crypto_universe:
  - "BTC/USDT"
  - "ETH/USDT"
  # Add more crypto pairs
```

## Tips

- Start with small position sizes to test the interface
- Use the price chart to identify entry/exit points
- Monitor P&L % to track strategy performance
- Reset portfolio when testing different approaches
- Trade history helps analyze your decisions

## Limitations

- **Simulated execution**: No slippage, fees, or market impact modeled
- **Market hours**: Stock prices only update during market hours
- **Fractional shares**: Supports fractional units (useful for crypto)
- **No stop-loss**: Manual position management only

## Future Enhancements

- Automated strategy execution from backtest results
- Stop-loss and take-profit orders
- Position sizing recommendations
- Portfolio analytics dashboard
- Multi-portfolio support
- Export trade log to CSV

## Troubleshooting

**Issue**: Prices not updating
- **Solution**: Click "Refresh Prices" button or wait for auto-refresh

**Issue**: "Insufficient cash" error
- **Solution**: Check cash balance in portfolio overview, or reset portfolio

**Issue**: "Could not fetch price" warning
- **Solution**: Check internet connection and API limits

**Issue**: Portfolio reset doesn't work
- **Solution**: Check write permissions on `data_store/` directory

## Technical Details

- Portfolio state: JSON file in `data_store/paper_portfolio.json`
- Price data: Cached in `data_store/` with parquet format
- Order execution: Instant at current market price
- Position tracking: Average cost basis method
- Trade recording: ISO 8601 timestamps

---

**⚠️ Disclaimer**: This is a simulation tool for educational purposes only. Past performance does not guarantee future results. Always do your own research before trading real money.
