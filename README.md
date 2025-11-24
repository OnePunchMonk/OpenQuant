# OpenQuant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experimental AI-powered automated trading platform with strategy generation, backtesting, and 24/7 paper trading.

## 🚀 Quick Start

### Try Live Demo
**[🌐 Launch Live App](https://share.streamlit.io/)** ← Deploy your own in 5 minutes!

### Run Locally
```bash
git clone https://github.com/OnePunchMonk/OpenQuant.git
cd OpenQuant
pip install -r requirements.txt
streamlit run run_app.py
```

## About

This is a personal project exploring the use of LLMs (specifically Google Gemini) to help generate and backtest quantitative trading strategies. The system uses LangChain for agent orchestration and vectorbt for backtesting.

## What it does

- Fetches market data using yfinance and FRED APIs
- Uses an LLM to generate trading strategy ideas based on market conditions
- Backtests strategies against historical data
- Provides a web interface (Streamlit) to interact with the system
- Supports both stocks and crypto markets

## Tech Stack

- Python 3.10+
- LangChain - LLM orchestration
- Google Gemini API - Strategy generation
- vectorbt - Backtesting engine
- yfinance - Stock market data
- CCXT - Crypto market data
- FRED API - Economic indicators
- Streamlit - Web interface
- Plotly - Interactive charts
- **🖥️ Streamlit**: Interactive web-based dashboard
- **📊 matplotlib + plotly**: Professional trading charts
- **💾 Parquet**: Efficient data storage format

## 📁 Repository Structure

```
AgentQuant/
├── 📋 config.yaml              # Stock universe configuration
├── 🚀 run_app.py              # Application entry point
├── 📊 requirements.txt         # Python dependencies
├── 📁 src/
│   ├── 🤖 agent/              # AI Agent Brain
│   │   ├── langchain_planner.py    # LLM-powered strategy generation
│   │   ├── policy.py               # Trading policies
│   │   └── runner.py               # Agent execution engine
│   ├── 💾 data/               # Data Pipeline
│   │   ├── ingest.py              # Market data fetching
│   │   └── schemas.py             # Data structures
│   ├── ⚙️ features/           # Feature Engineering
│   │   ├── engine.py              # Technical indicators
│   │   └── regime.py              # Market regime detection
│   ├── 📈 strategies/         # Strategy Library
│   │   ├── momentum.py            # Momentum strategies
│   │   ├── multi_strategy.py      # Advanced strategies
│   │   └── strategy_registry.py   # Strategy catalog
│   ├── ⚡ backtest/          # Backtesting Engine
│   │   ├── runner.py              # Backtest execution
│   │   ├── metrics.py             # Performance analytics
│   │   └── simple_backtest.py     # Basic backtesting
│   ├── 📊 visualization/     # Charts & Reports
│   │   └── plots.py               # Interactive visualizations
│   ├── 🖥️ app/               # User Interface
│   │   └── streamlit_app.py       # Web dashboard
│   └── 🔧 utils/             # Utilities
│       ├── config.py              # Configuration management
│       └── logging.py             # System logging
├── 💾 data_store/            # Market data cache
├── 📊 figures/               # Generated charts
└── 🧪 tests/                # Test suite
```

## 🚀 Quick Start (Poetry)

### Step 1: Setup Environment

git clone https://github.com/onepunchmonk/AgentQuant.git
```bash
# Clone the repository
git clone https://github.com/onepunchmonk/AgentQuant.git
cd AgentQuant

# Install Poetry (if not present)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
poetry install --no-root

# Run the dashboard
poetry run streamlit run run_app.py
```

### Docker

```bash
docker build -t alphaswarm-web .
docker run -p 8501:8501 alphaswarm-web
```

### Compose

```bash
docker compose up --build
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/OnePunchMonk/OpenQuant.git
cd OpenQuant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your asset universe in `config.yaml`:
```yaml
universe:
  - "SPY"
  - "QQQ"
  - "TLT"
  
crypto_universe:
  - "BTC/USDT"
  - "ETH/USDT"
```

4. **Set up your API keys** (⚠️ IMPORTANT - Keep these secret!):
```bash
# Copy the template file
cp .env.example .env

# Edit .env and add your real API keys
# This file is gitignored and will NEVER be committed to GitHub
GOOGLE_API_KEY=your_actual_gemini_api_key_here
FRED_API_KEY=your_actual_fred_api_key_here  # optional
```

**Security Note**: The `.env` file is already in `.gitignore` and will not be tracked by Git. Never commit API keys directly in code or config files.

5. Run the app:
```bash
streamlit run run_app.py
```

## Features

- **Strategy Generation**: Uses LLM to propose trading strategies based on market data
- **Pre-built Strategy Blocks**: Select from validated strategy templates (RSI, Bollinger Bands, etc.)
- **Multi-Asset Support**: Works with both stocks (yfinance) and crypto (CCXT)
- **Backtesting**: Test strategies against historical data using vectorbt
- **Interactive Charts**: View results with Plotly visualizations
- **Automated Paper Trading**: Full-featured simulated trading bot
  - Select any strategy (Momentum, Mean Reversion, Breakout, etc.)
  - **Auto-executes trades every hour** based on strategy signals
  - Fetches fresh 1-hour data continuously for crypto
  - Virtual portfolio with $100k starting capital
  - Real-time position tracking with P&L monitoring
  - Automated BUY/SELL signal generation and execution
  - Complete trade and signal history
  - Start/Stop controls for automated trading
  - Manual trading option also available
  - Persistent state across sessions

## Project Structure

```
OpenQuant/
├── src/
│   ├── agent/          # LLM agent logic
│   ├── app/            # Streamlit interface
│   ├── backtest/       # Backtesting engine
│   ├── data/           # Data fetching (yfinance, CCXT, FRED)
│   ├── features/       # Technical indicators
│   ├── strategies/     # Strategy implementations
│   ├── utils/          # Config and logging
│   └── visualization/  # Plotting utilities
├── config.yaml         # Configuration
├── requirements.txt    # Dependencies
└── run_app.py         # Entry point
```

## Notes

- This is an experimental project and not intended for actual trading
- Backtests are based on historical data and don't guarantee future performance
- The LLM-generated strategies should be reviewed before use
- Requires API keys for Google Gemini (required) and FRED (optional)  

### Ready for Production Use? **Almost!** ⚠️

**What Works Today:**
- ✅ Complete strategy research automation
- ✅ Real market data integration  
- ✅ Professional backtesting results
- ✅ Mathematical strategy formulation
- ✅ Risk-adjusted performance metrics
- ✅ Publication-ready visualizations

**Remaining Friction for Live Trading:**
- ⚠️ **Broker Integration**: Need APIs for live order execution
- ⚠️ **Real-time Data**: Currently uses daily data, needs intraday feeds
- ⚠️ **Risk Controls**: Production-grade position limits and stops
- ⚠️ **Regulatory Compliance**: Trade reporting and audit trails
- ⚠️ **Latency Optimization**: Sub-second execution for high-frequency strategies

## 💡 Suggested Future Features

### 🎯 Immediate Enhancements (Next 3 months)
1. **Multi-Asset Classes**: Bonds, commodities, crypto, forex
2. **Intraday Strategies**: Minute/hourly frequency trading
3. **Options Strategies**: Covered calls, protective puts, spreads
4. **Sentiment Integration**: News, social media, earnings calls
5. **ESG Scoring**: Environmental and social impact metrics

### 🚀 Advanced Capabilities (6-12 months)  
6. **Reinforcement Learning**: Self-improving agents through market feedback
7. **Portfolio Optimization**: Modern portfolio theory with constraints
8. **Multi-Strategy Ensembles**: Combine strategies with dynamic allocation
9. **Alternative Data**: Satellite imagery, credit card transactions, weather
10. **Real-time Alerts**: Strategy performance monitoring and notifications

### 🌟 Production Features (12+ months)
11. **Broker Integration**: Interactive Brokers, Alpaca, TD Ameritrade APIs
12. **Paper Trading**: Risk-free live strategy testing
13. **Institutional Features**: Prime brokerage, custody, compliance
14. **Multi-Language Support**: R, Julia, C++ strategy implementation
15. **Cloud Deployment**: Scalable infrastructure on AWS/GCP/Azure

## 📊 Example Output: From Code to Strategy

### Input (config.yaml):
```yaml
universe: ["AAPL", "MSFT", "GOOGL"]
```

### Agent-Generated Strategy Example:

**Strategy Type**: Momentum Cross-Over  
**Mathematical Formula**:
```
Signal(t) = SMA(Close, 21) - SMA(Close, 63)
Position(t) = +1 if Signal(t) > 0, -1 if Signal(t) < 0
Allocation = {AAPL: 40%, MSFT: 35%, GOOGL: 25%}
```

**Performance Metrics**:
- Total Return: 127.3%
- Sharpe Ratio: 1.84  
- Max Drawdown: -12.7%
- Win Rate: 64.2%

**Visual Output**: Interactive charts showing equity curves, drawdown periods, and rolling metrics.

## 🎯 Target Users

### 1. **Individual Investors** 
- Replace expensive fund managers with AI-powered strategies
- No coding knowledge required
- Professional-grade results

### 2. **Quantitative Researchers**
- Accelerate strategy development by 10x
- Focus on high-level ideas vs implementation
- Rapid prototyping and testing

### 3. **Portfolio Managers**  
- Generate alpha through systematic approaches

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `YourUsername/OpenQuant`
   - Branch: `main`
   - Main file: `run_app.py`
   - Click "Deploy!"

3. **Add Secrets** (Settings > Secrets):
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   FRED_API_KEY = "your_fred_api_key"
   ```

4. **Your live app is ready!** 🎉
   - Automated trading runs 24/7
   - Portfolio state persists
   - Share your unique URL

📖 **[Complete Deployment Guide](DEPLOY_GUIDE.md)**

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/OnePunchMonk/OpenQuant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OnePunchMonk/OpenQuant/discussions)
- **Documentation**: See `docs/` folder

---

## License

MIT License - see LICENSE file for details.

## Disclaimer

⚠️ **This is an experimental research project. Not financial advice.** Do not use for actual trading without proper due diligence. Past performance does not guarantee future results. Trading involves risk of loss.
