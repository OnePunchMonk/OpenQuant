"""Automated Paper Trading Engine.

Executes trades automatically based on strategy signals with hourly data updates.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import json

from src.data.ingest import fetch_crypto_ohlcv, fetch_ohlcv_data
from src.features.engine import compute_features
from src.utils.config import config

logger = logging.getLogger(__name__)

# Trading state file
TRADING_STATE_FILE = Path("data_store/trading_state.json")


class AutoTradingEngine:
    """Automated trading engine that executes strategies on live data."""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.active_strategy = None
        self.strategy_params = {}
        self.last_update = None
        self.update_interval = config.get("paper_trading", {}).get("auto_update_interval", 3600)
        self.lookback_hours = config.get("paper_trading", {}).get("lookback_hours", 100)
        self.signals_history = []
        self.load_state()
    
    def load_state(self):
        """Load trading engine state from disk."""
        if TRADING_STATE_FILE.exists():
            try:
                with open(TRADING_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.active_strategy = data.get('active_strategy')
                    self.strategy_params = data.get('strategy_params', {})
                    self.last_update = data.get('last_update')
                    self.signals_history = data.get('signals_history', [])
            except Exception as e:
                logger.warning(f"Could not load trading state: {e}")
    
    def save_state(self):
        """Save trading engine state to disk."""
        TRADING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(TRADING_STATE_FILE, 'w') as f:
                json.dump({
                    'active_strategy': self.active_strategy,
                    'strategy_params': self.strategy_params,
                    'last_update': self.last_update,
                    'signals_history': self.signals_history
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save trading state: {e}")
    
    def activate_strategy(self, strategy_name: str, params: Dict[str, Any], assets: List[str]):
        """Activate a strategy for automated trading."""
        self.active_strategy = {
            'name': strategy_name,
            'params': params,
            'assets': assets,
            'activated_at': datetime.now().isoformat()
        }
        self.strategy_params = params
        self.save_state()
        logger.info(f"Activated strategy: {strategy_name} with params {params}")
    
    def deactivate_strategy(self):
        """Stop automated trading."""
        self.active_strategy = None
        self.strategy_params = {}
        self.save_state()
        logger.info("Deactivated automated trading")
    
    def should_update(self) -> bool:
        """Check if it's time for the next update."""
        if not self.last_update:
            return True
        
        last_time = datetime.fromisoformat(self.last_update)
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed >= self.update_interval
    
    def fetch_latest_data(self, symbol: str, asset_mode: str = 'Crypto', lookback_hours: int = None) -> pd.DataFrame:
        """
        Fetch latest hourly data for trading decisions.
        
        Args:
            symbol: Asset symbol
            asset_mode: 'Crypto' or 'Stocks'
            lookback_hours: Number of hours of historical data to fetch (defaults to config value)
            
        Returns:
            DataFrame with OHLCV data
        """
        if lookback_hours is None:
            lookback_hours = self.lookback_hours
        
        try:
            if asset_mode == 'Crypto':
                # Fetch last N hours of 1h data
                df = fetch_crypto_ohlcv(symbol, timeframe='1h', limit=lookback_hours)
            else:
                # For stocks, fetch recent daily data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_hours)
                df = fetch_ohlcv_data(symbol, start_date, end_date)
                if isinstance(df, dict):
                    df = df.get(symbol)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_signals(self, df: pd.DataFrame, strategy_name: str, params: Dict) -> str:
        """
        Calculate trading signals based on strategy.
        
        Args:
            df: OHLCV data
            strategy_name: Name of strategy (momentum, mean_reversion, etc.)
            params: Strategy parameters
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        if df is None or df.empty or len(df) < 50:
            return 'HOLD'
        
        try:
            # Ensure we have Close column
            if 'Close' not in df.columns and 'close' not in df.columns:
                return 'HOLD'
            
            close_col = 'Close' if 'Close' in df.columns else 'close'
            close = df[close_col]
            
            if strategy_name == 'momentum':
                # Moving average crossover
                fast_window = params.get('fast_window', 10)
                slow_window = params.get('slow_window', 30)
                
                fast_ma = close.rolling(window=fast_window).mean()
                slow_ma = close.rolling(window=slow_window).mean()
                
                # Check crossover
                if len(fast_ma) < 2 or len(slow_ma) < 2:
                    return 'HOLD'
                
                current_fast = fast_ma.iloc[-1]
                current_slow = slow_ma.iloc[-1]
                prev_fast = fast_ma.iloc[-2]
                prev_slow = slow_ma.iloc[-2]
                
                # Golden cross - buy signal
                if prev_fast <= prev_slow and current_fast > current_slow:
                    return 'BUY'
                # Death cross - sell signal
                elif prev_fast >= prev_slow and current_fast < current_slow:
                    return 'SELL'
                else:
                    return 'HOLD'
            
            elif strategy_name == 'mean_reversion':
                # RSI-based mean reversion
                rsi_period = params.get('rsi_period', 14)
                oversold = params.get('oversold_threshold', 30)
                overbought = params.get('overbought_threshold', 70)
                
                # Calculate RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                current_rsi = rsi.iloc[-1]
                
                if current_rsi < oversold:
                    return 'BUY'
                elif current_rsi > overbought:
                    return 'SELL'
                else:
                    return 'HOLD'
            
            elif strategy_name == 'breakout':
                # Bollinger Bands breakout
                window = params.get('window', 20)
                num_std = params.get('num_std', 2)
                
                rolling_mean = close.rolling(window=window).mean()
                rolling_std = close.rolling(window=window).std()
                
                upper_band = rolling_mean + (rolling_std * num_std)
                lower_band = rolling_mean - (rolling_std * num_std)
                
                current_price = close.iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]
                
                if current_price > current_upper:
                    return 'BUY'
                elif current_price < current_lower:
                    return 'SELL'
                else:
                    return 'HOLD'
            
            elif strategy_name == 'volatility':
                # ATR-based volatility strategy
                window = params.get('atr_window', 14)
                multiplier = params.get('atr_multiplier', 2.0)
                
                # Calculate True Range
                high = df['High'] if 'High' in df.columns else df['high']
                low = df['Low'] if 'Low' in df.columns else df['low']
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=window).mean()
                
                # Simple trend following with ATR filter
                sma = close.rolling(window=20).mean()
                current_atr = atr.iloc[-1]
                avg_atr = atr.mean()
                
                # High volatility - go with trend
                if current_atr > avg_atr * multiplier:
                    if close.iloc[-1] > sma.iloc[-1]:
                        return 'BUY'
                    else:
                        return 'SELL'
                else:
                    return 'HOLD'
            
            else:
                # Unknown strategy - hold
                return 'HOLD'
        
        except Exception as e:
            logger.error(f"Error calculating signals: {e}")
            return 'HOLD'
    
    def execute_signal(self, symbol: str, signal: str, current_price: float, asset_mode: str = 'Crypto'):
        """
        Execute trading signal by placing orders.
        
        Args:
            symbol: Asset symbol
            signal: 'BUY', 'SELL', or 'HOLD'
            current_price: Current market price
            asset_mode: 'Crypto' or 'Stocks'
        """
        if signal == 'HOLD':
            return
        
        try:
            # Calculate position size (simple fixed percentage of portfolio)
            position_size_pct = self.strategy_params.get('position_size', 0.1)  # 10% default
            max_position_value = self.portfolio.cash * position_size_pct
            
            if signal == 'BUY':
                # Check if we already have a position
                if symbol in self.portfolio.positions:
                    logger.info(f"Already holding {symbol}, skipping buy signal")
                    return
                
                # Calculate shares to buy
                shares = max_position_value / current_price
                
                # Execute buy order
                success = self.portfolio.execute_order(symbol, 'buy', shares, current_price)
                if success:
                    logger.info(f"✅ BUY executed: {shares:.4f} {symbol} @ ${current_price:.2f}")
                    self.signals_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'strategy': self.active_strategy['name']
                    })
            
            elif signal == 'SELL':
                # Check if we have a position to sell
                if symbol not in self.portfolio.positions:
                    logger.info(f"No position in {symbol}, skipping sell signal")
                    return
                
                # Sell entire position
                shares = self.portfolio.positions[symbol]['shares']
                
                # Execute sell order
                success = self.portfolio.execute_order(symbol, 'sell', shares, current_price)
                if success:
                    logger.info(f"✅ SELL executed: {shares:.4f} {symbol} @ ${current_price:.2f}")
                    self.signals_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'signal': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'strategy': self.active_strategy['name']
                    })
            
            self.save_state()
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def run_trading_cycle(self, asset_mode: str = 'Crypto'):
        """
        Execute one trading cycle: fetch data, calculate signals, execute trades.
        
        Args:
            asset_mode: 'Crypto' or 'Stocks'
        """
        if not self.active_strategy:
            logger.warning("No active strategy - skipping trading cycle")
            return
        
        logger.info(f"🔄 Running trading cycle for {self.active_strategy['name']}")
        
        strategy_name = self.active_strategy['name']
        assets = self.active_strategy['assets']
        
        for symbol in assets:
            try:
                # Fetch latest data
                logger.info(f"📊 Fetching data for {symbol}...")
                df = self.fetch_latest_data(symbol, asset_mode)
                
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Get current price
                close_col = 'Close' if 'Close' in df.columns else 'close'
                current_price = float(df[close_col].iloc[-1])
                
                # Calculate signal
                logger.info(f"🧮 Calculating signal for {symbol}...")
                signal = self.calculate_signals(df, strategy_name, self.strategy_params)
                
                logger.info(f"📡 Signal for {symbol}: {signal} (Price: ${current_price:.2f})")
                
                # Execute signal
                if signal != 'HOLD':
                    self.execute_signal(symbol, signal, current_price, asset_mode)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update last run time
        self.last_update = datetime.now().isoformat()
        self.save_state()
        
        logger.info("✅ Trading cycle complete")
