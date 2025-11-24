"""Paper Trading live view components.

Provides auto-refreshing Streamlit section showing latest price and recent
agent/log output for a selected asset (stock or crypto).
Includes portfolio tracking, order execution simulation, and P&L monitoring.
Now with automated strategy execution engine.
"""
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from src.data.ingest import fetch_crypto_ohlcv, fetch_ohlcv_data, fetch_universe
from src.utils.logging import tail_log
from src.utils.config import config
from src.app.auto_trader import AutoTradingEngine
from src.strategies.lego_blocks import LEGO_BLOCKS, LEGO_BLOCK_MAP

logger = logging.getLogger(__name__)


# Portfolio state file
PORTFOLIO_FILE = Path("data_store/paper_portfolio.json")


class PaperPortfolio:
    """Manages paper trading portfolio state."""
    
    def __init__(self):
        self.cash = 100000.0  # Default starting cash
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trade_history = []  # List of executed trades
        self.load()
    
    def load(self):
        """Load portfolio state from disk."""
        if PORTFOLIO_FILE.exists():
            try:
                with open(PORTFOLIO_FILE, 'r') as f:
                    data = json.load(f)
                    self.cash = data.get('cash', 100000.0)
                    self.positions = data.get('positions', {})
                    self.trade_history = data.get('trade_history', [])
            except Exception as e:
                st.warning(f"Could not load portfolio: {e}")
    
    def save(self):
        """Save portfolio state to disk."""
        PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump({
                    'cash': self.cash,
                    'positions': self.positions,
                    'trade_history': self.trade_history
                }, f, indent=2)
        except Exception as e:
            st.error(f"Could not save portfolio: {e}")
    
    def execute_order(self, symbol: str, side: str, shares: float, price: float) -> bool:
        """
        Execute a paper trade order.
        
        Args:
            symbol: Asset symbol
            side: 'buy' or 'sell'
            shares: Number of shares/units
            price: Execution price
            
        Returns:
            True if order executed successfully
        """
        cost = shares * price
        
        if side.lower() == 'buy':
            if self.cash < cost:
                st.error(f"Insufficient cash: ${self.cash:,.2f} < ${cost:,.2f}")
                return False
            
            # Update position
            if symbol in self.positions:
                old_shares = self.positions[symbol]['shares']
                old_avg = self.positions[symbol]['avg_price']
                new_shares = old_shares + shares
                new_avg = ((old_shares * old_avg) + (shares * price)) / new_shares
                self.positions[symbol] = {'shares': new_shares, 'avg_price': new_avg}
            else:
                self.positions[symbol] = {'shares': shares, 'avg_price': price}
            
            self.cash -= cost
            
        elif side.lower() == 'sell':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                st.error(f"Insufficient position in {symbol}")
                return False
            
            # Update position
            self.positions[symbol]['shares'] -= shares
            if self.positions[symbol]['shares'] <= 0:
                del self.positions[symbol]
            
            self.cash += cost
        else:
            st.error(f"Invalid side: {side}")
            return False
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'shares': shares,
            'price': price,
            'value': cost
        })
        
        self.save()
        return True
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions."""
        position_value = sum(
            self.positions[sym]['shares'] * prices.get(sym, 0)
            for sym in self.positions
        )
        return self.cash + position_value
    
    def get_pnl(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate P&L for each position."""
        pnl = {}
        for symbol, pos in self.positions.items():
            current_price = prices.get(symbol, 0)
            cost_basis = pos['shares'] * pos['avg_price']
            current_value = pos['shares'] * current_price
            pnl[symbol] = current_value - cost_basis
        return pnl
    
    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = 100000.0
        self.positions = {}
        self.trade_history = []
        self.save()


def _latest_price(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float('nan')
    for col in ["Close", "close", "Adj Close"]:
        if col in df.columns:
            return float(df[col].iloc[-1])
    # Fallback to last numeric value
    return float(df.select_dtypes("number").iloc[-1].iloc[0])


def show_paper_trading(asset_mode: str):
    st.header("📡 Automated Paper Trading")
    
    # Initialize portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = PaperPortfolio()
    
    portfolio = st.session_state.portfolio
    
    # Initialize trading engine
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = AutoTradingEngine(portfolio)
    
    engine = st.session_state.trading_engine
    
    # Automated Trading Control Panel
    st.subheader("🤖 Automated Trading Engine")
    
    with st.expander("⚙️ Strategy Configuration", expanded=not engine.active_strategy):
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy selection
            strategy_options = ["Select Strategy..."] + list(LEGO_BLOCK_MAP.keys())
            selected_strategy = st.selectbox(
                "Select Trading Strategy",
                strategy_options,
                key="auto_strategy_select"
            )
            
            # Asset selection based on mode
            if asset_mode == 'Crypto':
                universe = config.get('crypto_universe', [])
            else:
                universe = config.get('universe', [])
            
            selected_assets = st.multiselect(
                "Select Assets to Trade",
                universe,
                default=[universe[0]] if universe else [],
                key="auto_assets_select"
            )
        
        with col2:
            # Strategy parameters
            if selected_strategy != "Select Strategy..." and selected_strategy in LEGO_BLOCK_MAP:
                block = LEGO_BLOCK_MAP[selected_strategy]
                st.caption(f"**{block['name']}**")
                st.caption(block['description'])
                
                st.markdown("**Parameters:**")
                strategy_params = {}
                for param_name, param_value in block['params'].items():
                    if isinstance(param_value, int):
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=param_value,
                            key=f"auto_param_{param_name}"
                        )
                    elif isinstance(param_value, float):
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=param_value,
                            format="%.4f",
                            key=f"auto_param_{param_name}"
                        )
                
                # Position sizing
                position_size = st.slider(
                    "Position Size (% of portfolio)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key="position_size"
                )
                strategy_params['position_size'] = position_size / 100.0
        
        # Activation controls
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🚀 Start Auto-Trading", type="primary", disabled=bool(engine.active_strategy)):
                if selected_strategy == "Select Strategy..." or not selected_assets:
                    st.error("Please select a strategy and at least one asset")
                else:
                    strategy_name = LEGO_BLOCK_MAP[selected_strategy]['strategy_type']
                    engine.activate_strategy(strategy_name, strategy_params, selected_assets)
                    st.success(f"✅ Auto-trading activated: {selected_strategy}")
                    st.rerun()
        
        with col_btn2:
            if st.button("⏸️ Stop Auto-Trading", type="secondary", disabled=not engine.active_strategy):
                engine.deactivate_strategy()
                st.warning("⏸️ Auto-trading stopped")
                st.rerun()
        
        with col_btn3:
            if st.button("▶️ Run Trading Cycle Now", disabled=not engine.active_strategy):
                with st.spinner("Executing trading cycle..."):
                    engine.run_trading_cycle(asset_mode)
                st.success("✅ Trading cycle complete!")
                st.rerun()
    
    # Active Strategy Status
    if engine.active_strategy:
        st.success("🟢 **Auto-Trading ACTIVE**")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Strategy", engine.active_strategy['name'].replace('_', ' ').title())
        with col_info2:
            st.metric("Trading Assets", len(engine.active_strategy['assets']))
        with col_info3:
            if engine.last_update:
                last_update = datetime.fromisoformat(engine.last_update)
                minutes_ago = int((datetime.now() - last_update).total_seconds() / 60)
                st.metric("Last Update", f"{minutes_ago} min ago")
            else:
                st.metric("Last Update", "Never")
        
        # Auto-refresh logic
        if engine.should_update():
            st.info("⏰ Time for next update! Running trading cycle...")
            with st.spinner("Executing automated trading cycle..."):
                engine.run_trading_cycle(asset_mode)
            st.rerun()
        else:
            # Show countdown to next update
            if engine.last_update:
                last_time = datetime.fromisoformat(engine.last_update)
                elapsed = (datetime.now() - last_time).total_seconds()
                remaining = engine.update_interval - elapsed
                minutes_remaining = int(remaining / 60)
                st.caption(f"⏳ Next automatic cycle in ~{minutes_remaining} minutes")
    else:
        st.info("ℹ️ **Auto-Trading INACTIVE** - Configure and start a strategy above")
    
    st.divider()
    
    # Manual Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"Manual refresh available below")
    with col2:
        if st.button("🔄 Refresh Prices", key="paper_refresh"):
            st.rerun()
    with col3:
        if st.button("🔄 Reset Portfolio", key="reset_portfolio"):
            portfolio.reset()
            engine.deactivate_strategy()
            st.success("Portfolio reset to $100,000 cash")
            st.rerun()
    
    st.divider()
    
    # Asset selection based on mode
    if asset_mode == 'Crypto':
        universe = config.get('crypto_universe', [])
    else:
        universe = config.get('universe', [])
    
    # Fetch latest prices for all assets in universe
    current_prices = {}
    for symbol in universe:
        try:
            if asset_mode == 'Crypto':
                df = fetch_crypto_ohlcv(symbol)
            else:
                data = fetch_ohlcv_data(ticker=symbol)
                df = data.get(symbol)
            
            if df is not None and not df.empty:
                current_prices[symbol] = _latest_price(df)
        except Exception as e:
            st.warning(f"Could not fetch price for {symbol}: {e}")
            current_prices[symbol] = 0.0
    
    # Portfolio Overview Section
    st.subheader("💼 Portfolio Overview")
    
    total_value = portfolio.get_portfolio_value(current_prices)
    initial_value = 100000.0
    total_pnl = total_value - initial_value
    total_pnl_pct = (total_pnl / initial_value) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Value", f"${total_value:,.2f}", 
                 delta=f"{total_pnl_pct:+.2f}%")
    with col2:
        st.metric("Cash", f"${portfolio.cash:,.2f}")
    with col3:
        position_value = total_value - portfolio.cash
        st.metric("Positions Value", f"${position_value:,.2f}")
    with col4:
        st.metric("Total P&L", f"${total_pnl:,.2f}",
                 delta=f"{total_pnl:+,.2f}")
    
    st.divider()
    
    # Current Positions Table
    if portfolio.positions:
        st.subheader("📊 Current Positions")
        
        positions_data = []
        pnl_data = portfolio.get_pnl(current_prices)
        
        for symbol, pos in portfolio.positions.items():
            current_price = current_prices.get(symbol, 0)
            shares = pos['shares']
            avg_price = pos['avg_price']
            current_value = shares * current_price
            cost_basis = shares * avg_price
            pnl = pnl_data.get(symbol, 0)
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
            
            positions_data.append({
                'Symbol': symbol,
                'Shares': f"{shares:.4f}",
                'Avg Price': f"${avg_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${current_value:,.2f}",
                'P&L': f"${pnl:,.2f}",
                'P&L %': f"{pnl_pct:+.2f}%"
            })
        
        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df, use_container_width=True, hide_index=True)
    else:
        st.info("No positions currently held")
    
    st.divider()
    
    # Trading Interface
    st.subheader("⚡ Execute Trade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_asset = st.selectbox("Asset", universe, key="trade_asset")
        side = st.selectbox("Side", ["Buy", "Sell"], key="trade_side")
    
    with col2:
        current_price = current_prices.get(selected_asset, 0)
        st.metric(f"{selected_asset} Price", f"${current_price:,.4f}")
        
        shares = st.number_input("Shares/Units", min_value=0.0001, value=1.0, 
                                step=0.1, format="%.4f", key="trade_shares")
    
    trade_value = shares * current_price
    st.caption(f"Trade Value: ${trade_value:,.2f}")
    
    if st.button(f"Execute {side} Order", type="primary", key="execute_trade"):
        if current_price <= 0:
            st.error("Invalid price - cannot execute trade")
        else:
            success = portfolio.execute_order(selected_asset, side.lower(), shares, current_price)
            if success:
                st.success(f"✅ {side} order executed: {shares:.4f} {selected_asset} @ ${current_price:.2f}")
                st.rerun()
    
    st.divider()
    
    # Price Chart for Selected Asset
    st.subheader(f"📈 {selected_asset} Price Chart")
    
    if asset_mode == 'Crypto':
        df = fetch_crypto_ohlcv(selected_asset)
    else:
        data = fetch_ohlcv_data(ticker=selected_asset)
        df = data.get(selected_asset)
    
    if df is not None and not df.empty:
        tail_df = df.tail(120)
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f'{selected_asset} Price', 'Volume')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=tail_df.index,
                open=tail_df['Open'],
                high=tail_df['High'],
                low=tail_df['Low'],
                close=tail_df['Close'],
                name=selected_asset
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if tail_df['Close'].iloc[i] < tail_df['Open'].iloc[i] else 'green' 
                 for i in range(len(tail_df))]
        fig.add_trace(
            go.Bar(x=tail_df.index, y=tail_df['Volume'], name='Volume',
                  marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {selected_asset}")
    
    st.divider()
    
    # Automated Trading Signals History
    if engine.signals_history:
        st.subheader("🤖 Automated Trading Signals (Last 20)")
        
        signals_df = pd.DataFrame(engine.signals_history[-20:][::-1])
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format for display
        display_signals = signals_df.copy()
        display_signals['shares'] = display_signals['shares'].apply(lambda x: f"{x:.4f}")
        display_signals['price'] = display_signals['price'].apply(lambda x: f"${x:.2f}")
        display_signals.columns = ['Time', 'Symbol', 'Signal', 'Price', 'Shares', 'Strategy']
        
        # Color code signals
        def highlight_signal(row):
            if row['Signal'] == 'BUY':
                return ['background-color: rgba(34, 197, 94, 0.2)'] * len(row)
            elif row['Signal'] == 'SELL':
                return ['background-color: rgba(239, 68, 68, 0.2)'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            display_signals.style.apply(highlight_signal, axis=1),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # Trade History (Manual + Auto)
    if portfolio.trade_history:
        st.subheader("📜 Complete Trade History (Last 20)")
        
        trades_df = pd.DataFrame(portfolio.trade_history[-20:][::-1])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format for display
        display_df = trades_df.copy()
        display_df['shares'] = display_df['shares'].apply(lambda x: f"{x:.4f}")
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
        display_df['value'] = display_df['value'].apply(lambda x: f"${x:,.2f}")
        display_df.columns = ['Time', 'Symbol', 'Side', 'Shares', 'Price', 'Value']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # System Logs
    st.subheader("📋 System Logs")
    log_lines = tail_log(lines=40)
    if log_lines:
        st.code("".join(log_lines), language="text")
    else:
        st.info("No log output available yet.")
