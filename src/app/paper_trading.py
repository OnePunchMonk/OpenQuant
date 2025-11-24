"""Paper Trading live view components.

Provides auto-refreshing Streamlit section showing latest price and recent
agent/log output for a selected asset (stock or crypto).
"""
from datetime import datetime
import streamlit as st
import pandas as pd
from typing import Dict

from src.data.ingest import fetch_crypto_ohlcv, fetch_ohlcv_data, fetch_universe
from src.utils.logging import tail_log
from src.utils.config import config


def _latest_price(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float('nan')
    for col in ["Close", "close", "Adj Close"]:
        if col in df.columns:
            return float(df[col].iloc[-1])
    # Fallback to last numeric value
    return float(df.select_dtypes("number").iloc[-1].iloc[0])


def show_paper_trading(asset_mode: str):
    st.header("📡 Paper Trading Live View")
    refresh_seconds = config.get("paper_trading", {}).get("refresh_seconds", 60)
    st.caption(f"Auto-refresh every {refresh_seconds} seconds")
    
    # Note: st.experimental_autorefresh was removed in newer Streamlit versions
    # Using manual refresh button instead
    if st.button("🔄 Refresh", key="paper_refresh"):
        st.rerun()

    # Asset selection based on mode
    if asset_mode == 'Crypto':
        universe = config.get('crypto_universe', [])
    else:
        universe = config.get('universe', [])

    asset = st.selectbox("Select Asset", universe)

    # Fetch latest data (for crypto we always re-fetch recent window)
    if asset_mode == 'Crypto':
        df = fetch_crypto_ohlcv(asset)
    else:
        data = fetch_ohlcv_data(ticker=asset)
        df = data.get(asset)

    price = _latest_price(df)
    st.metric(label=f"Latest Price ({asset})", value=f"{price:,.4f}")

    # Display mini chart (last 120 points)
    if df is not None and not df.empty:
        tail_df = df.tail(120)
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(
            x=tail_df.index,
            open=tail_df['Open'], high=tail_df['High'], low=tail_df['Low'], close=tail_df['Close'],
            name=asset
        )])
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Agent Thinking / Logs
    st.subheader("🧠 Agent Thinking (Recent Log Tail)")
    log_lines = tail_log(lines=40)
    if log_lines:
        st.code("".join(log_lines), language="text")
    else:
        st.info("No log output available yet.")
