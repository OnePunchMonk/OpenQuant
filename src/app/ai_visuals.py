"""AI Brain Visualization and P&L Card Generator.

Provides engaging visual elements for strategy generation and result sharing.
"""
import streamlit as st
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFont
import io


class AIBrain:
    """Simulates AI agent thinking process with real-time log display."""
    
    THINKING_STEPS = [
        "🔍 Analyzing market regime...",
        "📊 Computing technical indicators...",
        "🧮 Calculating RSI divergence on {}...",
        "📈 Detecting volatility spike patterns...",
        "🎯 Evaluating momentum signals...",
        "⚡ Identifying breakout opportunities...",
        "🔮 Analyzing correlation matrix...",
        "💡 Generating strategy hypothesis...",
        "🎲 Selecting optimal strategy template...",
        "⚙️ Optimizing strategy parameters...",
        "🧪 Running Monte Carlo simulation...",
        "📉 Calculating risk metrics...",
        "✅ Strategy generation complete!"
    ]
    
    @staticmethod
    def show_thinking(asset_name: str = "SPY", strategy_type: str = "momentum"):
        """
        Display AI brain thinking process with animated text.
        
        Args:
            asset_name: Asset being analyzed
            strategy_type: Type of strategy being generated
        """
        brain_container = st.empty()
        log_container = st.empty()
        
        with brain_container.container():
            st.markdown("### 🧠 AI Brain Activity")
            st.caption(f"Analyzing {asset_name} for {strategy_type} strategy...")
        
        log_text = ""
        
        # Show thinking steps with delays
        for i, step in enumerate(AIBrain.THINKING_STEPS):
            # Format step with asset name if placeholder exists
            formatted_step = step.format(asset_name) if '{}' in step else step
            
            # Add step to log
            log_text += f"[{datetime.now().strftime('%H:%M:%S')}] {formatted_step}\n"
            
            # Update log display with typing effect
            with log_container.container():
                st.code(log_text, language="log")
            
            # Variable delay for realism
            delay = random.uniform(0.3, 0.8) if i < len(AIBrain.THINKING_STEPS) - 1 else 1.0
            time.sleep(delay)
        
        # Success message
        brain_container.success("✨ Strategy Generated Successfully!")
        time.sleep(0.5)
    
    @staticmethod
    def show_thinking_minimal(placeholder, steps: int = 5):
        """
        Lightweight version for quick operations.
        
        Args:
            placeholder: Streamlit placeholder to update
            steps: Number of thinking steps to show
        """
        selected_steps = random.sample(AIBrain.THINKING_STEPS[:-1], min(steps, len(AIBrain.THINKING_STEPS)-1))
        selected_steps.append(AIBrain.THINKING_STEPS[-1])  # Always end with complete
        
        log_text = ""
        for step in selected_steps:
            log_text += f"⚡ {step}\n"
            placeholder.code(log_text, language="log")
            time.sleep(0.4)


class PLCardGenerator:
    """Generates Instagram-ready P&L cards from backtest results."""
    
    @staticmethod
    def create_card(
        strategy_name: str,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        num_trades: int,
        asset: str = "SPY",
        equity_curve: Optional[list] = None
    ) -> Image.Image:
        """
        Create a beautiful P&L card image.
        
        Args:
            strategy_name: Name of the strategy
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio value
            max_drawdown: Maximum drawdown percentage
            num_trades: Number of trades executed
            asset: Asset symbol
            equity_curve: Optional equity curve data for mini chart
            
        Returns:
            PIL Image object
        """
        # Card dimensions (Instagram post format: 1080x1080)
        width, height = 1080, 1080
        
        # Color scheme
        bg_color = (15, 23, 42)  # Dark blue
        text_color = (255, 255, 255)  # White
        accent_color = (34, 197, 94) if total_return >= 0 else (239, 68, 68)  # Green/Red
        secondary_color = (148, 163, 184)  # Gray
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts (fallback to default if not available)
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            metric_font = ImageFont.truetype("arial.ttf", 96)
            label_font = ImageFont.truetype("arial.ttf", 42)
            small_font = ImageFont.truetype("arial.ttf", 36)
        except:
            title_font = ImageFont.load_default()
            metric_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw header
        draw.text((60, 60), "🚀 AlphaSwarm", fill=text_color, font=title_font)
        draw.text((60, 160), f"{strategy_name} Strategy", fill=secondary_color, font=label_font)
        draw.text((60, 220), f"Asset: {asset}", fill=secondary_color, font=small_font)
        
        # Draw main profit metric
        profit_sign = "+" if total_return >= 0 else ""
        draw.text((60, 340), f"{profit_sign}{total_return:.1f}%", fill=accent_color, font=metric_font)
        draw.text((60, 460), "Total Return", fill=secondary_color, font=label_font)
        
        # Draw divider
        draw.line([(60, 560), (1020, 560)], fill=secondary_color, width=2)
        
        # Draw metrics grid
        metrics_y = 620
        metrics = [
            ("Sharpe Ratio", f"{sharpe_ratio:.2f}"),
            ("Max Drawdown", f"{max_drawdown:.1f}%"),
            ("Trades", f"{num_trades}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            x = 60 + (i * 330)
            draw.text((x, metrics_y), value, fill=text_color, font=label_font)
            draw.text((x, metrics_y + 60), label, fill=secondary_color, font=small_font)
        
        # Draw mini equity curve if provided
        if equity_curve and len(equity_curve) > 1:
            chart_x, chart_y = 60, 800
            chart_w, chart_h = 960, 180
            
            # Normalize curve to fit chart area
            min_val, max_val = min(equity_curve), max(equity_curve)
            val_range = max_val - min_val if max_val != min_val else 1
            
            points = []
            for i, val in enumerate(equity_curve):
                x = chart_x + (i / (len(equity_curve) - 1)) * chart_w
                y = chart_y + chart_h - ((val - min_val) / val_range) * chart_h
                points.append((x, y))
            
            # Draw curve
            if len(points) > 1:
                draw.line(points, fill=accent_color, width=4)
        
        # Draw footer
        footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        draw.text((60, 1000), footer_text, fill=secondary_color, font=small_font)
        draw.text((width - 300, 1000), "openquant.ai", fill=secondary_color, font=small_font)
        
        return img
    
    @staticmethod
    def create_plotly_card(
        strategy_name: str,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        num_trades: int,
        asset: str = "SPY",
        equity_curve: Optional[list] = None
    ) -> go.Figure:
        """
        Create a P&L card using Plotly (alternative to PIL).
        
        Returns:
            Plotly Figure object
        """
        # Create figure with custom layout
        fig = go.Figure()
        
        # Add equity curve if provided
        if equity_curve and len(equity_curve) > 0:
            color = 'green' if total_return >= 0 else 'red'
            fig.add_trace(go.Scatter(
                y=equity_curve,
                mode='lines',
                line=dict(color=color, width=3),
                fill='tozeroy',
                fillcolor=f'rgba({"34,197,94" if total_return >= 0 else "239,68,68"}, 0.2)',
                name='Equity Curve',
                showlegend=False
            ))
        
        # Annotations for metrics
        profit_sign = "+" if total_return >= 0 else ""
        annotations = [
            # Title
            dict(x=0.5, y=0.95, text="🚀 AlphaSwarm", showarrow=False,
                 font=dict(size=28, color='white'), xref='paper', yref='paper'),
            dict(x=0.5, y=0.88, text=f"{strategy_name} Strategy | {asset}", showarrow=False,
                 font=dict(size=16, color='lightgray'), xref='paper', yref='paper'),
            
            # Main metric
            dict(x=0.5, y=0.72, text=f"{profit_sign}{total_return:.1f}%", showarrow=False,
                 font=dict(size=48, color='lightgreen' if total_return >= 0 else 'lightcoral'),
                 xref='paper', yref='paper'),
            dict(x=0.5, y=0.65, text="Total Return", showarrow=False,
                 font=dict(size=14, color='lightgray'), xref='paper', yref='paper'),
            
            # Bottom metrics
            dict(x=0.25, y=0.08, text=f"Sharpe: {sharpe_ratio:.2f}", showarrow=False,
                 font=dict(size=14, color='white'), xref='paper', yref='paper'),
            dict(x=0.5, y=0.08, text=f"Max DD: {max_drawdown:.1f}%", showarrow=False,
                 font=dict(size=14, color='white'), xref='paper', yref='paper'),
            dict(x=0.75, y=0.08, text=f"Trades: {num_trades}", showarrow=False,
                 font=dict(size=14, color='white'), xref='paper', yref='paper'),
        ]
        
        fig.update_layout(
            annotations=annotations,
            plot_bgcolor='rgb(15, 23, 42)',
            paper_bgcolor='rgb(15, 23, 42)',
            height=800,
            width=800,
            margin=dict(l=20, r=20, t=100, b=100),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        )
        
        return fig


def show_ai_brain_demo():
    """Demo function for testing AI Brain visualization."""
    st.title("🧠 AI Brain Visualization Demo")
    
    if st.button("Start AI Thinking"):
        AIBrain.show_thinking(asset_name="BTC/USDT", strategy_type="momentum")


def show_pnl_card_demo():
    """Demo function for testing P&L Card generator."""
    st.title("📊 P&L Card Generator Demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_name = st.text_input("Strategy Name", "Momentum Crossover")
        total_return = st.number_input("Total Return (%)", value=127.5)
        sharpe_ratio = st.number_input("Sharpe Ratio", value=1.85)
    
    with col2:
        max_drawdown = st.number_input("Max Drawdown (%)", value=-15.2)
        num_trades = st.number_input("Number of Trades", value=45, step=1)
        asset = st.text_input("Asset", "SPY")
    
    if st.button("Generate P&L Card"):
        # Generate sample equity curve
        import numpy as np
        equity_curve = [100]
        for _ in range(100):
            equity_curve.append(equity_curve[-1] * (1 + np.random.randn() * 0.02))
        
        # Show Plotly version
        fig = PLCardGenerator.create_plotly_card(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            num_trades=int(num_trades),
            asset=asset,
            equity_curve=equity_curve
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ P&L Card Generated! Right-click to save image.")
