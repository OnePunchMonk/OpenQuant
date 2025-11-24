# Streamlit Cloud Deployment Configuration

## Repository
- Repository: OnePunchMonk/OpenQuant
- Branch: main
- Main file: run_app.py

## Python Version
- Python 3.10

## Secrets Configuration
Add the following secrets in Streamlit Cloud dashboard (Settings > Secrets):

```toml
# .streamlit/secrets.toml (Streamlit Cloud only)

# Required: Google Gemini API Key
GOOGLE_API_KEY = "your_gemini_api_key_here"

# Optional: FRED API Key for economic data
FRED_API_KEY = "your_fred_api_key_here"
```

## Environment Variables
No additional environment variables needed beyond secrets.

## Build Command
Streamlit Cloud automatically installs dependencies from requirements.txt

## Health Check
The app runs on the default Streamlit port (8501) and is accessible immediately after deployment.

## Features Enabled
- ✅ Automated paper trading (runs every hour)
- ✅ AI Brain visualization
- ✅ P&L card generation
- ✅ Multi-asset support (stocks + crypto)
- ✅ Strategy Lego blocks
- ✅ Real-time backtesting

## Notes
- The paper trading portfolio persists in data_store/ directory
- Automated trading cycles run every hour when a strategy is active
- No credit card or payment required for deployment
- Free tier supports continuous deployment

## Deployment URL
After deployment, your app will be available at:
https://openquant.streamlit.app (or assigned URL)
