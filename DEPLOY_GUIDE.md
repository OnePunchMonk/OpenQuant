# 🚀 Deploy OpenQuant to Streamlit Cloud

## Quick Deploy (5 minutes)

### Step 1: Go to Streamlit Cloud
1. Visit: https://share.streamlit.io/
2. Sign in with GitHub account

### Step 2: Deploy New App
1. Click **"New app"** button
2. Select your repository: **OnePunchMonk/OpenQuant**
3. Branch: **main**
4. Main file path: **run_app.py**
5. Click **"Deploy!"**

### Step 3: Configure Secrets (IMPORTANT!)
1. While app is deploying, click **Settings** (gear icon)
2. Go to **"Secrets"** section
3. Paste this configuration:

```toml
# Required: Get from https://makersuite.google.com/app/apikey
GOOGLE_API_KEY = "AIzaSy..."

# Optional: Get from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "abc123..."
```

4. Replace with your actual API keys
5. Click **"Save"**

### Step 4: Wait for Deployment
- First deployment takes 2-3 minutes
- Watch the build logs for any errors
- App will auto-restart after secrets are saved

### Step 5: Access Your Live App! 🎉
Your app will be available at:
```
https://[your-app-name].streamlit.app
```

---

## Features Available on Live Link

✅ **Automated Paper Trading**
- Runs 24/7 on Streamlit Cloud
- Auto-executes trades every hour
- Persistent portfolio state

✅ **AI Brain Visualization**
- Real-time strategy generation animation
- 13-step thinking process display

✅ **P&L Card Generator**
- Instagram-ready result cards
- One-click generation
- Social sharing optimized

✅ **Multi-Asset Support**
- Stocks via yfinance
- Crypto via CCXT (Binance)
- Live 1-hour data updates

✅ **Strategy Lego Blocks**
- 5 pre-built strategies
- Customizable parameters
- Instant backtesting

---

## Troubleshooting

### Issue: "GOOGLE_API_KEY not found"
**Solution**: Add API key in Settings > Secrets as shown in Step 3

### Issue: "Module not found"
**Solution**: Check requirements.txt has all dependencies. Redeploy.

### Issue: "App keeps restarting"
**Solution**: 
1. Check logs for errors
2. Verify secrets are saved correctly
3. Ensure no syntax errors in secrets.toml

### Issue: "Automated trading not working"
**Solution**: 
1. Portfolio state persists in cloud storage
2. Auto-updates run every hour (3600 seconds)
3. Click "Run Trading Cycle Now" to test immediately

---

## Important Notes

### Free Tier Limits
- ✅ Unlimited public apps
- ✅ 1 GB RAM per app
- ✅ Persistent storage for portfolio
- ✅ Auto-sleep after inactivity (wakes on access)

### Keeping App Alive
- App sleeps after 7 days of no visitors
- Wakes instantly when accessed
- Consider using a monitoring service (UptimeRobot) to ping every hour

### Security
- ✅ Secrets are encrypted
- ✅ Never shown in logs
- ✅ Not accessible via API
- ✅ Separate from .env file

### Updates
- Push to main branch → auto-deploys
- No manual intervention needed
- Rolling updates (zero downtime)

---

## Monitoring Your Deployment

### View Logs
1. Click **"Manage app"** from Streamlit Cloud dashboard
2. View real-time logs
3. Check automated trading execution
4. Monitor portfolio updates

### Check Portfolio State
- Navigate to Paper Trading view
- Portfolio persists across restarts
- Trade history maintained
- Signal history logged

### Performance Metrics
- Check Streamlit Cloud dashboard
- View resource usage
- Monitor response times
- Track visitor analytics

---

## Share Your Live App

Once deployed, share your link:
- Twitter/X: "Check out my automated crypto trading bot! 🤖 [link]"
- Reddit: Post in r/algotrading, r/streamlit
- GitHub: Add live demo link to README
- LinkedIn: Showcase your project

### Example URLs
- Main app: `https://openquant.streamlit.app`
- Direct to paper trading: `https://openquant.streamlit.app?view=paper_trading`

---

## Advanced Configuration

### Custom Domain (Paid Plan)
- Upgrade to Streamlit Cloud for Teams
- Add custom domain (e.g., trade.yourdomain.com)
- SSL certificate included

### Resource Scaling
- Free tier: 1 GB RAM
- Paid tier: Up to 32 GB RAM
- Contact Streamlit for enterprise needs

### Environment Variables
Add in Settings > Secrets:
```toml
# Override default update interval
AUTO_UPDATE_INTERVAL = "1800"  # 30 minutes

# Increase lookback window
LOOKBACK_HOURS = "200"
```

---

## Next Steps

1. ✅ Deploy to Streamlit Cloud
2. ✅ Configure API keys in secrets
3. ✅ Test automated trading
4. ✅ Share your live link
5. ✅ Monitor performance
6. ✅ Iterate and improve

**Your live app is ready to trade 24/7! 🚀**

---

## Support

- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: https://github.com/OnePunchMonk/OpenQuant/issues

**Happy Trading! 📈**
