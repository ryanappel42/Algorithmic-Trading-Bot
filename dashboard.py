import streamlit as st
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algo Trading Bot — Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean light theme styling ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 13px;
        color: #000000;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #000000;
    }
    .metric-sub {
        font-size: 13px;
        margin-top: 4px;
        color: #000000;
    }
    .positive { color: #16a34a; }
    .negative { color: #dc2626; }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #000000;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .signal-buy {
        background: #dcfce7;
        color: #16a34a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .signal-sell {
        background: #fee2e2;
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .tp-badge {
        background: #fef3c7;
        color: #d97706;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .stSelectbox label { color: #000000; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #000000; }
    p { color: #000000; }
    label { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# ── Timezone ───────────────────────────────────────────────────────────────
est = pytz.timezone("US/Eastern")

# ── Shared chart font settings ─────────────────────────────────────────────
CHART_FONT = {"color": "#000000", "size": 12}
CHART_AXIS = {
    "showgrid"  : True,
    "gridcolor" : "#e2e8f0",
    "color"     : "#000000",
    "tickfont"  : {"color": "#000000", "size": 11},
    "titlefont" : {"color": "#000000", "size": 12},
}

# ── Load credentials ───────────────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY") or st.secrets.get("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or st.secrets.get("ALPACA_SECRET_KEY", "")
BASE_URL   = "https://paper-api.alpaca.markets"

TAKE_PROFIT_PCT = 0.10
STARTING_VALUE  = 100000.00

# ── Connect to Alpaca ──────────────────────────────────────────────────────
@st.cache_resource
def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

api = get_api()

# ── Watchlist ──────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "V",    "JPM",   "ORCL", "COST",
    "ADBE", "CRM",  "AMD",   "NFLX", "PYPL",
    "MA",   "UNH",  "HD",    "BAC",  "QCOM"
]

# ── Helper functions ───────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_account():
    account = api.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash"           : float(account.cash),
        "buying_power"   : float(account.buying_power),
    }

@st.cache_data(ttl=300)
def get_positions():
    positions = api.list_positions()
    if not positions:
        return pd.DataFrame()
    data = []
    for p in positions:
        entry_price   = float(p.avg_entry_price)
        current_price = float(p.current_price)
        market_value  = float(p.market_value)
        unrealized_pl = float(p.unrealized_pl)
        plpc          = float(p.unrealized_plpc) * 100
        qty           = int(float(p.qty))

        # Days held estimate
        try:
            cost_basis = float(p.cost_basis)
            days_held  = "—"
        except:
            days_held = "—"

        near_tp = "💰" if plpc >= 8 else ""

        data.append({
            "Ticker"        : p.symbol,
            "Shares"        : qty,
            "Avg Entry"     : f"${entry_price:,.2f}",
            "Current Price" : f"${current_price:,.2f}",
            "Market Value"  : market_value,
            "Unrealized P&L": unrealized_pl,
            "P&L %"         : plpc,
            "Near TP"       : near_tp,
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def get_orders(limit=200):
    orders = api.list_orders(status="all", limit=limit)
    if not orders:
        return pd.DataFrame()
    data = []
    for o in orders:
        data.append({
            "Time"    : pd.to_datetime(o.created_at).strftime("%b %d %Y %I:%M %p"),
            "Ticker"  : o.symbol,
            "Side"    : o.side.upper(),
            "Shares"  : o.qty,
            "Status"  : o.status.upper(),
            "Raw Time": pd.to_datetime(o.created_at),
        })
    return pd.DataFrame(data)

@st.cache_data(ttl=300)
def get_activities(limit=200):
    """Pull account activities to get filled order prices for realized P&L"""
    try:
        activities = api.get_activities(activity_types="FILL", page_size=limit)
        data = []
        for a in activities:
            data.append({
                "time"      : pd.to_datetime(a.transaction_time),
                "ticker"    : a.symbol,
                "side"      : a.side,
                "qty"       : float(a.qty),
                "price"     : float(a.price),
                "amount"    : float(a.qty) * float(a.price)
            })
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

def classify_sell_reason(orders_df):
    """
    Attempts to classify sells as take profit vs regular SELL signal.
    Take profit sells typically happen alongside a BUY signal on the same day
    for the same ticker (bot sold then could rebuy).
    This is a heuristic — not perfect but indicative.
    """
    if orders_df.empty:
        return 0, 0

    sells = orders_df[orders_df["Side"] == "SELL"].copy()
    sells["date"] = pd.to_datetime(sells["Raw Time"]).dt.date

    # Heuristic: if the log contains "Take profit" in the reason we'd know
    # Since we can't read bot logs from here, we estimate:
    # Any sell where the ticker also has a buy on the same day = likely take profit
    buys = orders_df[orders_df["Side"] == "BUY"].copy()
    buys["date"] = pd.to_datetime(buys["Raw Time"]).dt.date

    take_profit_count = 0
    signal_sell_count = 0

    for _, sell in sells.iterrows():
        same_day_buy = buys[
            (buys["Ticker"] == sell["Ticker"]) &
            (buys["date"] == sell["date"])
        ]
        if not same_day_buy.empty:
            take_profit_count += 1
        else:
            signal_sell_count += 1

    return take_profit_count, signal_sell_count

@st.cache_data(ttl=3600)
def get_stock_data(ticker):
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df["rsi"]         = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["macd"]        = ta.trend.MACD(df["Close"]).macd()
    df["macd_sig"]    = ta.trend.MACD(df["Close"]).macd_signal()
    df["bb_high"]     = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["bb_low"]      = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
    df["vol_ma"]      = df["Volume"].rolling(20).mean()
    df["returns_1d"]  = df["Close"].pct_change(1)
    df["returns_5d"]  = df["Close"].pct_change(5)
    df["returns_20d"] = df["Close"].pct_change(20)
    df["volatility"]  = df["returns_1d"].rolling(20).std()
    df["ma_20"]       = df["Close"].rolling(20).mean()
    df["ma_50"]       = df["Close"].rolling(50).mean()
    df["ma_cross"]    = (df["ma_20"] > df["ma_50"]).astype(int)
    df.dropna(inplace=True)
    return df

def get_ml_signal(df):
    try:
        import joblib
        features = ["rsi","macd","macd_sig","bb_high","bb_low",
                    "vol_ma","returns_1d","returns_5d","returns_20d",
                    "volatility","ma_20","ma_50","ma_cross"]
        model = xgb.XGBClassifier()
        model.load_model("models/aapl_xgb_model.ubj")        
        latest = df[features].iloc[-1:]
        pred   = model.predict(latest)[0]
        prob   = model.predict_proba(latest)[0]
        conf   = prob[1] if pred == 1 else prob[0]
        return "BUY" if pred == 1 else "SELL", conf
    except:
        return "N/A", 0.0

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/stock-market.png", width=60)
    st.title("Algo Trading Bot")
    st.caption("Live Paper Trading Dashboard")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Portfolio Overview", "📈 Stock Analysis", "🔄 Bot Activity"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("🤖 Bot runs daily at 9:30 AM EST")
    st.caption("🔄 Data refreshes every 5 minutes")
    st.caption("💰 Take profit: 10% per position")
    st.caption("📄 Paper trading — no real money")
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Portfolio Overview":

    st.title("📊 Portfolio Overview")
    st.caption(f"Last updated: {datetime.now(est).strftime('%B %d, %Y %I:%M %p EST')}")

    try:
        account   = get_account()
        positions = get_positions()
        orders    = get_orders()

        portfolio_value = account["portfolio_value"]
        cash            = account["cash"]
        total_pnl       = portfolio_value - STARTING_VALUE
        pnl_pct         = (total_pnl / STARTING_VALUE) * 100
        total_invested  = portfolio_value - cash
        exposure_pct    = (total_invested / portfolio_value) * 100

        # Take profit stats
        tp_count, signal_sell_count = classify_sell_reason(orders)

        # ── KPI row ────────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${portfolio_value:,.2f}</div>
                <div class="metric-sub">Started at $100,000</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            color = "positive" if total_pnl >= 0 else "negative"
            arrow = "▲" if total_pnl >= 0 else "▼"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value {color}">{arrow} ${abs(total_pnl):,.2f}</div>
                <div class="metric-sub {color}">{pnl_pct:+.2f}%</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cash Available</div>
                <div class="metric-value">${cash:,.2f}</div>
                <div class="metric-sub">{100 - exposure_pct:.1f}% of portfolio</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Deployed Capital</div>
                <div class="metric-value">${total_invested:,.2f}</div>
                <div class="metric-sub">{exposure_pct:.1f}% of portfolio</div>
            </div>""", unsafe_allow_html=True)

        with c5:
            open_pos = len(positions) if not positions.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value">{open_pos}</div>
                <div class="metric-sub">of 20 watchlist stocks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Take profit stats row ──────────────────────────────────────────
        st.markdown('<div class="section-title">Exit Analysis</div>', unsafe_allow_html=True)

        e1, e2, e3, e4 = st.columns(4)

        total_sells = len(orders[orders["Side"] == "SELL"]) if not orders.empty else 0

        with e1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Sells</div>
                <div class="metric-value">{total_sells}</div>
                <div class="metric-sub">All time</div>
            </div>""", unsafe_allow_html=True)

        with e2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💰 Take Profits</div>
                <div class="metric-value positive">{tp_count}</div>
                <div class="metric-sub">Triggered at 10%+</div>
            </div>""", unsafe_allow_html=True)

        with e3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Signal Sells</div>
                <div class="metric-value">{signal_sell_count}</div>
                <div class="metric-sub">Model said SELL</div>
            </div>""", unsafe_allow_html=True)

        with e4:
            tp_rate = (tp_count / total_sells * 100) if total_sells > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Take Profit Rate</div>
                <div class="metric-value">{tp_rate:.0f}%</div>
                <div class="metric-sub">of all exits</div>
            </div>""", unsafe_allow_html=True)

        st.caption("💰 Take profit exits are estimated — positions sold on same day as a rebuy signal are classified as take profit triggers.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Exposure gauge ─────────────────────────────────────────────────
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div class="section-title">Portfolio Exposure</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = exposure_pct,
                delta = {"reference": 40, "valueformat": ".1f",
                         "font": {"color": "#000000"}},
                title = {"text": "% Capital Deployed",
                         "font": {"size": 14, "color": "#000000"}},
                number= {"suffix": "%", "font": {"size": 28, "color": "#000000"}},
                gauge = {
                    "axis"    : {"range": [0, 100], "tickwidth": 1,
                                 "tickcolor": "#000000",
                                 "tickfont": {"color": "#000000"}},
                    "bar"     : {"color": "#3b82f6"},
                    "bgcolor" : "white",
                    "borderwidth": 2,
                    "bordercolor": "#e2e8f0",
                    "steps"   : [
                        {"range": [0, 40],  "color": "#f0fdf4"},
                        {"range": [40, 70], "color": "#fefce8"},
                        {"range": [70, 100],"color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line"     : {"color": "#dc2626", "width": 3},
                        "thickness": 0.75,
                        "value"    : 40
                    }
                }
            ))
            fig_gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                font={"color": "#000000", "size": 12}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption("📊 Shows what percentage of your $100,000 portfolio is currently invested. The red line marks the 40% maximum — the bot will not open new positions beyond this limit, always keeping at least 60% as a cash buffer.")

        # ── Open positions table ───────────────────────────────────────────
        with col2:
            st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)
            if positions.empty:
                st.info("No open positions right now. The bot is holding cash.")
            else:
                display_df = positions.copy()
                display_df["Market Value"]   = display_df["Market Value"].apply(lambda x: f"${x:,.2f}")
                display_df["Unrealized P&L"] = display_df["Unrealized P&L"].apply(lambda x: f"${x:+,.2f}")
                display_df["P&L %"]          = display_df["P&L %"].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(
                    display_df[["Ticker","Shares","Avg Entry","Current Price","Market Value","Unrealized P&L","P&L %","Near TP"]],
                    use_container_width=True,
                    hide_index=True,
                    height=240
                )
            st.caption("📋 💰 = position within 2% of 10% take profit threshold. Unrealized P&L is not locked in until the bot sells.")

        # ── Portfolio composition pie chart ────────────────────────────────
        st.markdown('<div class="section-title">Portfolio Composition</div>', unsafe_allow_html=True)

        if not positions.empty:
            pos_data = get_positions()
            labels   = pos_data["Ticker"].tolist() + ["Cash"]
            values   = pos_data["Market Value"].tolist() + [cash]

            fig_pie = go.Figure(go.Pie(
                labels    = labels,
                values    = values,
                hole      = 0.5,
                textinfo  = "label+percent",
                textfont  = {"size": 13, "color": "#000000"},
                marker    = {"line": {"color": "white", "width": 2}}
            ))
            fig_pie.update_layout(
                height       = 340,
                showlegend   = True,
                legend       = {"orientation": "v", "x": 1, "y": 0.5,
                                "font": {"color": "#000000", "size": 12}},
                paper_bgcolor= "white",
                plot_bgcolor = "white",
                font         = {"color": "#000000", "size": 12},
                annotations  = [{"text": "Portfolio", "x": 0.5, "y": 0.5,
                                 "font_size": 14, "showarrow": False,
                                 "font_color": "#000000"}]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("🥧 Breakdown of how your capital is allocated across open positions and cash. A larger cash slice means the bot is being selective — only investing when it has high confidence signals.")
        else:
            st.info("No positions open — portfolio is 100% cash.")

    except Exception as e:
        st.error(f"Could not connect to Alpaca: {e}")
        st.info("Check that your API keys are configured correctly.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Stock Analysis":

    st.title("📈 Stock Analysis")
    st.caption(f"Last updated: {datetime.now(est).strftime('%B %d, %Y %I:%M %p EST')}")
    st.caption("Live price data with ML buy/sell signals")

    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Select a stock to analyze", WATCHLIST)
    with col2:
        period = st.selectbox("Time period", ["1 month", "3 months", "6 months", "1 year"], index=3)

    period_days = {"1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365}
    days        = period_days[period]

    with st.spinner(f"Loading {ticker} data..."):
        df                 = get_stock_data(ticker)
        signal, confidence = get_ml_signal(df)
        df_plot            = df.tail(days)

    # ── Signal banner ──────────────────────────────────────────────────────
    price      = df["Close"].iloc[-1]
    rsi        = df["rsi"].iloc[-1]
    macd_val   = df["macd"].iloc[-1]
    volatility = df["volatility"].iloc[-1] * 100

    # Check if currently held and P&L
    position_info = ""
    try:
        pos           = api.get_position(ticker)
        pos_plpc      = float(pos.unrealized_plpc) * 100
        pos_pl        = float(pos.unrealized_pl)
        position_info = f"Currently held: {int(float(pos.qty))} shares | P&L: ${pos_pl:+.2f} ({pos_plpc:+.1f}%)"
        if pos_plpc >= 8:
            position_info += " 💰 Near take profit"
    except:
        position_info = "Not currently held"

    st.info(f"**{ticker} Position:** {position_info}")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${price:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        sig_class = "signal-buy" if signal == "BUY" else "signal-sell"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ML Signal</div>
            <div style="margin-top:8px"><span class="{sig_class}">{signal}</span></div>
            <div class="metric-sub" style="margin-top:6px">Confidence: {confidence:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        rsi_color = "#dc2626" if rsi > 70 else "#16a34a" if rsi < 30 else "#000000"
        rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value" style="color:{rsi_color}">{rsi:.1f}</div>
            <div class="metric-sub" style="color:{rsi_color}">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        macd_color = "#16a34a" if macd_val > 0 else "#dc2626"
        macd_label = "Bullish" if macd_val > 0 else "Bearish"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MACD</div>
            <div class="metric-value" style="color:{macd_color}">{macd_val:.2f}</div>
            <div class="metric-sub" style="color:{macd_color}">{macd_label}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Daily Volatility</div>
            <div class="metric-value">{volatility:.2f}%</div>
            <div class="metric-sub">20-day average</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price chart with Bollinger Bands ───────────────────────────────────
    st.markdown('<div class="section-title">Price Chart with Bollinger Bands</div>', unsafe_allow_html=True)

    is_up  = df_plot["Close"].iloc[-1] >= df_plot["Close"].iloc[0]
    color  = "#16a34a" if is_up else "#dc2626"

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["bb_high"],
        line=dict(color="#94a3b8", width=1, dash="dot"),
        name="Bollinger Upper Band", fill=None
    ))
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["bb_low"],
        line=dict(color="#94a3b8", width=1, dash="dot"),
        name="Bollinger Lower Band",
        fill="tonexty",
        fillcolor="rgba(148,163,184,0.1)"
    ))
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["ma_20"],
        line=dict(color="#3b82f6", width=1.5),
        name="20-Day Moving Average"
    ))
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["ma_50"],
        line=dict(color="#f59e0b", width=1.5),
        name="50-Day Moving Average"
    ))
    fig_price.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot["Close"],
        line=dict(color=color, width=2),
        name=f"{ticker} Close Price"
    ))
    fig_price.update_layout(
        height       = 420,
        paper_bgcolor= "white",
        plot_bgcolor = "white",
        font         = {"color": "#000000", "size": 12},
        legend       = {"orientation": "h", "y": 1.08,
                        "font": {"color": "#000000", "size": 11}},
        xaxis        = {**CHART_AXIS, "title": "Date"},
        yaxis        = {**CHART_AXIS, "title": "Price (USD)", "tickprefix": "$"},
        hovermode    = "x unified",
        margin       = dict(l=60, r=20, t=40, b=40)
    )
    st.plotly_chart(fig_price, use_container_width=True)
    st.caption("📈 One year of daily closing prices. The shaded band shows Bollinger Bands — when price touches the upper band the stock may be overbought, lower band may be oversold. Blue line = 20-day moving average, orange = 50-day moving average. When the blue crosses above orange it signals an uptrend.")

    # ── RSI and MACD charts ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">RSI — Relative Strength Index</div>', unsafe_allow_html=True)
        fig_rsi = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#fee2e2", opacity=0.3, line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="#dcfce7", opacity=0.3, line_width=0)
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#dc2626", line_width=1,
                          annotation_text="Overbought (70)",
                          annotation_font_color="#dc2626",
                          annotation_position="top right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#16a34a", line_width=1,
                          annotation_text="Oversold (30)",
                          annotation_font_color="#16a34a",
                          annotation_position="bottom right")
        fig_rsi.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["rsi"],
            line=dict(color="#6366f1", width=2),
            name="RSI (14-day)"
        ))
        fig_rsi.update_layout(
            height       = 260,
            paper_bgcolor= "white",
            plot_bgcolor = "white",
            font         = {"color": "#000000", "size": 12},
            showlegend   = True,
            legend       = {"font": {"color": "#000000", "size": 11}},
            yaxis        = {**CHART_AXIS, "range": [0, 100], "title": "RSI Value"},
            xaxis        = {**CHART_AXIS, "title": "Date"},
            margin       = dict(l=50, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.caption("📉 RSI measures momentum on a 0-100 scale. Above 70 (red zone) means the stock has risen too fast and may pull back — overbought. Below 30 (green zone) means it may have fallen too far and could bounce — oversold. The bot uses RSI as one of its 13 features when generating signals.")

    with col2:
        st.markdown('<div class="section-title">MACD — Moving Average Convergence Divergence</div>', unsafe_allow_html=True)
        macd_colors = ["#16a34a" if v >= 0 else "#dc2626"
                       for v in (df_plot["macd"] - df_plot["macd_sig"])]
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(
            x=df_plot.index,
            y=df_plot["macd"] - df_plot["macd_sig"],
            marker_color=macd_colors,
            name="Histogram (Gap)",
            opacity=0.6
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["macd"],
            line=dict(color="#3b82f6", width=2),
            name="MACD Line"
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["macd_sig"],
            line=dict(color="#f59e0b", width=2),
            name="Signal Line"
        ))
        fig_macd.update_layout(
            height       = 260,
            paper_bgcolor= "white",
            plot_bgcolor = "white",
            font         = {"color": "#000000", "size": 12},
            legend       = {"orientation": "h", "y": 1.1,
                            "font": {"color": "#000000", "size": 11}},
            yaxis        = {**CHART_AXIS, "title": "MACD Value"},
            xaxis        = {**CHART_AXIS, "title": "Date"},
            margin       = dict(l=50, r=20, t=20, b=40)
        )
        st.plotly_chart(fig_macd, use_container_width=True)
        st.caption("📊 MACD shows the relationship between two moving averages of a stock's price. When the blue MACD line crosses above the orange signal line it suggests building upward momentum — bullish. Crossing below suggests weakening momentum — bearish. Green and red bars show the gap between the two lines.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — BOT ACTIVITY
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Bot Activity":

    st.title("🔄 Bot Activity")
    st.caption(f"Last updated: {datetime.now(est).strftime('%B %d, %Y %I:%M %p EST')}")
    st.caption("Complete log of every trade the bot has executed")

    try:
        orders = get_orders(limit=200)

        if orders.empty:
            st.info("No trades executed yet.")
        else:
            # ── Summary stats ──────────────────────────────────────────────
            total_trades = len(orders)
            buys         = len(orders[orders["Side"] == "BUY"])
            sells        = len(orders[orders["Side"] == "SELL"])
            filled       = len(orders[orders["Status"] == "FILLED"])
            tp_count, signal_sell_count = classify_sell_reason(orders)

            c1, c2, c3, c4, c5, c6 = st.columns(6)

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Orders</div>
                    <div class="metric-value">{total_trades}</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Buy Orders</div>
                    <div class="metric-value positive">{buys}</div>
                </div>""", unsafe_allow_html=True)

            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sell Orders</div>
                    <div class="metric-value negative">{sells}</div>
                </div>""", unsafe_allow_html=True)

            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Fill Rate</div>
                    <div class="metric-value">{(filled/total_trades*100):.0f}%</div>
                </div>""", unsafe_allow_html=True)

            with c5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">💰 Take Profits</div>
                    <div class="metric-value positive">{tp_count}</div>
                    <div class="metric-sub">at 10% threshold</div>
                </div>""", unsafe_allow_html=True)

            with c6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Signal Sells</div>
                    <div class="metric-value">{signal_sell_count}</div>
                    <div class="metric-sub">model triggered</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Trade activity chart ───────────────────────────────────────
            st.markdown('<div class="section-title">Trade Activity by Stock</div>',
                        unsafe_allow_html=True)

            ticker_counts = orders.groupby(["Ticker","Side"]).size().reset_index(name="Count")

            fig_bar = px.bar(
                ticker_counts,
                x="Ticker", y="Count", color="Side",
                color_discrete_map={"BUY": "#16a34a", "SELL": "#dc2626"},
                barmode="group",
                labels={"Count": "Number of Orders",
                        "Ticker": "Stock Symbol",
                        "Side": "Order Type"}
            )
            fig_bar.update_layout(
                height       = 320,
                paper_bgcolor= "white",
                plot_bgcolor = "white",
                font         = {"color": "#000000", "size": 12},
                legend_title = "Order Type",
                legend       = {"font": {"color": "#000000", "size": 11},
                                "title_font": {"color": "#000000"}},
                xaxis        = {**CHART_AXIS, "title": "Stock Symbol"},
                yaxis        = {**CHART_AXIS, "title": "Number of Orders"},
                margin       = dict(l=50, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("📊 Total number of buy and sell orders the bot has placed for each stock since launch. Stocks with more activity are ones the model has had stronger or more frequent signals on.")

            # ── Daily trade volume chart ───────────────────────────────────
            st.markdown('<div class="section-title">Daily Trading Activity</div>',
                        unsafe_allow_html=True)

            orders["Date"] = pd.to_datetime(orders["Raw Time"]).dt.date
            daily = orders.groupby(["Date","Side"]).size().reset_index(name="Count")

            fig_daily = px.bar(
                daily,
                x="Date", y="Count", color="Side",
                color_discrete_map={"BUY": "#16a34a", "SELL": "#dc2626"},
                barmode="stack",
                labels={"Count": "Number of Orders", "Date": "Trading Day"}
            )
            fig_daily.update_layout(
                height       = 280,
                paper_bgcolor= "white",
                plot_bgcolor = "white",
                font         = {"color": "#000000", "size": 12},
                legend_title = "Order Type",
                legend       = {"font": {"color": "#000000", "size": 11}},
                xaxis        = {**CHART_AXIS, "title": "Trading Day"},
                yaxis        = {**CHART_AXIS, "title": "Number of Orders"},
                margin       = dict(l=50, r=20, t=20, b=40)
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            st.caption("📅 How many trades the bot executed each day. Busier days typically mean more stocks passed the 60% confidence threshold or take profit rules triggered.")

            # ── Full trade log ─────────────────────────────────────────────
            st.markdown('<div class="section-title">Full Trade Log</div>',
                        unsafe_allow_html=True)

            def color_side(val):
                if val == "BUY":
                    return "color: #16a34a; font-weight: 600"
                elif val == "SELL":
                    return "color: #dc2626; font-weight: 600"
                return ""

            display_orders = orders.drop(columns=["Raw Time"])
            st.dataframe(
                display_orders.style.map(color_side, subset=["Side"]),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            st.caption("📋 Every order the bot has executed since launch. FILLED means the trade went through successfully at market price. Take profit sells appear as SELL orders and can be identified by same-day rebuys of the same ticker.")

    except Exception as e:
        st.error(f"Could not load trade data: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built by Ryan Appel | Fintech & Big Data Analytics | Virginia Tech Pamplin College of Business | Paper trading only — not financial advice")