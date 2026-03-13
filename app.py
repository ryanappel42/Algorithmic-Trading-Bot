import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import plotly.graph_objects as go
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    .metric-card {
        background: #111418;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📈 ML Algorithmic Trading Dashboard")
st.caption("XGBoost signal generator · Trained on 5 years of AAPL data · Built with Python")

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ticker   = st.text_input("Ticker", value="AAPL").upper()
    years    = st.slider("Years of data", 2, 10, 5)
    fwd_days = st.selectbox("Prediction horizon", [3, 5, 10], index=1)
    st.divider()
    st.caption("Model: XGBoost Classifier")
    st.caption("Features: RSI, MACD, Bollinger Bands,")
    st.caption("Moving averages, Volatility, Returns")
    run = st.button("▶ Run Analysis", type="primary", use_container_width=True)

# ── Functions ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(ticker, years):
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df

def add_features(df, fwd_days):
    df = df.copy()
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
    df["target"]      = (df["Close"].shift(-fwd_days) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df, features):
    split = int(len(df) * 0.8)
    X_train = df[features].iloc[:split]
    y_train = df["target"].iloc[:split]
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model, split

def backtest(df, model, features, split):
    X_test  = df[features].iloc[split:]
    y_test  = df["target"].iloc[split:]
    preds   = model.predict(X_test)
    bt               = pd.DataFrame(index=df.index[split:])
    bt["price"]      = df["Close"].iloc[split:].values
    bt["actual"]     = y_test.values
    bt["predicted"]  = preds
    bt["daily_ret"]  = df["Close"].iloc[split:].pct_change().values
    bt["strat_ret"]  = bt["daily_ret"] * bt["predicted"]
    bt["bh_cum"]     = (1 + bt["daily_ret"]).cumprod()
    bt["strat_cum"]  = (1 + bt["strat_ret"]).cumprod()
    return bt, preds, y_test

# ── Main ───────────────────────────────────────────────────────────────────
if run:
    features = ["rsi","macd","macd_sig","bb_high","bb_low",
                "vol_ma","returns_1d","returns_5d","returns_20d",
                "volatility","ma_20","ma_50","ma_cross"]

    with st.spinner(f"Fetching {ticker} data..."):
        raw = load_data(ticker, years)
        df  = add_features(raw, fwd_days)

    with st.spinner("Training XGBoost model..."):
        model, split = train_model(df, features)

    with st.spinner("Running backtest..."):
        bt, preds, y_test = backtest(df, model, features, split)

    # ── Accuracy metrics ───────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score
    acc      = accuracy_score(y_test, preds)
    buy_acc  = bt[bt["predicted"]==1]["actual"].mean()
    strat_ret = bt["strat_cum"].iloc[-1] - 1
    bh_ret    = bt["bh_cum"].iloc[-1] - 1
    sharpe    = (bt["strat_ret"].mean() / bt["strat_ret"].std()) * np.sqrt(252)
    max_dd    = ((bt["strat_cum"] - bt["strat_cum"].cummax()) / bt["strat_cum"].cummax()).min()

    # ── KPI row ────────────────────────────────────────────────────────────
    st.subheader(f"{ticker} — Analysis Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Accuracy",  f"{acc:.1%}")
    c2.metric("Buy Signal Accuracy", f"{buy_acc:.1%}")
    c3.metric("Strategy Return",   f"{strat_ret:.1%}", f"{strat_ret - bh_ret:.1%} vs B&H")
    c4.metric("Sharpe Ratio",      f"{sharpe:.2f}")
    c5.metric("Max Drawdown",      f"{max_dd:.1%}")

    st.divider()

    # ── Signal chart ───────────────────────────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=bt.index, y=bt["price"],
        mode="lines", name="Price",
        line=dict(color="#4a9eff", width=1.5)
    ))
    buys  = bt[bt["predicted"]==1]
    sells = bt[bt["predicted"]==0]
    fig1.add_trace(go.Scatter(
        x=buys.index, y=buys["price"],
        mode="markers", name="Buy signal",
        marker=dict(color="#00d4a0", size=7, symbol="triangle-up")
    ))
    fig1.add_trace(go.Scatter(
        x=sells.index, y=sells["price"],
        mode="markers", name="Sell signal",
        marker=dict(color="#ff4d6d", size=7, symbol="triangle-down")
    ))
    fig1.update_layout(
        title=f"{ticker} — ML Buy/Sell Signals (Test Period)",
        template="plotly_dark", height=420,
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Cumulative returns chart ───────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=bt.index, y=bt["strat_cum"]*100,
        mode="lines", name="ML Strategy",
        line=dict(color="#00d4a0", width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=bt.index, y=bt["bh_cum"]*100,
        mode="lines", name=f"Buy & Hold {ticker}",
        line=dict(color="#4a9eff", width=2)
    ))
    fig2.add_hline(y=100, line_dash="dash", line_color="#4a5568")
    fig2.update_layout(
        title="Strategy vs Buy & Hold — $100 starting value",
        template="plotly_dark", height=380,
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Feature importance ─────────────────────────────────────────────────
    imp = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig3 = go.Figure(go.Bar(
        x=imp["Importance"], y=imp["Feature"],
        orientation="h", marker_color="#3b82f6"
    ))
    fig3.update_layout(
        title="Feature Importance",
        template="plotly_dark", height=380
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Latest signal ──────────────────────────────────────────────────────
    st.divider()
    latest_features = df[features].iloc[-1:]
    latest_pred     = model.predict(latest_features)[0]
    latest_prob     = model.predict_proba(latest_features)[0]
    latest_price    = df["Close"].iloc[-1]

    st.subheader("🔔 Latest Signal")
    col1, col2 = st.columns(2)
    with col1:
        if latest_pred == 1:
            st.success(f"### ▲ BUY — {ticker} @ ${latest_price:.2f}")
            st.write(f"Model confidence: **{latest_prob[1]:.1%}**")
        else:
            st.error(f"### ▼ SELL/HOLD — {ticker} @ ${latest_price:.2f}")
            st.write(f"Model confidence: **{latest_prob[0]:.1%}**")
    with col2:
        st.caption(f"Prediction horizon: {fwd_days} trading days")
        st.caption(f"Based on data up to: {df.index[-1].date()}")
        st.caption("⚠️ For educational purposes only. Not financial advice.")

else:
    st.info("👈 Configure settings in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### What this dashboard does
    - Downloads real stock price data from Yahoo Finance
    - Engineers 13 technical indicators as ML features
    - Trains an XGBoost classifier to predict 5-day price direction
    - Backtests the strategy and compares to buy & hold
    - Shows the latest BUY or SELL signal with model confidence
    """)