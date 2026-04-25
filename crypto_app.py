import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

import streamlit as st


# ============================================================
# 1️⃣ DATA LOADING & CLEANING
# ============================================================

def load_data(csv_file):
    """
    Load and clean a crypto CSV file.
    Expected columns: Date, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(csv_file)

    # Standardize column names (in case of different capitalizations)
    df.columns = [col.strip().capitalize() for col in df.columns]

    # Ensure required columns exist
    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Convert Date to datetime and set as index
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=numeric_cols)

    return df


# ============================================================
# 3️⃣ STATISTICAL SUMMARY
# ============================================================

def get_statistics(df):
    """
    Return a dictionary with basic statistics for the coin.
    """
    close = df["Close"]
    volume = df["Volume"]

    # Daily returns
    returns = close.pct_change().dropna()

    stats = {
        "Average Close Price": close.mean(),
        "Minimum Close Price": close.min(),
        "Maximum Close Price": close.max(),
        "Average Daily Return": returns.mean(),
        "Volatility (Std of Returns)": returns.std(),
        "Total Days": len(df),
        "Average Volume": volume.mean(),
    }
    return stats


# ============================================================
# 5️⃣ RISK–REWARD SCORE
# ============================================================

def calculate_risk_reward_score(df):
    """
    Calculate a simple risk–reward score based on daily returns.
    Uses a Sharpe-like ratio: annualized_return / annualized_volatility.
    Returns (score, label).
    """
    close = df["Close"]
    returns = close.pct_change().dropna()

    if len(returns) < 2:
        return float("nan"), "Not enough data"

    # Annualize assuming ~252 trading days
    avg_daily_return = returns.mean()
    daily_volatility = returns.std()

    annual_return = avg_daily_return * 252
    annual_volatility = daily_volatility * math.sqrt(252)

    if annual_volatility == 0:
        score = float("nan")
    else:
        score = annual_return / annual_volatility

    # Simple interpretation labels
    if math.isnan(score):
        label = "Not enough data"
    elif score > 1.0:
        label = "Good (High Risk-Reward)"
    elif score > 0.0:
        label = "Moderate"
    else:
        label = "Risky (Low Risk-Reward)"

    return score, label


# ============================================================
# 6️⃣ NEXT-DAY PRICE PREDICTION (ML MODEL)
# ============================================================

def build_supervised(prices, n_lags=7):
    """
    Convert price series into supervised learning format.
    X_t = [p_(t-7), ..., p_(t-1)], y_t = p_t
    """
    X, y = [], []
    for i in range(n_lags, len(prices)):
        X.append(prices[i - n_lags:i])
        y.append(prices[i])
    return np.array(X), np.array(y)


def predict_next_day(df, n_lags=7):
    """
    Train a RandomForestRegressor to predict the next day's closing price.
    Returns predicted_price (float) or None if not enough data.
    """
    close = df["Close"].values

    if len(close) <= n_lags:
        return None

    X, y = build_supervised(close, n_lags=n_lags)

    if len(X) < 10:
        # Not enough data for a decent model
        return None

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # Fit model on all available supervised data
    model.fit(X, y)

    # Last window for prediction
    last_window = close[-n_lags:]
    next_pred = model.predict([last_window])[0]

    return float(next_pred)


# ============================================================
# 7️⃣ COMPARE COINS
# ============================================================

def compare_coins(coin_dfs):
    """
    Compare multiple coins using basic stats and risk–reward score.
    coin_dfs: dict {coin_name: df}
    Returns a DataFrame sorted by RiskRewardScore (descending).
    """
    records = []

    for coin_name, df in coin_dfs.items():
        stats = get_statistics(df)
        score, label = calculate_risk_reward_score(df)

        record = {
            "Coin": coin_name,
            "Avg Close": stats["Average Close Price"],
            "Min Close": stats["Minimum Close Price"],
            "Max Close": stats["Maximum Close Price"],
            "Avg Daily Return": stats["Average Daily Return"],
            "Volatility": stats["Volatility (Std of Returns)"],
            "RiskRewardScore": score,
            "Risk Category": label,
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values("RiskRewardScore", ascending=False)
    return summary_df.reset_index(drop=True)


# ============================================================
# 8️⃣ PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(df, amount=1000.0):
    """
    Simulate investing a fixed amount at the first available close price
    and holding until the last close price.
    Returns a dictionary with final value, profit, and return%.
    """
    close = df["Close"]

    if len(close) < 2:
        return {
            "Initial Investment": amount,
            "Final Value": amount,
            "Profit": 0.0,
            "Return %": 0.0,
        }

    first_price = close.iloc[0]
    last_price = close.iloc[-1]

    if first_price <= 0:
        return {
            "Initial Investment": amount,
            "Final Value": amount,
            "Profit": 0.0,
            "Return %": 0.0,
        }

    # Number of coins bought
    coins = amount / first_price
    final_value = coins * last_price
    profit = final_value - amount
    return_pct = (profit / amount) * 100

    return {
        "Initial Investment": amount,
        "Final Value": final_value,
        "Profit": profit,
        "Return %" : return_pct,
    }


# ============================================================
# 9️⃣ ALERT / THRESHOLD SYSTEM
# ============================================================

def alert_system(df):
    """
    Generate simple alerts based on last day's movement & volume.
    Rules:
      - Drop > 5% -> "Sell/Warning"
      - Jump > 8% -> "Opportunity"
      - Volume > 2x average -> "High Activity"
    Returns a list of alert strings.
    """
    alerts = []

    close = df["Close"]
    volume = df["Volume"]

    if len(close) < 2:
        return ["Not enough data for alerts."]

    # Daily returns
    returns = close.pct_change().dropna()
    last_return = returns.iloc[-1]

    avg_volume = volume.mean()
    last_volume = volume.iloc[-1]

    # Price movement alerts
    if last_return <= -0.05:
        alerts.append(f"⚠️ Price dropped {last_return*100:.2f}% in the last day (possible sell warning).")
    elif last_return >= 0.08:
        alerts.append(f"🚀 Price jumped {last_return*100:.2f}% in the last day (potential opportunity).")

    # Volume alert
    if last_volume >= 2 * avg_volume:
        alerts.append("📊 Trading volume is more than 2× the average (high market activity).")

    if not alerts:
        alerts.append("✅ No major alerts. Market is relatively stable recently.")

    return alerts


# ============================================================
# 2️⃣ PLOTTING (DEFINED NEAR THE END, USED LAST IN UI)
# ============================================================

def plot_graphs(df, coin_name="Coin"):
    """
    Create price, volume, and returns plots.
    Returns 3 matplotlib Figure objects.
    """
    # Close price plot
    fig_price, ax1 = plt.subplots()
    ax1.plot(df.index, df["Close"])
    ax1.set_title(f"{coin_name} - Close Price Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")
    ax1.grid(True)

    # Volume plot
    fig_volume, ax2 = plt.subplots()
    ax2.bar(df.index, df["Volume"])
    ax2.set_title(f"{coin_name} - Volume Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    # Returns plot
    returns = df["Close"].pct_change() * 100
    fig_returns, ax3 = plt.subplots()
    ax3.plot(df.index, returns)
    ax3.set_title(f"{coin_name} - Daily Returns (%)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Daily Return (%)")
    ax3.axhline(0, linestyle="--")
    ax3.grid(True)

    return fig_price, fig_volume, fig_returns


# ============================================================
# 🔟 STREAMLIT APP (USES EVERYTHING ABOVE)
# ============================================================

def streamlit_app():
    st.set_page_config(page_title="Crypto Analysis & Prediction", layout="wide")

    st.title("📈 Crypto Analysis & Prediction App")
    st.write(
        "Use this app to **analyze crypto data**, compute **risk–reward scores**, "
        "predict the **next day's price**, **compare coins**, and visualize everything."
    )

    st.sidebar.header("Step 1: Upload CSV Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV files (one per coin).",
        type="csv",
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👈 Please upload at least one CSV file to begin.")
        return

    # Load all uploaded coins
    coin_dfs = {}
    for file in uploaded_files:
        try:
            df = load_data(file)
            coin_name = file.name.rsplit(".", 1)[0]
            coin_dfs[coin_name] = df
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

    if not coin_dfs:
        st.error("No valid CSV files loaded.")
        return

    # Sidebar controls
    st.sidebar.header("Step 2: Choose Options")
    coin_list = list(coin_dfs.keys())
    selected_coin = st.sidebar.selectbox("Select a coin to analyze", coin_list)

    investment_amount = st.sidebar.number_input(
        "Investment amount for simulation",
        min_value=100.0,
        max_value=1_000_000.0,
        value=1000.0,
        step=100.0,
    )

    df = coin_dfs[selected_coin]

    # --------------------------------------------------------
    # ----------------- COMPARE ALL COINS --------------------
    # --------------------------------------------------------
    
    st.subheader("🏁 Compare All Uploaded Coins")
    comparison_df = compare_coins(coin_dfs)

    if not comparison_df.empty:
        best = comparison_df.iloc[0]
        st.success(
            f"🏆 **Best Overall Coin to Invest In: {best['Coin']}**\n\n"
            f"- Highest Risk–Reward Score: **{best['RiskRewardScore']:.2f}**\n"
            f"- Volatility: **{best['Volatility']:.4f}**\n"
            f"- Avg Daily Return: **{best['Avg Daily Return']:.4f}**"
        )

        st.dataframe(
            comparison_df.style.format({
                "Avg Close": "{:.2f}",
                "Min Close": "{:.2f}",
                "Max Close": "{:.2f}",
                "Avg Daily Return": "{:.4f}",
                "Volatility": "{:.4f}",
                "RiskRewardScore": "{:.2f}",
            })
        )

    else:
        st.write("Not enough data to compare coins.")

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------
    st.subheader(f"📊 Basic Statistics - {selected_coin}")
    stats = get_statistics(df)
    stats_df = pd.DataFrame([stats])
    st.table(stats_df.style.format({
        "Average Close Price": "{:.2f}",
        "Minimum Close Price": "{:.2f}",
        "Maximum Close Price": "{:.2f}",
        "Average Daily Return": "{:.4f}",
        "Volatility (Std of Returns)": "{:.4f}",
        "Average Volume": "{:.0f}",
    }))

    # --------------------------------------------------------
    # RISK–REWARD SCORE
    # --------------------------------------------------------
    st.subheader(f"⚖️ Risk–Reward Score - {selected_coin}")
    score, label = calculate_risk_reward_score(df)

    col1, col2 = st.columns(2)
    with col1:
        if math.isnan(score):
            st.metric("Risk–Reward Score", "N/A")
        else:
            st.metric("Risk–Reward Score", f"{score:.2f}")
    with col2:
        st.write(f"**Category:** {label}")

    # --------------------------------------------------------
    # NEXT-DAY PREDICTION
    # --------------------------------------------------------
    st.subheader(f"🔮 Next-Day Price Prediction - {selected_coin}")
    predicted_price = predict_next_day(df)

    if predicted_price is None:
        st.warning("Not enough data to make a prediction.")
    else:
        last_close = df['Close'].iloc[-1]
        diff = predicted_price - last_close
        diff_pct = (diff / last_close) * 100 if last_close != 0 else 0.0
        st.metric(
            label="Predicted Next Close Price",
            value=f"{predicted_price:.2f}",
            delta=f"{diff:.2f} ({diff_pct:.2f}%)"
        )

    # --------------------------------------------------------
    # PORTFOLIO SIMULATION
    # --------------------------------------------------------
    st.subheader(f"💼 Portfolio Simulation - {selected_coin}")
    sim_result = simulate_portfolio(df, amount=investment_amount)
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Investment", f"{sim_result['Initial Investment']:.2f}")
    col2.metric("Final Value", f"{sim_result['Final Value']:.2f}")
    col3.metric("Profit / Loss", f"{sim_result['Profit']:.2f} ({sim_result['Return %']:.2f}%)")

    # --------------------------------------------------------
    # ALERT SYSTEM
    # --------------------------------------------------------
    st.subheader(f"🚨 Alerts - {selected_coin}")
    alerts = alert_system(df)
    for a in alerts:
        st.write("-", a)

    # --------------------------------------------------------
    # PLOTTING (LAST)
    # --------------------------------------------------------
    st.subheader(f"📉 Charts - {selected_coin}")
    st.caption("Plotting is shown last, as requested.")

    fig_price, fig_volume, fig_returns = plot_graphs(df, coin_name=selected_coin)
    st.pyplot(fig_price)
    st.pyplot(fig_volume)
    st.pyplot(fig_returns)


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    # To run:  streamlit run crypto_app.py
    streamlit_app()