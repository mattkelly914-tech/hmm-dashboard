"""
HMM Market Regime Classification Dashboard
--------------------------------------------
Streamlit app that downloads price data via yfinance, engineers a two-feature
observation matrix (log returns + realized volatility), fits a Gaussian HMM
to decode hidden "market regimes", visualizes the results, and routes a
regime-aware backtester across three sub-strategies.

Run with:
    streamlit run hmm_regime_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# --------------------------------------------------------------------------
# Page config / dark theme styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HMM Market Regime Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background-color: #12161f;
        border-right: 1px solid #262b36;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #d8dee9 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #161b26;
        border: 1px solid #262b36;
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #b8c2d4 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f5f7fa !important;
    }
    .regime-banner {
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        border: 1px solid #262b36;
        margin-top: 10px;
        margin-bottom: 24px;
    }
    .regime-banner h1 {
        font-size: 2.6rem;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .regime-banner p {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-top: 8px;
    }
    .regime-alert {
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid;
        margin-bottom: 16px;
        font-size: 1.02rem;
        animation: pulse-alert 1.8s ease-in-out infinite;
    }
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.72; }
    }
    div[data-testid="stTable"] table {
        color: #e6e6e6;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

# Regime color scheme (state label -> (name, rgba fill, solid hex))
REGIME_STYLE = {
    "Bullish": {"fill": "rgba(46, 204, 113, 0.18)", "solid": "#2ecc71"},
    "Bearish": {"fill": "rgba(231, 76, 60, 0.20)", "solid": "#e74c3c"},
    "Neutral": {"fill": "rgba(149, 165, 166, 0.20)", "solid": "#95a5a6"},
}
# Fixed 0/1/2 slot names, per spec: State 0 = Green, State 1 = Red, State 2 = Gray
SLOT_ORDER = ["Bullish", "Bearish", "Neutral"]

# --------------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker", value="SPY").strip().upper()
lookback_years = st.sidebar.number_input(
    "Lookback Years", min_value=1, max_value=30, value=10, step=1
)
n_iter = st.sidebar.slider("HMM max iterations", 50, 500, 100, step=50)
random_state = st.sidebar.number_input("Random state (seed)", value=42, step=1)
alert_window = st.sidebar.number_input(
    "Regime shift alert window (trading days)",
    min_value=1, max_value=30, value=3, step=1,
    help="Show a shift alert when the current regime started within this many "
         "trading days of the most recent data point.",
)
run_button = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: GaussianHMM(n_components=3, covariance_type='full').\n\n"
    "Observables: daily log returns + 14-day annualized realized volatility."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔀 Regime Router")
strategy_options = ["Trend (SMA 10/50)", "Mean Reversion (RSI-3)", "Cash"]
route_bullish = st.sidebar.selectbox(
    "Bullish state uses", strategy_options, index=0
)
route_bearish = st.sidebar.selectbox(
    "Bearish state uses", strategy_options, index=2
)
route_neutral = st.sidebar.selectbox(
    "Neutral state uses", strategy_options, index=1
)

st.title("📉 HMM Market Regime Classification Dashboard")
st.caption(
    f"Ticker **{ticker}** · Lookback **{lookback_years}y** · "
    "Hidden Markov Model with 3 latent regimes"
)


# --------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_prices(tkr: str, years: int) -> pd.DataFrame:
    period_str = f"{years}y"
    df = yf.download(tkr, period=period_str, auto_adjust=True, progress=False)
    if df.empty:
        return df
    # Flatten yfinance MultiIndex columns if present (single ticker case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def build_observations(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"]
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))
    out["RealizedVol"] = out["LogReturn"].rolling(window=14).std() * np.sqrt(252)
    out = out.dropna()
    return out


@st.cache_resource(show_spinner=False)
def fit_hmm(X: np.ndarray, n_iter: int, seed: int):
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(X)
    states = model.predict(X)
    return model, states


def contiguous_spans(index: pd.DatetimeIndex, states: np.ndarray):
    """Yield (start_date, end_date, state) for contiguous runs of the same state."""
    spans = []
    start_idx = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start_idx]:
            spans.append((index[start_idx], index[i - 1], states[start_idx]))
            start_idx = i
    return spans


# --------------------------------------------------------------------------
# Sub-strategies & regime router (used by the Dynamic Regime Router tab)
# --------------------------------------------------------------------------
def build_strat_trend(price: pd.Series) -> pd.Series:
    """Long when 10-day SMA > 50-day SMA, else cash."""
    sma_fast = price.rolling(10).mean()
    sma_slow = price.rolling(50).mean()
    return (sma_fast > sma_slow).astype(int)


def build_strat_reversion(price: pd.Series, period: int = 3) -> pd.Series:
    """Stateful mean-reversion signal: go long when 3-period RSI drops below
    30, hold the long until RSI crosses above 70 (then exit to cash).
    Implemented vectorized via a forward-filled state series.
    """
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral during warmup

    raw_state = pd.Series(np.nan, index=price.index)
    raw_state[rsi < 30] = 1  # entry trigger
    raw_state[rsi > 70] = 0  # exit trigger

    return raw_state.ffill().fillna(0).astype(int)


def build_strat_cash(index: pd.Index) -> pd.Series:
    """Always flat / in cash."""
    return pd.Series(0, index=index)


def route_composite_signal(df: pd.DataFrame, route_map: dict) -> pd.DataFrame:
    """Adds strat_trend, strat_reversion, strat_cash, and composite_signal
    columns, routing each day's signal by that day's hmm_state per route_map
    (e.g. {0: 'trend', 1: 'cash', 2: 'reversion'}).
    """
    df = df.copy()
    price = df["Close"]

    df["strat_trend"] = build_strat_trend(price)
    df["strat_reversion"] = build_strat_reversion(price)
    df["strat_cash"] = build_strat_cash(df.index)

    strat_col_for = {
        "trend": "strat_trend",
        "reversion": "strat_reversion",
        "cash": "strat_cash",
    }

    conditions = [df["hmm_state"] == state for state in route_map.keys()]
    choices = [df[strat_col_for[route_map[state]]] for state in route_map.keys()]
    df["composite_signal"] = np.select(conditions, choices, default=0)

    # Shift forward 1 day to avoid look-ahead bias
    df["strategy_returns"] = (
        df["composite_signal"].shift(1).fillna(0) * df["log_returns"]
    )
    return df


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_periods = len(equity_curve)
    return total_return ** (periods_per_year / n_periods) - 1


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for label, ret_col, equity_col in [
        ("Dynamic Regime Router", "strategy_returns", "equity_router"),
        ("Buy & Hold", "log_returns", "equity_bh"),
    ]:
        equity = df[equity_col]
        rets = df[ret_col]
        rows[label] = {
            "Total Return": f"{(equity.iloc[-1] - 1) * 100:.2f}%",
            "CAGR": f"{annualized_return(equity) * 100:.2f}%",
            "Max Drawdown": f"{max_drawdown(equity) * 100:.2f}%",
            "Ann. Volatility": f"{rets.std() * np.sqrt(252) * 100:.2f}%",
            "Sharpe Ratio": f"{annualized_sharpe(rets):.2f}",
        }
    return pd.DataFrame(rows).T


def plot_equity_curves(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity_router"],
        name="Dynamic Regime Router",
        line=dict(color="#00CC96", width=2.2),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity_bh"],
        name="Buy & Hold",
        line=dict(color="#7f8c9a", width=1.6, dash="dot"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Dynamic Regime Router vs. Buy & Hold — Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=560,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(
            orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
            font=dict(size=13, color="#e6e6e6"),
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=70),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


# --------------------------------------------------------------------------
# Main flow
# --------------------------------------------------------------------------
if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

with st.spinner(f"Downloading {ticker} price history..."):
    raw = download_prices(ticker, lookback_years)

if raw.empty or "Close" not in raw.columns:
    st.error(
        f"No data returned for '{ticker}'. Check the symbol or try a different "
        "lookback period."
    )
    st.stop()

data = build_observations(raw)

if len(data) < 50:
    st.error("Not enough data after processing to fit a reliable HMM. Try a longer lookback.")
    st.stop()

# 2D observation matrix: [log returns, realized volatility]
X = np.column_stack([data["LogReturn"].values, data["RealizedVol"].values])

with st.spinner("Fitting Gaussian HMM..."):
    model, raw_states = fit_hmm(X, n_iter, random_state)

data["RawState"] = raw_states

# --------------------------------------------------------------------------
# Re-map arbitrary HMM component indices -> meaningful slots 0/1/2
# so that: slot 0 = highest mean return (Bullish/Green),
#          slot 1 = lowest mean return (Bearish/Red),
#          slot 2 = the remaining, middle state (Neutral/Gray)
# --------------------------------------------------------------------------
state_means = (
    data.groupby("RawState")["LogReturn"].mean().sort_values(ascending=False)
)
raw_order = state_means.index.tolist()  # [best, ..., worst] by mean return

while len(raw_order) < 3:
    raw_order.append(-1)

bullish_raw, mid_raw, bearish_raw = raw_order[0], raw_order[1], raw_order[2]

remap = {}
if bullish_raw != -1:
    remap[bullish_raw] = 0  # Bullish -> slot 0 -> Green
if bearish_raw != -1:
    remap[bearish_raw] = 1  # Bearish -> slot 1 -> Red
if mid_raw is not None and mid_raw != -1:
    remap[mid_raw] = 2  # Neutral -> slot 2 -> Gray

data["State"] = data["RawState"].map(remap)
slot_to_name = {0: "Bullish", 1: "Bearish", 2: "Neutral"}
data["RegimeName"] = data["State"].map(slot_to_name)

# --------------------------------------------------------------------------
# Tabs: Read Me | Regime Classifier | Dynamic Regime Router
# --------------------------------------------------------------------------
tab_readme, tab_classifier, tab_router = st.tabs(
    ["📖 Read Me", "🔍 Regime Classifier", "🔀 Dynamic Regime Router"]
)

# ============================================================================
# TAB 0 — Read Me (plain-language explainer for non-finance/non-math users)
# ============================================================================
with tab_readme:
    st.markdown(
        """
## What is this dashboard?

This tool looks at a stock or fund's price history and tries to answer a
simple question: **"is the market currently calm and rising, calm and
falling, or choppy and uncertain?"** It labels every day in the past with
one of three moods, and then uses that labeling to test a simple trading
idea. Nothing here is financial advice — it's a way to explore an idea
using historical data.

---

## The three "moods" (regimes)

| Label | What it means in plain English |
|---|---|
| 🟢 **Bullish** | Prices have generally been climbing, and the ride has been relatively smooth. |
| 🔴 **Bearish** | Prices have generally been falling, often with sharper, scarier swings. |
| ⚪ **Neutral** | No strong trend either way — prices are drifting sideways or chopping around. |

A "regime" is just a stretch of time that shares one of these moods. The
market doesn't wear a label — this tool is making its best statistical
guess based on patterns in the price data.

---

## How does it decide the mood? (Hidden Markov Model, in plain terms)

The dashboard uses something called a **Hidden Markov Model**, or **HMM**.
That name sounds intimidating, but the idea is simple:

- Imagine the market is always secretly "in" one of the three moods above,
  but you can't see the mood directly — you can only see the day-to-day
  price movements it produces. The mood is **hidden**.
- The model looks at two things about each day: how much the price moved
  (**return**) and how jumpy or calm the recent price action has been
  (**volatility** — see definitions below).
- Based on patterns in those two numbers across the whole price history,
  the model groups days into three clusters, then ranks them by their
  average return: the cluster with the best average return becomes
  "Bullish," the worst becomes "Bearish," and the one left over becomes
  "Neutral."
- It also learns how likely the market is to *stay* in a mood versus
  *switch* to a different one from one day to the next — this is the
  "Markov" part: tomorrow's mood depends only on today's mood, not on
  the entire history before it.

The model isn't told in advance what "Bullish" or "Bearish" should look
like — it finds the groupings on its own, and this dashboard just applies
sensible labels afterward based on which group did best and worst.

---

## Key terms, defined simply

- **Log return** — a way of measuring how much a price moved from one day
  to the next, expressed as a percentage-like number. You can think of it
  as "daily percent change," calculated in a way that adds up cleanly
  over many days.
- **Realized volatility** — a measure of how much the price has been
  bouncing around recently (over the last 14 trading days here), scaled
  up to represent a full year. Higher volatility means bigger, more
  unpredictable swings — calm markets have low volatility, turbulent ones
  have high volatility.
- **Trading day** — a day the stock market is open (weekdays, excluding
  holidays).
- **SMA (Simple Moving Average)** — the average price over the last *N*
  days. A "10-day SMA" is just the average closing price over the past 10
  trading days. Comparing a short SMA to a longer one is a common way to
  spot whether a trend is forming.
- **RSI (Relative Strength Index)** — a number between 0 and 100 that
  measures whether a stock has recently been bought or sold more
  aggressively than usual. Very low RSI (under 30) is often read as
  "oversold" (potentially due for a bounce); very high RSI (over 70) is
  read as "overbought."
- **Equity curve** — a line chart showing how \$1 invested at the start
  would have grown (or shrunk) over time, if you'd followed a given
  strategy.
- **Total Return** — the overall percentage gain or loss from the very
  first day to the very last day.
- **CAGR (Compound Annual Growth Rate)** — the Total Return expressed as
  a smooth "per year" growth rate, so strategies over different time
  periods can be compared fairly.
- **Max Drawdown** — the single worst peak-to-valley decline the strategy
  experienced. If your equity curve climbed to \$2, then fell to \$1.50
  before recovering, that's a 25% drawdown. Smaller (less negative) is
  better — it's a measure of "how bad did it get at the worst point."
- **Annualized Volatility** — how bumpy the strategy's returns were,
  scaled to a yearly figure. Lower usually means a smoother, less
  stressful ride.
- **Sharpe Ratio** — a single number that roughly answers "how much
  return did this strategy generate for the amount of bumpiness
  (risk) it took on?" Higher is generally better; it lets you compare a
  smooth, modest strategy against a wild, higher-returning one on equal
  footing.

---

## What the "Regime Classifier" tab shows

That tab colors the price chart by which mood the model thinks the market
was in on each day (green = Bullish, red = Bearish, gray = Neutral), shows
summary statistics for each mood, and displays a banner with today's
current predicted mood. It also has an alert that lights up when the mood
has *just* changed recently, so you can see regime shifts as they happen.

## What the "Dynamic Regime Router" tab shows

This tab tests a simple idea: **what if you used a different trading
strategy depending on the market's current mood?**

- In a 🟢 Bullish mood, follow the trend (the "SMA 10/50" strategy: stay
  invested when the short-term average price is above the long-term
  average).
- In a 🔴 Bearish mood, sit in cash to avoid the worst of the damage.
- In a ⚪ Neutral mood, try to buy dips and sell bounces (the "RSI"
  strategy).

It then compares how that "smart switching" approach would have performed
against simply buying and holding the whole time, using the metrics
defined above (Total Return, CAGR, Max Drawdown, Sharpe Ratio). The
comparison is run on **historical data only** — it does not predict the
future, and past results are not a guarantee of what will happen next.

---

## A few honest caveats

- This is a simplified educational model, not a professional trading
  system. Real-world trading involves costs, taxes, slippage, and risks
  that aren't captured here.
- The model's "Bullish/Bearish/Neutral" labels are relative to the ticker
  and time period you choose — they can shift if you change the ticker,
  lookback period, or random seed.
- Nothing on this page is investment advice. It's a tool for exploring
  how a statistical idea behaves on historical prices.
"""
    )

# ============================================================================
# TAB 1 — Regime Classifier (original dashboard)
# ============================================================================
with tab_classifier:
    st.subheader("Regime Statistics")
    cols = st.columns(3)
    for slot in [0, 1, 2]:
        name = slot_to_name[slot]
        subset = data[data["State"] == slot]
        mean_ret = subset["LogReturn"].mean() if len(subset) else np.nan
        mean_vol = subset["RealizedVol"].mean() if len(subset) else np.nan
        n_days = len(subset)
        with cols[slot]:
            st.markdown(f"**State {slot} — {name}**")
            st.metric(
                label="Mean Daily Log Return",
                value=f"{mean_ret:.4%}" if pd.notna(mean_ret) else "N/A",
            )
            st.metric(
                label="Mean Realized Volatility (ann.)",
                value=f"{mean_vol:.2%}" if pd.notna(mean_vol) else "N/A",
            )
            st.caption(f"{n_days} trading days classified in this state")

    st.markdown("---")

    st.subheader(f"{ticker} Price with HMM Regime Overlay")

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=f"{ticker} Close",
            line=dict(color="#f5f5f5", width=1.6),
            customdata=data["RegimeName"],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Price: %{y:.2f}"
                "<br>Regime: <b>%{customdata}</b><extra></extra>"
            ),
        )
    )

    spans = contiguous_spans(data.index, data["State"].values)
    shapes = []
    for start, end, slot in spans:
        name = slot_to_name.get(slot, "Unknown")
        style = REGIME_STYLE.get(name, {"fill": "rgba(150,150,150,0.15)"})
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=style["fill"],
            line_width=0,
            layer="below",
        ))
    fig_price.update_layout(shapes=shapes)

    # Dummy traces so a regime legend shows up (vrects don't appear in legend)
    for slot in [0, 1, 2]:
        name = slot_to_name[slot]
        style = REGIME_STYLE[name]
        fig_price.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=style["solid"]),
                name=f"State {slot} — {name}",
            )
        )

    fig_price.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=13, color="#e6e6e6"),
        ),
        hovermode="x unified",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )

    st.plotly_chart(fig_price, use_container_width=True)
    st.caption(
        "The background shading marks which regime the model assigned to "
        "each period — colors match the legend dots above (green = "
        "Bullish, red = Bearish, gray = Neutral). Hover over the price "
        "line to see the exact regime for any date."
    )

    # --------------------------------------------------------------------------
    # Regime-shift alert — fires when the current regime began recently
    # --------------------------------------------------------------------------
    current_span_start, current_span_end, current_span_slot = spans[-1]
    days_in_current_regime = len(data.loc[current_span_start:current_span_end])

    if len(spans) > 1 and days_in_current_regime <= alert_window:
        prev_slot = spans[-2][2]
        prev_name = slot_to_name.get(prev_slot, "Unknown")
        curr_name = slot_to_name.get(current_span_slot, "Unknown")
        alert_style = REGIME_STYLE.get(curr_name, {"solid": "#e6e6e6"})
        shift_date = current_span_start.strftime("%Y-%m-%d")
        day_word = "day" if days_in_current_regime == 1 else "days"
        st.markdown(
            f"""
            <div class="regime-alert" style="border-color:{alert_style['solid']}; color:{alert_style['solid']};">
                🚨 <b>REGIME SHIFT DETECTED</b> — market flipped from
                <b>{prev_name}</b> to <b>{curr_name}</b> on <b>{shift_date}</b>
                ({days_in_current_regime} trading {day_word} ago).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Current regime banner
    current_slot = int(data["State"].iloc[-1])
    current_name = slot_to_name[current_slot]
    current_date = data.index[-1].strftime("%Y-%m-%d")
    current_ret = data["LogReturn"].iloc[-1]
    current_vol = data["RealizedVol"].iloc[-1]
    style = REGIME_STYLE[current_name]

    banner_html = f"""
    <div class="regime-banner" style="background-color:{style['fill']}; border-color:{style['solid']};">
        <p style="text-transform:uppercase; letter-spacing:2px; opacity:0.7;">Current Predicted Market Regime</p>
        <h1 style="color:{style['solid']};">State {current_slot} — {current_name.upper()}</h1>
        <p>
            As of <b>{current_date}</b> · Daily Log Return: <b>{current_ret:.4%}</b> ·
            Realized Volatility (ann.): <b>{current_vol:.2%}</b>
        </p>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    with st.expander("Show raw HMM model details"):
        st.write("Transition matrix (rows/cols follow original HMM component order):")
        st.dataframe(pd.DataFrame(model.transmat_))
        st.write("Means per raw component [LogReturn, RealizedVol]:")
        st.dataframe(pd.DataFrame(model.means_, columns=["LogReturn", "RealizedVol"]))
        st.caption(
            "Raw HMM component indices were re-mapped to slots 0/1/2 by descending "
            "mean return (0=Bullish/Green, 1=Bearish/Red, 2=Neutral/Gray) for a "
            "consistent, interpretable color scheme."
        )

# ============================================================================
# TAB 2 — Dynamic Regime Router (backtester)
# ============================================================================
with tab_router:
    st.subheader("Dynamic Regime-Routing Backtester")
    st.caption(
        "Each day's trading signal is picked from one of three sub-strategies "
        "based on that day's HMM-classified regime, then compared against a "
        "simple Buy & Hold benchmark."
    )

    label_to_key = {
        "Trend (SMA 10/50)": "trend",
        "Mean Reversion (RSI-3)": "reversion",
        "Cash": "cash",
    }
    route_map = {
        0: label_to_key[route_bullish],   # Bullish
        1: label_to_key[route_bearish],   # Bearish
        2: label_to_key[route_neutral],   # Neutral
    }

    router_df = data.rename(columns={"LogReturn": "log_returns", "State": "hmm_state"})
    router_df = route_composite_signal(router_df, route_map)

    router_df["equity_router"] = np.exp(router_df["strategy_returns"].cumsum())
    router_df["equity_bh"] = np.exp(router_df["log_returns"].cumsum())

    with st.expander("Current regime → strategy routing", expanded=True):
        route_cols = st.columns(3)
        for slot, col in zip([0, 1, 2], route_cols):
            name = slot_to_name[slot]
            strat_label = {"trend": "Trend (SMA 10/50)", "reversion": "Mean Reversion (RSI-3)", "cash": "Cash"}[route_map[slot]]
            with col:
                st.markdown(f"**State {slot} — {name}**")
                st.markdown(f"→ {strat_label}")

    fig_equity = plot_equity_curves(router_df)
    st.plotly_chart(fig_equity, use_container_width=True)

    st.subheader("Risk / Return Summary")
    summary = build_summary_table(router_df)
    st.dataframe(summary, use_container_width=True)

    st.caption(
        "Signals are shifted forward one day before being applied to returns, "
        "so today's regime and indicator values only affect tomorrow's position "
        "(no look-ahead bias)."
    )
