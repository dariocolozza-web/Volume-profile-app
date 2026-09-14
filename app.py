"""
Volume Profile Analyzer - Resonance QuantLab
App Streamlit: l'utente inserisce un ticker e ottiene
il Volume Profile + Rolling POC calcolati al volo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, obbligatorio in ambiente server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import yfinance as yf

# ============================================================
# CONFIGURAZIONE FISSA (invariata rispetto allo script originale)
# ============================================================

YEARS = 2
LOOKBACK = 250
ROWS = 200
ROLLING_POC_WINDOWS = [28, 60, 150, 200, 250]

st.set_page_config(page_title="Volume Profile Analyzer", layout="wide")


# ============================================================
# DOWNLOAD DATI (cache per non martellare Yahoo Finance)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_data(ticker, years=2):
    end_date = datetime.today() + timedelta(days=1)
    start_date = datetime.today() - timedelta(days=365 * years + 10)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None

    df = df[required_cols].copy()
    df = df.dropna(subset=["High", "Low", "Close", "Volume"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ============================================================
# VOLUME PROFILE
# ============================================================

def calculate_volume_profile(df, lookback=250, rows=200):
    data = df.tail(lookback).copy()

    highest_price = data["High"].max()
    lowest_price = data["Low"].min()
    levels = np.linspace(lowest_price, highest_price, rows + 1)

    highs = data["High"].to_numpy()[:, None]
    lows = data["Low"].to_numpy()[:, None]
    vols = data["Volume"].to_numpy()[:, None]

    level_low = levels[:-1][None, :]
    level_high = levels[1:][None, :]

    mask = (highs > level_low) & (lows < level_high)
    volume_by_row = (mask * vols).sum(axis=0)

    profile = pd.DataFrame({
        "price_low": levels[:-1],
        "price_high": levels[1:],
        "price_mid": (levels[:-1] + levels[1:]) / 2,
        "volume": volume_by_row,
    })

    max_volume = profile["volume"].max()
    profile["volume_norm"] = profile["volume"] / max_volume if max_volume > 0 else 0

    poc_row = profile.loc[profile["volume"].idxmax()]

    return {
        "profile": profile,
        "poc": poc_row["price_mid"],
        "poc_volume": poc_row["volume"],
        "highest": highest_price,
        "lowest": lowest_price,
        "data": data,
    }


# ============================================================
# ROLLING POC (vettorizzato: niente doppio loop Python)
# ============================================================

def calculate_rolling_poc_vectorized(df, lookback, rows=200):
    n = len(df)
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    vols = df["Volume"].to_numpy()

    result = np.full(n, np.nan)

    if n < lookback:
        return result

    for i in range(lookback - 1, n):
        h = highs[i + 1 - lookback:i + 1]
        l = lows[i + 1 - lookback:i + 1]
        v = vols[i + 1 - lookback:i + 1]

        highest = h.max()
        lowest = l.min()
        levels = np.linspace(lowest, highest, rows + 1)

        mask = (h[:, None] > levels[:-1][None, :]) & (l[:, None] < levels[1:][None, :])
        volume_by_row = (mask * v[:, None]).sum(axis=0)

        max_idx = int(np.argmax(volume_by_row))
        result[i] = (levels[max_idx] + levels[max_idx + 1]) / 2

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def compute_all_rolling_poc(df, windows, rows=200):
    out = df.copy()
    for w in windows:
        out[f"Rolling_POC_{w}"] = calculate_rolling_poc_vectorized(out, min(w, len(out)), rows)
    return out


# ============================================================
# PLOT VOLUME PROFILE
# ============================================================

def plot_volume_profile(result, ticker):
    profile = result["profile"]
    poc = result["poc"]
    data = result["data"].copy()

    last_close = data["Close"].iloc[-1]
    last_date = data["Date"].iloc[-1]

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(data["Date"], data["Close"], linewidth=1.6, label=f"{ticker} Close")
    ax.set_title(
        f"{ticker} — Prezzo Daily + Volume Profile (ultimi {len(data)} giorni)\n"
        f"Ultima data: {last_date.date()}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Prezzo")

    date_nums = mdates.date2num(data["Date"])
    first_num, last_num = date_nums[0], date_nums[-1]
    chart_days = last_num - first_num

    profile_left = last_num + chart_days * 0.03
    profile_max_width = chart_days * 0.22

    max_volume = profile["volume"].max()
    if max_volume == 0:
        st.error("Volume massimo pari a zero: impossibile costruire il profilo per questo ticker.")
        return None

    bar_lengths = profile["volume_norm"] * profile_max_width

    ax.barh(
        profile["price_mid"], bar_lengths,
        height=(profile["price_high"] - profile["price_low"]) * 0.85,
        left=profile_left, alpha=0.70, edgecolor="black", linewidth=0.15,
        label="Volume Profile",
    )

    ax.axhline(poc, linestyle="--", linewidth=2, label=f"POC: {poc:.2f}")
    ax.axhline(last_close, linestyle="-", linewidth=2, label=f"Last Close: {last_close:.2f}")

    ax.text(profile_left + profile_max_width * 1.02, poc, f"POC {poc:.2f}",
            va="center", fontsize=9, fontweight="bold")
    ax.text(profile_left + profile_max_width * 1.02, last_close, f"Close {last_close:.2f}",
            va="center", fontsize=9, fontweight="bold")

    # Asse Y dinamico: funziona sia su ES (6000 pts) sia su titoli a pochi euro
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.grid(True, which="major", alpha=0.35)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    ax.set_xlim(first_num, profile_left + profile_max_width * 1.35)
    ax.legend(loc="best")
    plt.tight_layout()

    return fig


# ============================================================
# PLOT ROLLING POC
# ============================================================

def plot_price_and_rolling_poc(df, ticker, windows):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df["Date"], df["Close"], linewidth=2.0, label=f"{ticker} Close")

    for w in windows:
        col = f"Rolling_POC_{w}"
        if col in df.columns:
            ax.plot(df["Date"], df[col], linewidth=1.3, linestyle="--", label=f"Rolling POC {w}D")

    ax.set_title(f"{ticker} — Close vs Multi-Horizon Rolling POC", fontsize=13, fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel("Prezzo")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.grid(True, which="major", alpha=0.35)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    ax.legend(loc="best", ncol=2)
    plt.tight_layout()

    return fig


# ============================================================
# NODI DI VOLUME
# ============================================================

def extract_volume_nodes(profile, top_n=10):
    high_nodes = profile.sort_values("volume", ascending=False).head(top_n)
    low_nodes = profile[profile["volume"] > 0].sort_values("volume", ascending=True).head(top_n)
    return high_nodes, low_nodes


# ============================================================
# INTERFACCIA STREAMLIT
# ============================================================

st.title("📊 Volume Profile Analyzer")
st.caption("Resonance QuantLab — inserisci un ticker Yahoo Finance (es. AAPL, RACE, ES=F, BTC-USD)")

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Ticker", value="RACE", label_visibility="collapsed", placeholder="Inserisci il ticker (es. AAPL)")
with col2:
    analizza = st.button("Analizza", type="primary", use_container_width=True)

if analizza:
    if not ticker or not ticker.strip():
        st.warning("Inserisci un ticker valido.")
    else:
        ticker = ticker.strip().upper()

        with st.spinner(f"Scarico i dati per {ticker}..."):
            df = download_data(ticker, YEARS)

        if df is None or df.empty:
            st.error(f"Nessun dato trovato per il ticker '{ticker}'. Verifica che sia corretto su Yahoo Finance.")
        else:
            with st.spinner("Calcolo il Volume Profile..."):
                result = calculate_volume_profile(df, lookback=min(LOOKBACK, len(df)), rows=ROWS)
                high_nodes, low_nodes = extract_volume_nodes(result["profile"], top_n=10)

            with st.spinner("Calcolo il Rolling POC multi-finestra (qualche secondo)..."):
                df_with_poc = compute_all_rolling_poc(df, ROLLING_POC_WINDOWS, rows=ROWS)

            # --- Metriche riassuntive ---
            last_close = result["data"]["Close"].iloc[-1]
            poc = result["poc"]
            distance = last_close - poc
            distance_pct = distance / last_close * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ultimo Close", f"{last_close:.2f}")
            m2.metric("POC", f"{poc:.2f}")
            m3.metric("Distanza da POC", f"{distance:+.2f}", f"{distance_pct:+.2f}%")
            m4.metric("Barre analizzate", f"{len(result['data'])}")

            if last_close > poc:
                st.info("**Lettura:** prezzo sopra il POC — il mercato tratta sopra l'area di massimo consenso del lookback.")
            elif last_close < poc:
                st.info("**Lettura:** prezzo sotto il POC — il mercato tratta sotto l'area di massimo consenso del lookback.")
            else:
                st.info("**Lettura:** prezzo esattamente sul POC.")

            # --- Grafici ---
            st.subheader("Volume Profile")
            fig1 = plot_volume_profile(result, ticker)
            if fig1:
                st.pyplot(fig1)
                plt.close(fig1)

            st.subheader("Rolling POC multi-orizzonte")
            fig2 = plot_price_and_rolling_poc(df_with_poc, ticker, ROLLING_POC_WINDOWS)
            st.pyplot(fig2)
            plt.close(fig2)

            # --- Tabelle nodi di volume ---
            st.subheader("Nodi di Volume")
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**Top High Volume Nodes**")
                st.dataframe(
                    high_nodes[["price_mid", "volume", "volume_norm"]]
                    .rename(columns={"price_mid": "Prezzo", "volume": "Volume stimato", "volume_norm": "Forza relativa"})
                    .style.format({"Prezzo": "{:.2f}", "Volume stimato": "{:,.0f}", "Forza relativa": "{:.2f}"}),
                    hide_index=True, use_container_width=True,
                )
            with t2:
                st.markdown("**Top Low Volume Nodes**")
                st.dataframe(
                    low_nodes[["price_mid", "volume", "volume_norm"]]
                    .rename(columns={"price_mid": "Prezzo", "volume": "Volume stimato", "volume_norm": "Forza relativa"})
                    .style.format({"Prezzo": "{:.2f}", "Volume stimato": "{:,.0f}", "Forza relativa": "{:.2f}"}),
                    hide_index=True, use_container_width=True,
                )

            with st.expander("Note metodologiche"):
                st.markdown(
                    "- Il **POC** (Point of Control) è la fascia di prezzo con il massimo volume stimato.\n"
                    "- Gli **High Volume Nodes** sono aree di accettazione, possibile magnete o stabilizzazione.\n"
                    "- I **Low Volume Nodes** sono aree di minore accettazione, dove il prezzo può muoversi più velocemente.\n"
                    "- Il calcolo assegna il volume di ogni candela a tutte le fasce di prezzo che attraversa (High/Low), "
                    "replicando la logica Pine Script originale: **non** è un vero volume-at-price tick-by-tick, "
                    "ma una stima OHLCV-based."
                )
else:
    st.info("Inserisci un ticker e premi **Analizza** per generare il report.")
