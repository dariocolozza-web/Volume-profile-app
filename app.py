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
from scipy.signal import find_peaks
import yfinance as yf

# ============================================================
# CONFIGURAZIONE FISSA (invariata rispetto allo script originale)
# ============================================================

YEARS = 2
LOOKBACK_DEFAULT = 250
LOOKBACK_MIN = 20
LOOKBACK_MAX = 750
ROWS = 200
ROLLING_POC_WINDOWS = [28, 60, 150, 200, 250]

# Parametri rilevamento value area multiple
VA_PROMINENCE = 0.05     # quanto deve sporgere un picco per contare come cluster
VA_SMOOTH_WINDOW = 5     # ampiezza media mobile sul profilo
VA_LOWER_PCT = 0.10      # coda inferiore di ogni cluster
VA_UPPER_PCT = 0.90      # coda superiore di ogni cluster

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
# RILEVAMENTO VALUE AREA MULTIPLE (cluster di volume)
# ============================================================

def smooth_profile(volumes, window=5):
    """Media mobile centrata per ridurre il rumore prima della ricerca picchi."""
    if window <= 1:
        return volumes.copy()
    kernel = np.ones(window) / window
    return np.convolve(volumes, kernel, mode="same")


def detect_value_areas(profile, prominence_pct=VA_PROMINENCE,
                       smooth_window=VA_SMOOTH_WINDOW,
                       min_distance_pct=0.03,
                       lower_pct=VA_LOWER_PCT, upper_pct=VA_UPPER_PCT):
    """
    Identifica i cluster di volume (value area multiple) nel profilo.

    Logica:
      1. smoothing del profilo
      2. ricerca dei picchi locali con prominenza minima
      3. i confini fra cluster sono le valli (minimi) fra picchi consecutivi
      4. per ogni cluster le code sono i percentili del volume cumulato interno

    Ritorna una lista di dict, uno per cluster, ordinati per prezzo crescente.
    """
    volumes = profile["volume"].to_numpy(dtype=float)
    prices = profile["price_mid"].to_numpy(dtype=float)
    n = len(volumes)

    if n == 0 or volumes.max() <= 0:
        return []

    smoothed = smooth_profile(volumes, smooth_window)
    max_vol = smoothed.max()
    min_distance = max(1, int(n * min_distance_pct))

    peaks, _ = find_peaks(
        smoothed,
        prominence=prominence_pct * max_vol,
        distance=min_distance,
    )

    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(smoothed))])

    # Confini dei cluster: valli fra picchi consecutivi
    boundaries = [0]
    for i in range(len(peaks) - 1):
        seg_start, seg_end = peaks[i], peaks[i + 1]
        valley_idx = seg_start + int(np.argmin(smoothed[seg_start:seg_end + 1]))
        boundaries.append(valley_idx)
    boundaries.append(n - 1)

    clusters = []
    total_all = volumes.sum()

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue

        seg_vol = volumes[start:end + 1]
        seg_prices = prices[start:end + 1]
        total = seg_vol.sum()
        if total <= 0:
            continue

        cum = np.cumsum(seg_vol) / total
        low_idx = min(int(np.searchsorted(cum, lower_pct)), len(seg_prices) - 1)
        high_idx = min(int(np.searchsorted(cum, upper_pct)), len(seg_prices) - 1)
        peak_local = int(np.argmax(seg_vol))

        clusters.append({
            "price_min": float(seg_prices[0]),
            "price_max": float(seg_prices[-1]),
            "va_low": float(seg_prices[low_idx]),
            "va_high": float(seg_prices[high_idx]),
            "poc": float(seg_prices[peak_local]),
            "volume_total": float(total),
            "volume_share": float(total / total_all) if total_all > 0 else 0.0,
        })

    return clusters


# ============================================================
# PLOT VOLUME PROFILE
# ============================================================

def plot_volume_profile(result, ticker, clusters=None):
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

    # --- Value area multiple: bande + linee sulle code ---
    if clusters:
        palette = ["#e07a5f", "#3d9970", "#9b5de5", "#f2a65a", "#2a9d8f", "#c1121f"]
        for i, c in enumerate(clusters):
            col = palette[i % len(palette)]

            # Banda che copre la value area del cluster (dal 10% al 90%)
            ax.axhspan(c["va_low"], c["va_high"], color=col, alpha=0.10, zorder=0)

            # Linee sulle code: sono i livelli oltre i quali si lascia il cluster
            for level in (c["va_low"], c["va_high"]):
                ax.axhline(level, color=col, linestyle=":", linewidth=1.3,
                           alpha=0.85, zorder=1)

            # Etichetta a sinistra, dentro l'area del grafico prezzo
            ax.text(
                first_num + chart_days * 0.005,
                (c["va_low"] + c["va_high"]) / 2,
                f"VA{i+1}  {c['va_low']:.2f}–{c['va_high']:.2f}  ({c['volume_share']*100:.0f}%)",
                va="center", ha="left", fontsize=8, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.75, lw=0.8),
                zorder=5,
            )

    ax.axhline(poc, linestyle="--", linewidth=2, label=f"POC: {poc:.2f}")
    ax.axhline(last_close, linestyle="-", linewidth=2, label=f"Last Close: {last_close:.2f}")

    if clusters:
        ax.plot([], [], color="#e07a5f", linestyle=":", linewidth=1.3,
                label=f"Value Area ({int(VA_LOWER_PCT*100)}%–{int(VA_UPPER_PCT*100)}%)")

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
st.caption("Resonance QuantLab — inserisci un ticker Yahoo Finance (es. AAPL, NVDA, SPY, ES=F, BTC-USD)")

col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    ticker = st.text_input("Ticker", value="SPY", label_visibility="collapsed", placeholder="Inserisci il ticker (es. AAPL)")
with col2:
    lookback = st.slider(
        "Lookback", LOOKBACK_MIN, LOOKBACK_MAX, LOOKBACK_DEFAULT, 10,
        label_visibility="collapsed",
        help="Numero di barre daily usate per costruire il Volume Profile. "
             "Più lungo = quadro strutturale; più corto = fotografia recente.",
    )
with col3:
    analizza = st.button("Analizza", type="primary", use_container_width=True)

st.caption(f"Lookback: **{lookback}** barre daily (≈ {lookback/21:.0f} mesi di contrattazioni)")

with st.expander("Impostazioni rilevamento Value Area"):
    c1, c2 = st.columns(2)
    with c1:
        va_prominence = st.slider(
            "Sensibilità (prominenza minima)", 0.02, 0.25, VA_PROMINENCE, 0.01,
            help="Più basso = rileva più value area, anche piccole. "
                 "Più alto = solo i cluster più marcati.",
        )
    with c2:
        va_smooth = st.slider(
            "Smoothing del profilo", 1, 15, VA_SMOOTH_WINDOW, 2,
            help="Attenua il rumore prima di cercare i picchi. "
                 "Valori alti fondono cluster vicini.",
        )

if analizza:
    if not ticker or not ticker.strip():
        st.warning("Inserisci un ticker valido.")
    else:
        ticker = ticker.strip().upper()

        # Scarico abbastanza storico da coprire il lookback richiesto
        # (~252 barre per anno) più un margine per il Rolling POC
        years_needed = max(YEARS, int(np.ceil(lookback / 252)) + 1)

        with st.spinner(f"Scarico i dati per {ticker}..."):
            df = download_data(ticker, years_needed)

        if df is None or df.empty:
            st.error(f"Nessun dato trovato per il ticker '{ticker}'. Verifica che sia corretto su Yahoo Finance.")
        else:
            effective_lookback = min(lookback, len(df))
            if effective_lookback < lookback:
                st.warning(
                    f"Storico disponibile per {ticker}: **{len(df)} barre**, "
                    f"meno delle {lookback} richieste. Uso tutte le barre disponibili."
                )

            with st.spinner("Calcolo il Volume Profile..."):
                result = calculate_volume_profile(df, lookback=effective_lookback, rows=ROWS)
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
            clusters = detect_value_areas(
                result["profile"],
                prominence_pct=va_prominence,
                smooth_window=va_smooth,
            )

            st.subheader("Volume Profile")
            fig1 = plot_volume_profile(result, ticker, clusters=clusters)
            if fig1:
                st.pyplot(fig1)
                plt.close(fig1)

            # --- Tabella value area ---
            if clusters:
                st.subheader(f"Value Area rilevate ({len(clusters)})")
                va_df = pd.DataFrame([{
                    "Value Area": f"VA{i+1}",
                    "Coda inferiore": c["va_low"],
                    "POC locale": c["poc"],
                    "Coda superiore": c["va_high"],
                    "Ampiezza": c["va_high"] - c["va_low"],
                    "Quota volume": c["volume_share"],
                    "Contiene il prezzo": "●" if c["va_low"] <= last_close <= c["va_high"] else "",
                } for i, c in enumerate(clusters)])

                st.dataframe(
                    va_df.style.format({
                        "Coda inferiore": "{:.2f}", "POC locale": "{:.2f}",
                        "Coda superiore": "{:.2f}", "Ampiezza": "{:.2f}",
                        "Quota volume": "{:.1%}",
                    }),
                    hide_index=True, use_container_width=True,
                )

                # Dove si trova il prezzo rispetto ai cluster
                inside = [i + 1 for i, c in enumerate(clusters)
                          if c["va_low"] <= last_close <= c["va_high"]]
                if inside:
                    st.success(f"Il prezzo si trova **dentro VA{inside[0]}**. "
                               f"Le code di questa area sono i livelli oltre i quali "
                               f"il mercato transita verso un'altra zona di accettazione.")
                else:
                    st.warning("Il prezzo si trova **in una zona di transito** "
                               "(fra due value area), dove il volume è rado e "
                               "i movimenti tendono a essere più rapidi.")

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
                    "ma una stima OHLCV-based.\n"
                    "\n**Value Area multiple**\n"
                    "- Il profilo viene trattato come distribuzione **multimodale**: più cluster di accettazione "
                    "separati da valli di basso volume.\n"
                    "- I cluster sono individuati cercando i picchi locali con una prominenza minima; "
                    "i confini sono i minimi fra picchi consecutivi.\n"
                    f"- Per ogni cluster le **code** sono i livelli al {int(VA_LOWER_PCT*100)}% e "
                    f"{int(VA_UPPER_PCT*100)}% del volume cumulato *interno al cluster* "
                    f"(quindi contengono l'{int((VA_UPPER_PCT-VA_LOWER_PCT)*100)}% del suo volume).\n"
                    "- Superata una coda, il prezzo lascia quella zona di accettazione e transita "
                    "verso la value area adiacente.\n"
                    "- I percentili sono calcolati sul volume cumulato reale, **non** con un fit gaussiano: "
                    "i cluster reali sono spesso asimmetrici e un fit imporrebbe una forma che i dati non hanno."
                )
else:
    st.info("Inserisci un ticker e premi **Analizza** per generare il report.")
