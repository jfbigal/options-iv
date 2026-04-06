"""
Options IV Surface Dashboard
────────────────────────────
Tabs:
  1. Option Chain  — live snapshot with bid/ask, spread, IV
  2. Vol Smile     — SVI / SABR / Quadratic fit + residuals
  3. AR(1) Signals — mean-reversion of IV deviations from the quadratic smile

All data is synthetically generated; no real market feed required.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from io import StringIO
from core.bs import bs_price, bs_vega, bs_delta, bs_gamma, bs_theta, implied_vol, prob_itm
from core.models import (
    svi_iv, fit_svi,
    sabr_iv_hagan, fit_sabr,
    quad_iv, fit_quadratic, fit_ar1,
)
from core.data_gen import generate_chain, generate_history, SABR_PARAMS

# ══════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Options · IV Surface",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# Custom CSS  — dark, editorial, monospaced numbers
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d0f;
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #222;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #222;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 20px;
    color: #555;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #16161a;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #555;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    color: #00d4ff;
}

/* Data frames */
.stDataFrame { border: 1px solid #222; border-radius: 4px; }

/* Headers */
h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
     letter-spacing: -0.02em; color: #fff; margin-bottom: 0; }
h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #ccc; }

/* Selectbox / number input */
[data-baseweb="select"], [data-baseweb="input"] {
    background: #16161a !important;
    border-color: #333 !important;
}

/* Caption */
small, .stCaption { color: #444 !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; }

/* Tag pill */
.tag {
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.06em; margin-right: 4px;
}
.tag-blue  { background: rgba(0,212,255,0.12); color: #00d4ff; border: 1px solid rgba(0,212,255,0.25); }
.tag-green { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.tag-red   { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.tag-gray  { background: rgba(100,100,100,0.15); color: #888; border: 1px solid #333; }

/* Divider */
hr { border-color: #222; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Plotly theme
# ══════════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d0d0f",
    plot_bgcolor="#0d0d0f",
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#aaa"),
    xaxis=dict(gridcolor="#1e1e22", zeroline=False),
    yaxis=dict(gridcolor="#1e1e22", zeroline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333", borderwidth=1),
)

CYAN   = "#00d4ff"
GREEN  = "#22c55e"
RED    = "#ef4444"
AMBER  = "#f59e0b"
PURPLE = "#a78bfa"

# ══════════════════════════════════════════════════════════════════
# Session state helpers
# ══════════════════════════════════════════════════════════════════
def _init():
    st.session_state.setdefault("chain_df",   None)
    st.session_state.setdefault("history_df", None)
    st.session_state.setdefault("r",   0.20)
    st.session_state.setdefault("q",   0.002)
    st.session_state.setdefault("spot", 1050.0)

_init()

# ══════════════════════════════════════════════════════════════════
# Data loading (cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=300)
def _load_chain(spot, r, q):
    return generate_chain(spot=spot, r=r, q=q)

@st.cache_data(show_spinner=False, ttl=600)
def _load_history():
    return generate_history()

# ══════════════════════════════════════════════════════════════════
# Prep: compute IV + Greeks on the chain dataframe
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _prep_chain(spot, r, q):
    df = _load_chain(spot, r, q)
    rows = []
    for _, row in df.iterrows():
        S   = float(row["spot_mid"])
        K   = float(row["strike"])
        T   = float(row["T_true"])
        cp  = str(row["cp"])
        px  = float(row["last"])

        iv = implied_vol(px, S, K, T, r, q, cp)
        if not (np.isfinite(iv) and iv > 0):
            continue

        F  = S * np.exp((r - q) * T)
        k  = np.log(K / F)
        vg = bs_vega(S, K, T, r, q, iv)
        dl = bs_delta(S, K, T, r, q, iv, cp)
        gm = bs_gamma(S, K, T, r, q, iv)
        th = bs_theta(S, K, T, r, q, iv, cp)
        pi = prob_itm(S, K, T, r, q, iv, cp)

        row = row.copy()
        row["iv"]       = iv
        row["k"]        = k
        row["F"]        = F
        row["vega"]     = vg
        row["delta"]    = dl
        row["gamma"]    = gm
        row["theta"]    = th
        row["prob_itm"] = pi
        row["mny"]      = K / S
        row["days"]     = T * 365
        row["spread_pct"] = (float(row["ask"]) - float(row["bid"])) / max(float(row["mid"]), 0.01) * 100
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙  Parameters")
    spot_in = st.number_input("Spot S₀", value=1050.0, step=10.0, format="%.2f")
    r_in    = st.number_input("Risk-free rate r", value=0.20, step=0.01, format="%.4f")
    q_in    = st.number_input("Dividend yield q", value=0.002, step=0.001, format="%.4f")

    st.markdown("---")
    st.markdown("### 🔬 Liquidity filters")
    min_vol_in      = st.number_input("Min volume", value=0, step=1)
    max_spread_in   = st.number_input("Max spread %", value=40.0, step=1.0)
    mny_range       = st.slider("Moneyness range (K/S)", 0.5, 1.5, (0.70, 1.30), 0.01)

    st.markdown("---")
    st.markdown(
        "<small>Synthetic data — GGAL-like Argentine equity option chain. "
        "No real market connection.</small>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<small>📦 [Source code](https://github.com) · Built with Streamlit</small>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════
with st.spinner("Generating synthetic chain…"):
    df_full = _prep_chain(float(spot_in), float(r_in), float(q_in))

# Apply filters
df = df_full.copy()
if min_vol_in > 0:
    df = df[df["volume"] >= int(min_vol_in)]
df = df[df["spread_pct"] <= float(max_spread_in)]
df = df[(df["mny"] >= mny_range[0]) & (df["mny"] <= mny_range[1])]
df = df[np.isfinite(df["iv"]) & (df["iv"] > 0)]

expiries = sorted(df["expiration"].dropna().unique())

# ══════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📐 Options · IV Surface")
st.markdown(
    f'<span class="tag tag-blue">GGAL synthetic</span>'
    f'<span class="tag tag-gray">r = {r_in:.2%}</span>'
    f'<span class="tag tag-gray">q = {q_in:.3%}</span>'
    f'<span class="tag tag-green">{len(df)} contracts</span>',
    unsafe_allow_html=True
)
st.markdown("")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Spot", f"{spot_in:,.2f}")
m2.metric("Expiries", len(expiries))
m3.metric("Strikes", df["strike"].nunique())
m4.metric("ATM IV (21d C)", f"{df[(df['days']<30) & (df['cp']=='C') & (df['mny'].between(0.99,1.01))]['iv'].mean():.1%}" if len(df) else "—")
m5.metric("ATM IV (49d C)", f"{df[(df['days']>30) & (df['cp']=='C') & (df['mny'].between(0.99,1.01))]['iv'].mean():.1%}" if len(df) else "—")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════
tab_chain, tab_smile, tab_ar1 = st.tabs([
    "01 · Option Chain",
    "02 · Vol Smile & Models",
    "03 · AR(1) Mean-Reversion",
])

# ───────────────────────────────────────────────────────────────────
# TAB 1 — OPTION CHAIN
# ───────────────────────────────────────────────────────────────────
with tab_chain:
    st.markdown("#### Option Chain")

    cc1, cc2, cc3 = st.columns(3)
    exp_sel_c  = cc1.selectbox("Expiry", expiries, key="chain_exp",
                                format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    cp_sel_c   = cc2.selectbox("Side", ["Both", "C", "P"], key="chain_cp")
    show_cols  = cc3.multiselect(
        "Columns",
        ["symbol","cp","strike","mny","days","bid","ask","mid","last","spread_pct",
         "volume","iv","vega","delta","gamma","theta","prob_itm"],
        default=["symbol","cp","strike","mny","days","bid","ask","spread_pct","volume","iv","vega","delta"],
    )

    ch = df.copy()
    ch = ch[pd.to_datetime(ch["expiration"]) == pd.to_datetime(exp_sel_c)]
    if cp_sel_c != "Both":
        ch = ch[ch["cp"] == cp_sel_c]
    ch = ch.sort_values(["cp", "strike"])

    # ── Mirror chart: calls above 0, puts below 0 ──
    fig_chain = go.Figure()
    calls = ch[ch["cp"] == "C"]
    puts  = ch[ch["cp"] == "P"]

    fig_chain.add_trace(go.Bar(
        x=calls["strike"], y=calls["iv"],
        name="Calls IV", marker_color=GREEN, opacity=0.8,
    ))
    fig_chain.add_trace(go.Bar(
        x=puts["strike"], y=-puts["iv"],
        name="Puts IV", marker_color=RED, opacity=0.8,
    ))
    fig_chain.add_vline(x=float(spot_in), line_dash="dot", line_color=CYAN, line_width=1,
                        annotation_text="S₀", annotation_font_color=CYAN)
    fig_chain.add_hline(y=0, line_color="#333", line_width=1)
    fig_chain.update_layout(
        **PLOTLY_LAYOUT,
        height=320, barmode="overlay",
        xaxis_title="Strike", yaxis_title="IV (calls ↑  puts ↓)",
        title="Implied Volatility by Strike",
    )
    st.plotly_chart(fig_chain, use_container_width=True)

    # ── Volume bar ──
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=calls["strike"], y=calls["volume"],
                              name="Call volume", marker_color=GREEN, opacity=0.7))
    fig_vol.add_trace(go.Bar(x=puts["strike"], y=-puts["volume"],
                              name="Put volume", marker_color=RED, opacity=0.7))
    fig_vol.add_vline(x=float(spot_in), line_dash="dot", line_color=CYAN, line_width=1)
    fig_vol.add_hline(y=0, line_color="#333")
    fig_vol.update_layout(
        **PLOTLY_LAYOUT,
        height=220, barmode="overlay",
        xaxis_title="Strike", yaxis_title="Volume",
        title="Volume by Strike",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Table ──
    avail = [c for c in show_cols if c in ch.columns]
    fmt   = {"iv": "{:.1%}", "delta": "{:.3f}", "gamma": "{:.5f}",
             "vega": "{:.2f}", "theta": "{:.3f}", "prob_itm": "{:.1%}",
             "spread_pct": "{:.1f}%", "mny": "{:.3f}", "days": "{:.0f}"}
    display = ch[avail].copy()
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(
                lambda x: f.format(x) if isinstance(x, float) and np.isfinite(x) else "—")
    st.dataframe(display, use_container_width=True, height=420, hide_index=True)


# ───────────────────────────────────────────────────────────────────
# TAB 2 — VOL SMILE & MODELS
# ───────────────────────────────────────────────────────────────────
with tab_smile:
    st.markdown("#### Volatility Smile — Model Fitting")

    sm1, sm2, sm3, sm4 = st.columns(4)
    exp_sel_s  = sm1.selectbox("Expiry", expiries, key="smile_exp",
                                format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    model_mode = sm2.selectbox("Models", ["SVI", "SABR", "Quadratic", "All Three"])
    xaxis_s    = sm3.radio("X-axis", ["Strike", "Log-moneyness k"], horizontal=True)
    otm_only   = sm4.checkbox("OTM only for fit", value=True)

    ds = df.copy()
    ds = ds[pd.to_datetime(ds["expiration"]) == pd.to_datetime(exp_sel_s)]
    ds = ds[np.isfinite(ds["iv"]) & (ds["iv"] > 0) & np.isfinite(ds["vega"])]

    if ds.empty:
        st.warning("No valid IVs for this expiry.")
        st.stop()

    S0  = float(ds["spot_mid"].median())
    T0  = float(ds["T_true"].iloc[0])
    F0  = float(ds["F"].iloc[0])

    ds_fit = ds[ds["mny"].between(*mny_range)].copy()
    if otm_only:
        ds_fit = ds_fit[
            ((ds_fit["cp"] == "C") & (ds_fit["strike"] >= S0)) |
            ((ds_fit["cp"] == "P") & (ds_fit["strike"] <= S0))
        ].copy()

    xcol = "k" if "Log" in xaxis_s else "strike"
    atm_x = 0.0 if xcol == "k" else S0

    # ── Fit models ──────────────────────────────────────────────────
    svi_p = sabr_p = quad_p = None

    if model_mode in ("SVI", "All Three"):
        svi_p = fit_svi(ds_fit["k"], ds_fit["iv"], T0, weights=ds_fit["vega"].to_numpy())
        if svi_p:
            cols = st.columns(5)
            for col_w, (k_name, v) in zip(cols, svi_p.items()):
                col_w.metric(f"SVI {k_name}", f"{v:.5f}")

    if model_mode in ("SABR", "All Three"):
        sabr_p = fit_sabr(F0, ds_fit["strike"], ds_fit["iv"], T0,
                           beta=1.0, weights=ds_fit["vega"].to_numpy())
        if sabr_p:
            cols = st.columns(4)
            for col_w, k_name in zip(cols, ["alpha", "beta", "rho", "nu"]):
                cols[list(["alpha","beta","rho","nu"]).index(k_name)].metric(
                    f"SABR {k_name}", f"{sabr_p[k_name]:.5f}")

    if model_mode in ("Quadratic", "All Three"):
        quad_p = fit_quadratic(ds_fit["k"], ds_fit["iv"], weights=ds_fit["vega"].to_numpy())
        if quad_p:
            qc = st.columns(3)
            qc[0].metric("Quad a₀ (level)", f"{quad_p['a0']:.5f}")
            qc[1].metric("Quad a₁ (skew)",  f"{quad_p['a1']:.5f}")
            qc[2].metric("Quad a₂ (curv.)", f"{quad_p['a2']:.5f}")

    # ── Build dense grid for model curves ───────────────────────────
    k_grid = np.linspace(float(ds["k"].min()) - 0.02, float(ds["k"].max()) + 0.02, 200)
    K_grid = F0 * np.exp(k_grid)
    x_grid = k_grid if xcol == "k" else K_grid

    # ── Figure 1: Smile + Models ─────────────────────────────────────
    fig_smile = go.Figure()

    for cp_, color in [("C", GREEN), ("P", RED)]:
        tmp = ds[ds["cp"] == cp_].sort_values(xcol)
        if tmp.empty:
            continue
        fig_smile.add_trace(go.Scatter(
            x=tmp[xcol], y=tmp["iv"], mode="markers",
            name=f"IV mkt {cp_}",
            marker=dict(color=color, size=6, opacity=0.85,
                        line=dict(color="#0d0d0f", width=0.5)),
            hovertemplate="K=%{customdata[0]:,.0f}<br>IV=%{y:.1%}<br>%{customdata[1]}<extra></extra>",
            customdata=list(zip(tmp["strike"], tmp["symbol"])),
        ))

    if svi_p:
        iv_svi = svi_iv(k_grid, T0, svi_p["a"], svi_p["b"], svi_p["rho"], svi_p["m"], svi_p["sigma"])
        fig_smile.add_trace(go.Scatter(x=x_grid, y=iv_svi, mode="lines",
            name="SVI", line=dict(color=CYAN, width=2)))

    if sabr_p:
        iv_sabr = np.array([
            sabr_iv_hagan(F0, K, T0, sabr_p["alpha"], sabr_p["beta"], sabr_p["rho"], sabr_p["nu"])
            for K in K_grid
        ])
        fig_smile.add_trace(go.Scatter(x=x_grid, y=iv_sabr, mode="lines",
            name="SABR", line=dict(color=AMBER, width=2, dash="dash")))

    if quad_p:
        iv_quad = quad_iv(k_grid, quad_p["a0"], quad_p["a1"], quad_p["a2"])
        fig_smile.add_trace(go.Scatter(x=x_grid, y=iv_quad, mode="lines",
            name="Quadratic", line=dict(color=PURPLE, width=2, dash="dot")))

    fig_smile.add_vline(x=atm_x, line_dash="dot", line_color="#444", line_width=1)
    fig_smile.update_layout(**PLOTLY_LAYOUT, height=460,
        xaxis_title=xcol, yaxis_title="Implied Volatility",
        title="Vol Smile — Market vs Models")
    fig_smile.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_smile, use_container_width=True)

    # ── Figure 2: Residuals (IV − model) × vega ──────────────────────
    if any([svi_p, sabr_p, quad_p]):
        st.markdown("##### Model Residuals  ·  (IV − model) × vega")

        # Compute residuals on market data
        ds_res = ds.copy()
        ds_res["k"] = ds_res["k"].astype(float)

        if svi_p:
            ds_res["iv_svi"]  = svi_iv(ds_res["k"], T0, svi_p["a"], svi_p["b"],
                                        svi_p["rho"], svi_p["m"], svi_p["sigma"])
            ds_res["res_svi"] = (ds_res["iv"] - ds_res["iv_svi"]) * ds_res["vega"]
        if sabr_p:
            ds_res["iv_sabr"] = ds_res["strike"].apply(
                lambda K: sabr_iv_hagan(F0, K, T0, sabr_p["alpha"], sabr_p["beta"],
                                        sabr_p["rho"], sabr_p["nu"]))
            ds_res["res_sabr"] = (ds_res["iv"] - ds_res["iv_sabr"]) * ds_res["vega"]
        if quad_p:
            ds_res["iv_quad"] = quad_iv(ds_res["k"], quad_p["a0"], quad_p["a1"], quad_p["a2"])
            ds_res["res_quad"] = (ds_res["iv"] - ds_res["iv_quad"]) * ds_res["vega"]

        fig_res = go.Figure()
        res_cols = {"res_svi": (CYAN, "SVI"), "res_sabr": (AMBER, "SABR"), "res_quad": (PURPLE, "Quad")}
        for res_col, (color, label) in res_cols.items():
            if res_col not in ds_res.columns:
                continue
            for cp_, ls in [("C", "solid"), ("P", "dash")]:
                tmp = ds_res[ds_res["cp"] == cp_].sort_values(xcol)
                if tmp.empty or res_col not in tmp.columns:
                    continue
                fig_res.add_trace(go.Scatter(
                    x=tmp[xcol], y=tmp[res_col], mode="lines+markers",
                    name=f"{label} {cp_}",
                    line=dict(color=color, width=1.5, dash=ls),
                    marker=dict(size=4),
                ))

        fig_res.add_hline(y=0, line_color="#333", line_width=1)
        fig_res.add_vline(x=atm_x, line_dash="dot", line_color="#333", line_width=1)
        fig_res.update_layout(**PLOTLY_LAYOUT, height=320,
            xaxis_title=xcol, yaxis_title="(IV − model) × vega",
            title="Residuals (lower = better fit near ATM)")
        st.plotly_chart(fig_res, use_container_width=True)

    # ── IV term structure ──────────────────────────────────────────────
    st.markdown("##### ATM IV — Term Structure")
    ts_rows = []
    for exp in expiries:
        sub = df[(pd.to_datetime(df["expiration"]) == pd.to_datetime(exp)) &
                 (df["mny"].between(0.97, 1.03)) & (df["cp"] == "C")]
        if sub.empty:
            continue
        ts_rows.append({
            "expiry": pd.Timestamp(exp).strftime("%d %b"),
            "days": float(sub["days"].mean()),
            "atm_iv": float(sub["iv"].mean()),
            "atm_iv_p": float(df[(pd.to_datetime(df["expiration"]) == pd.to_datetime(exp)) &
                                  (df["mny"].between(0.97, 1.03)) & (df["cp"] == "P")]["iv"].mean()),
        })

    if ts_rows:
        ts_df = pd.DataFrame(ts_rows)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_df["days"], y=ts_df["atm_iv"],
            mode="lines+markers", name="ATM Call IV",
            line=dict(color=GREEN, width=2),
            marker=dict(size=8, color=GREEN)))
        fig_ts.add_trace(go.Scatter(x=ts_df["days"], y=ts_df["atm_iv_p"],
            mode="lines+markers", name="ATM Put IV",
            line=dict(color=RED, width=2, dash="dash"),
            marker=dict(size=8, color=RED)))
        fig_ts.update_layout(**PLOTLY_LAYOUT, height=280,
            xaxis_title="Days to Expiry", yaxis_title="ATM IV",
            title="ATM IV Term Structure")
        fig_ts.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ts, use_container_width=True)


# ───────────────────────────────────────────────────────────────────
# TAB 3 — AR(1) MEAN-REVERSION
# ───────────────────────────────────────────────────────────────────
with tab_ar1:
    st.markdown("#### AR(1) Mean-Reversion — IV Deviations from Quadratic Smile")
    st.markdown(
        "<small>Fits a quadratic smile per snapshot. Computes residuals ε = IV − IV_quad per option. "
        "Fits AR(1) on ε time series. High |ε| + low half-life = candidate mean-reversion trade.</small>",
        unsafe_allow_html=True
    )
    st.markdown("")

    ar1_c1, ar1_c2, ar1_c3 = st.columns(3)
    exp_sel_a  = ar1_c1.selectbox("Expiry", expiries, key="ar1_exp",
                                   format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    lmny_cut   = ar1_c2.slider("Log-moneyness cut ±", 0.05, 0.35, 0.12, 0.01)
    roll_win   = ar1_c3.number_input("Rolling percentile window", value=20, min_value=5, step=5)

    pct_lo = st.slider("LONG signal threshold (pct <)", 5, 40, 15, 5)
    pct_hi = 100 - pct_lo

    @st.cache_data(show_spinner=False, ttl=600)
    def _build_eps_panel(exp_sel_str, lmny_cut_v, min_pts=5):
        hist = _load_history()
        exp_dt = pd.to_datetime(exp_sel_str)

        exp_row = df[pd.to_datetime(df["expiration"]) == exp_dt].copy()
        if exp_row.empty:
            return pd.DataFrame()

        # camino preferido: usar expiry_key estable
        if ("expiry_key" in exp_row.columns) and ("expiry_key" in hist.columns):
            exp_key = int(exp_row["expiry_key"].iloc[0])
            sub = hist[hist["expiry_key"] == exp_key].copy()
        else:
            # fallback por expiration exacto, por si el cache todavía tiene la versión vieja
            sub = hist[pd.to_datetime(hist["expiration"]) == exp_dt].copy()

        if sub.empty:
            return pd.DataFrame()

        sub["snapshot_ts"] = pd.to_datetime(sub["snapshot_ts"]).dt.tz_localize(None)

        eps_rows = []
        for ts, grp in sub.groupby(pd.Grouper(key="snapshot_ts", freq="1min")):
            if grp.empty:
                continue
            S_v = grp["spot_mid"].dropna().iloc[0] if not grp["spot_mid"].dropna().empty else np.nan
            if not (np.isfinite(S_v) and S_v > 0):
                continue

            exp_naive = exp_dt.tz_localize(None) if exp_dt.tzinfo is not None else exp_dt
            snap_naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
            T_v = max((exp_naive - snap_naive).total_seconds() / (365 * 24 * 3600), 1e-4)
            F_v = S_v * np.exp((float(r_in) - float(q_in)) * T_v)

            grp = grp.copy()
            grp["k"] = np.log(grp["strike"].astype(float) / F_v)
            grp = grp[np.abs(grp["k"]) <= lmny_cut_v].copy()
            if grp.empty:
                continue

            # Compute IV (use last price as proxy)
            grp["iv_est"] = grp.apply(
                lambda r: implied_vol(float(r["last"]), S_v, float(r["strike"]),
                                       T_v, float(r_in), float(q_in), str(r["cp"])),
                axis=1
            )
            grp["vega_est"] = grp.apply(
                lambda r: bs_vega(S_v, float(r["strike"]), T_v, float(r_in), float(q_in),
                                   float(r["iv_est"])) if np.isfinite(r["iv_est"]) else np.nan,
                axis=1
            )
            grp = grp[np.isfinite(grp["iv_est"]) & (grp["iv_est"] > 0)].copy()

            for cp_ in ["C", "P"]:
                sub_cp = grp[grp["cp"] == cp_]
                if len(sub_cp) < min_pts:
                    continue
                wts = np.where(
                    np.isfinite(sub_cp["vega_est"].to_numpy(float)) &
                    (sub_cp["vega_est"].to_numpy(float) > 0),
                    sub_cp["vega_est"].to_numpy(float), 1.0)
                qp = fit_quadratic(sub_cp["k"].to_numpy(float), sub_cp["iv_est"].to_numpy(float), weights=wts)
                if qp is None:
                    continue
                k_arr = sub_cp["k"].to_numpy(float)
                iv_q  = quad_iv(k_arr, qp["a0"], qp["a1"], qp["a2"])
                for i, (_, row_cp) in enumerate(sub_cp.iterrows()):
                    eps_rows.append({
                        "snapshot_ts": ts,
                        "symbol":      str(row_cp["symbol"]),
                        "cp":          cp_,
                        "strike":      float(row_cp["strike"]),
                        "k":           float(row_cp["k"]),
                        "iv":          float(row_cp["iv_est"]),
                        "vega":        float(row_cp.get("vega_est", np.nan)),
                        "iv_quad":     float(iv_q[i]),
                        "eps":         float(row_cp["iv_est"]) - float(iv_q[i]),
                    })
        return pd.DataFrame(eps_rows)

    with st.spinner("Building ε panel from synthetic history…"):
        eps_panel = _build_eps_panel(str(exp_sel_a), float(lmny_cut))


    if eps_panel.empty:
        st.warning("No data for this expiry in the historical panel.")
        st.stop()

    # ── AR(1) stats per option ──────────────────────────────────────
    @st.cache_data(show_spinner=False, ttl=300)
    def _ar1_stats(eps_df, roll_w):
        eps = eps_df.copy()
        eps["snapshot_ts"] = pd.to_datetime(eps["snapshot_ts"])
        rows = []
        for (sym, cp_), g in eps.groupby(["symbol", "cp"]):
            g_idx = g.set_index("snapshot_ts").sort_index()
            eps_ts = g_idx["eps"].resample("1min").mean().dropna()
            if len(eps_ts) < 15:
                continue
            ar1 = fit_ar1(eps_ts.to_numpy())
            arr = eps_ts.to_numpy()
            win = arr[-int(roll_w):]
            win = win[np.isfinite(win)]
            last_eps = arr[-1] if len(arr) else np.nan
            pct_now  = float(np.sum(win <= last_eps) / len(win) * 100) if len(win) >= 3 else np.nan
            rows.append({
                "symbol":      sym,
                "cp":          cp_,
                "strike":      float(g["strike"].dropna().iloc[0]),
                "phi":         ar1["phi"],
                "half_life":   ar1["half_life"],
                "r2":          ar1["r2"],
                "n":           ar1["n"],
                "eps_last":    float(eps_ts.iloc[-1]),
                "eps_std":     float(eps_ts.std()),
                "pct_rolling": pct_now,
                "vega":        float(g["vega"].mean()),
            })
        return pd.DataFrame(rows)

    with st.spinner("Fitting AR(1) per option…"):
        ar1_df = _ar1_stats(eps_panel, int(roll_win))

    if ar1_df.empty:
        st.warning("Not enough data for AR(1) estimation.")
        st.stop()

    # ── Compute eps_now on current chain ───────────────────────────
    ds_now = df.copy()
    ds_now = ds_now[pd.to_datetime(ds_now["expiration"]) == pd.to_datetime(exp_sel_a)]
    ds_now = ds_now[np.isfinite(ds_now["iv"]) & (ds_now["iv"] > 0)].copy()

    for cp_ in ["C", "P"]:
        sub = ds_now[ds_now["cp"] == cp_]
        if len(sub) < 5:
            continue
        wts = np.where(np.isfinite(sub["vega"].to_numpy(float)) & (sub["vega"].to_numpy(float) > 0),
                        sub["vega"].to_numpy(float), 1.0)
        qp = fit_quadratic(sub["k"].to_numpy(float), sub["iv"].to_numpy(float), weights=wts)
        if qp is None:
            continue
        k_arr = sub["k"].to_numpy(float)
        ivq   = quad_iv(k_arr, qp["a0"], qp["a1"], qp["a2"])
        ds_now.loc[sub.index, "iv_quad"] = ivq
        ds_now.loc[sub.index, "eps_now"] = sub["iv"].to_numpy(float) - ivq

    ds_now = ds_now.dropna(subset=["eps_now"])
    ds_now = ds_now.merge(
        ar1_df[["symbol","cp","phi","half_life","r2","eps_std","pct_rolling","n"]],
        on=["symbol","cp"], how="left"
    )
    ds_now["z_score"]  = np.where(
        np.isfinite(ds_now.get("eps_std", pd.Series(np.nan, index=ds_now.index))) &
        (ds_now["eps_std"] > 0),
        ds_now["eps_now"] / ds_now["eps_std"], np.nan)
    ds_now["priority"] = ds_now["eps_now"].abs() * ds_now["vega"].fillna(0)

    def _signal(row):
        p = row.get("pct_rolling", np.nan)
        if np.isfinite(p):
            if p < pct_lo: return "🟢 LONG"
            if p > pct_hi: return "🔴 SHORT"
        return "⚪ FLAT"

    ds_now["signal"] = ds_now.apply(_signal, axis=1)

    # ── Summary metrics ───────────────────────────────────────────
    n_long  = (ds_now["signal"] == "🟢 LONG").sum()
    n_short = (ds_now["signal"] == "🔴 SHORT").sum()
    n_flat  = (ds_now["signal"] == "⚪ FLAT").sum()
    avg_hl  = ar1_df["half_life"].dropna().mean()
    avg_phi = ar1_df["phi"].dropna().mean()

    km1, km2, km3, km4, km5 = st.columns(5)
    km1.metric("🟢 LONG signals",  int(n_long))
    km2.metric("🔴 SHORT signals", int(n_short))
    km3.metric("⚪ FLAT",          int(n_flat))
    km4.metric("Avg half-life",    f"{avg_hl:.1f}p" if np.isfinite(avg_hl) else "—")
    km5.metric("Avg φ (AR1)",      f"{avg_phi:.3f}" if np.isfinite(avg_phi) else "—")

    # ── Scatter: eps_now vs half_life ─────────────────────────────
    fig_scatter = go.Figure()
    for sig_label, color in [("🟢 LONG", GREEN), ("🔴 SHORT", RED), ("⚪ FLAT", "#555")]:
        tmp = ds_now[ds_now["signal"] == sig_label]
        if tmp.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=tmp["half_life"], y=tmp["eps_now"],
            mode="markers", name=sig_label,
            marker=dict(color=color, size=7 + tmp["vega"].fillna(1) / 15,
                        opacity=0.8, line=dict(color="#0d0d0f", width=0.5)),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "ε now: %{y:+.4f}<br>half-life: %{x:.1f}p<extra></extra>"
            ),
            customdata=list(zip(tmp["symbol"], tmp["cp"])),
        ))
    fig_scatter.add_hline(y=0, line_color="#333", line_width=1)
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=380,
        xaxis_title="Half-life (periods)",
        yaxis_title="ε now (IV − IV_quad)",
        title="IV Deviation vs Mean-Reversion Speed  ·  size ∝ vega",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────
    st.markdown("##### Signal Table")

    ar1_tbl_cols = ["symbol","cp","strike","signal","phi","half_life","r2",
                     "eps_now","pct_rolling","z_score","vega","priority","n"]
    ar1_tbl_cols = [c for c in ar1_tbl_cols if c in ds_now.columns]

    tbl_sort = st.selectbox("Sort by", ["priority","eps_now","z_score","half_life"], key="ar1_sort")
    tbl_cp   = st.radio("Show", ["Both","C","P"], horizontal=True, key="ar1_cp_tbl")
    tbl_view = ds_now.copy()
    if tbl_cp != "Both":
        tbl_view = tbl_view[tbl_view["cp"] == tbl_cp]
    tbl_view = tbl_view.sort_values(tbl_sort, ascending=False, na_position="last")

    st.dataframe(
        tbl_view[ar1_tbl_cols].style.format({
            "phi":         "{:.3f}",
            "half_life":   "{:.1f}",
            "r2":          "{:.3f}",
            "eps_now":     "{:+.4f}",
            "pct_rolling": "{:.1f}",
            "z_score":     "{:+.2f}",
            "vega":        "{:.2f}",
            "priority":    "{:.2f}",
            "strike":      "{:,.0f}",
        }, na_rep="—"),
        use_container_width=True, height=360, hide_index=True,
    )

    # ── Time series for selected option ───────────────────────────
    st.markdown("##### Time Series — Selected Option")

    sym_opts = sorted(tbl_view["symbol"].unique().tolist())
    if not sym_opts:
        st.info("No options with active signals.")
        st.stop()

    ts_col1, ts_col2 = st.columns([3, 1])
    sym_sel = ts_col1.selectbox("Option", sym_opts, key="ar1_sym")
    cp_sel  = ts_col2.selectbox("CP", ["C", "P"], key="ar1_cp_ts")

    sym_hist_data = eps_panel[
        (eps_panel["symbol"] == sym_sel) & (eps_panel["cp"] == cp_sel)
    ].copy()

    if sym_hist_data.empty:
        st.warning(f"No historical ε data for {sym_sel} / {cp_sel}.")
        st.stop()

    sym_hist_data = sym_hist_data.set_index("snapshot_ts").sort_index()
    eps_ts = sym_hist_data["eps"].resample("1min").mean().dropna()
    iv_ts  = sym_hist_data["iv"].resample("1min").mean().dropna()
    ivq_ts = sym_hist_data["iv_quad"].resample("1min").mean().dropna()

    # Rolling pct
    eps_arr = eps_ts.to_numpy(float)
    pct_arr = np.full(len(eps_arr), np.nan)
    w = int(roll_win)
    for ii in range(w - 1, len(eps_arr)):
        win = eps_arr[ii - w + 1:ii + 1]
        win = win[np.isfinite(win)]
        if len(win) >= 3:
            pct_arr[ii] = float(np.sum(win <= eps_arr[ii]) / len(win) * 100)
    pct_ts = pd.Series(pct_arr, index=eps_ts.index)

    common_idx = eps_ts.index
    x_labels   = [t.strftime("%m/%d %H:%M") for t in common_idx]
    tick_every = max(1, len(x_labels) // 20)

    # AR1 metrics for this symbol
    ar1_row = ar1_df[(ar1_df["symbol"] == sym_sel) & (ar1_df["cp"] == cp_sel)]

    fig_ts_chart = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.33, 0.37, 0.30],
        vertical_spacing=0.05,
        subplot_titles=[
            f"IV vs Quadratic  ·  {sym_sel} ({cp_sel})",
            "ε = IV − IV_quad",
            "Rolling Percentile",
        ],
    )

    # Panel 1: IV vs IV_quad
    fig_ts_chart.add_trace(go.Scatter(
        x=x_labels, y=iv_ts.reindex(common_idx).values,
        mode="lines", name="IV mkt",
        line=dict(color=CYAN, width=1.5),
    ), row=1, col=1)
    fig_ts_chart.add_trace(go.Scatter(
        x=x_labels, y=ivq_ts.reindex(common_idx).values,
        mode="lines", name="IV quad",
        line=dict(color=AMBER, width=1.5, dash="dash"),
    ), row=1, col=1)

    # Panel 2: ε bars
    colors_eps = [RED if v > 0 else GREEN for v in eps_arr]
    fig_ts_chart.add_trace(go.Bar(
        x=x_labels, y=eps_arr, name="ε",
        marker_color=colors_eps, opacity=0.7,
    ), row=2, col=1)
    fig_ts_chart.add_hline(y=0, line_color="#333", row=2, col=1)

    # 1σ bands if available
    if not ar1_row.empty:
        eps_s = float(ar1_row["eps_std"].iloc[0])
        if np.isfinite(eps_s):
            fig_ts_chart.add_hline(y= eps_s, line_dash="dot", line_color="rgba(239,68,68,0.45)", row=2, col=1)
            fig_ts_chart.add_hline(y=-eps_s, line_dash="dot", line_color="rgba(34,197,94,0.45)", row=2, col=1)

    # Panel 3: pct
    fig_ts_chart.add_trace(go.Scatter(
        x=x_labels, y=pct_ts.reindex(common_idx).values,
        mode="lines", name="Pct",
        line=dict(color=PURPLE, width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.07)",
    ), row=3, col=1)
    fig_ts_chart.add_hrect(y0=0, y1=pct_lo, fillcolor="rgba(34,197,94,0.08)", line_width=0, row=3, col=1)
    fig_ts_chart.add_hrect(y0=pct_hi, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0, row=3, col=1)

    fig_ts_chart.update_yaxes(title_text="IV", row=1, col=1, tickformat=".0%")
    fig_ts_chart.update_yaxes(title_text="ε",  row=2, col=1)
    fig_ts_chart.update_yaxes(title_text="Pct",row=3, col=1, range=[0, 100])

    xaxis_cfg = dict(
        type="category",
        tickmode="array",
        tickvals=x_labels[::tick_every],
        tickangle=-45,
        tickfont=dict(size=9),
    )
    fig_ts_chart.update_layout(
        **PLOTLY_LAYOUT,
        height=620,
        barmode="relative",
        showlegend=True,
        legend=dict(orientation="h", y=1.03),
    )

    fig_ts_chart.update_xaxes(**xaxis_cfg, row=1, col=1)
    fig_ts_chart.update_xaxes(**xaxis_cfg, row=2, col=1)
    fig_ts_chart.update_xaxes(**xaxis_cfg, row=3, col=1)
    st.plotly_chart(fig_ts_chart, use_container_width=True)

    # Metrics for selected option
    if not ar1_row.empty:
        r1 = ar1_row.iloc[0]
        ra1, ra2, ra3, ra4, ra5 = st.columns(5)
        ra1.metric("φ (AR1)", f"{r1['phi']:.4f}" if np.isfinite(r1['phi']) else "—")
        hl_val = r1['half_life']
        ra2.metric("Half-life", f"{hl_val:.1f} periods" if np.isfinite(hl_val) else "—")
        ra3.metric("R²", f"{r1['r2']:.3f}" if np.isfinite(r1['r2']) else "—")
        ra4.metric("n obs", f"{int(r1['n']):,}")
        pct_now_val = ds_now.loc[
            (ds_now["symbol"] == sym_sel) & (ds_now["cp"] == cp_sel), "pct_rolling"
        ].values
        ra5.metric("Rolling pct", f"{float(pct_now_val[0]):.1f}" if len(pct_now_val) and np.isfinite(pct_now_val[0]) else "—")
