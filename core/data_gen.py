"""
Synthetic options chain generator.

Produces a realistic options chain with:
 - Volatility smile (skew + smile via SABR-like parametrization)
 - Multiple expiries
 - Bid/ask spread as function of moneyness and DTE
 - Volume / OI proportional to liquidity proxy
 - Intraday snapshots for AR(1) tab
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from core.bs import bs_price, bs_vega, implied_vol


# ── Config ────────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)

SPOT          = 1_050.0       # GGAL-like level
R             = 0.20          # ~AR rate
Q             = 0.002
EXPIRIES_DAYS = [21, 49]      # two expiries

# SABR-like IV generation params per expiry
SABR_PARAMS   = {
    21: dict(atm_iv=0.65, skew=-0.12, smile=0.08),
    49: dict(atm_iv=0.58, skew=-0.09, smile=0.06),
}

STRIKES_GRID  = np.arange(700, 1_401, 25)   # 29 strikes
N_SNAPSHOTS   = 60                            # 60 × 1-min snapshots per day
N_DAYS        = 8                             # 8 trading days of history


def _smile_iv(K, F, T, atm_iv, skew, smile):
    """Simple parametric smile: iv = atm + skew*k + smile*k^2, k = ln(K/F)."""
    k = np.log(K / F)
    iv = atm_iv + skew * k + smile * k**2
    return float(np.clip(iv, 0.05, 3.0))


def _make_expiry_ts(now: datetime, dte: int) -> datetime:
    exp = now + timedelta(days=dte)
    return exp.replace(hour=17, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def generate_chain(
    spot: float = SPOT,
    r: float = R,
    q: float = Q,
    snapshot_time: datetime | None = None,
    noise_sigma: float = 0.008,
) -> pd.DataFrame:
    """
    Generate a single-snapshot options chain.

    Parameters
    ----------
    spot          : underlying price
    r, q          : rates
    snapshot_time : datetime for the snapshot (default = now)
    noise_sigma   : IV noise std (realistic market microstructure)

    Returns
    -------
    pd.DataFrame with columns compatible with app_options.py
    """
    if snapshot_time is None:
        snapshot_time = datetime.now(timezone.utc)

    rows = []
    for dte, p in SABR_PARAMS.items():
        exp_ts = _make_expiry_ts(snapshot_time, dte)
        T = max((exp_ts - snapshot_time).total_seconds() / (365 * 24 * 3600), 1e-4)
        F = spot * np.exp((r - q) * T)

        for K in STRIKES_GRID:
            for cp in ["C", "P"]:
                # IV with noise
                base_iv = _smile_iv(K, F, T, **p)
                iv = float(np.clip(base_iv + RNG.normal(0, noise_sigma), 0.04, 3.0))

                theo = bs_price(spot, K, T, r, q, iv, cp)
                if not (np.isfinite(theo) and theo > 0):
                    continue

                vega = bs_vega(spot, K, T, r, q, iv)

                # Bid/ask spread: wider OTM and near-expiry
                mny = K / spot
                spread_pct = 0.04 + 0.10 * abs(mny - 1.0) + 0.02 * (1 / max(T * 365, 1))
                half_spread = theo * spread_pct / 2
                bid = max(0.01, theo - half_spread)
                ask = theo + half_spread
                mid = (bid + ask) / 2

                # Volume / OI: ATM liquid, wings sparse
                liq_proxy = np.exp(-6 * (mny - 1.0)**2) * np.sqrt(T * 365)
                volume     = int(max(0, RNG.poisson(lam=max(0.1, 200 * liq_proxy))))
                oi_total   = int(max(0, RNG.poisson(lam=max(0.1, 800 * liq_proxy))))

                # symbol: GGAL + expiry month + K + cp
                exp_label = exp_ts.strftime("%b%Y").lower()
                symbol    = f"GGAL{exp_label}{int(K)}{cp}"

                rows.append({
                    "symbol":      symbol,
                    "cp":          cp,
                    "strike":      float(K),
                    "expiration":  exp_ts,
                    "snapshot_ts": snapshot_time,
                    "spot_mid":    spot,
                    "spot_last":   spot * (1 + RNG.normal(0, 0.001)),
                    "bid":         round(bid, 2),
                    "ask":         round(ask, 2),
                    "mid":         round(mid, 2),
                    "last":        round(theo * (1 + RNG.normal(0, 0.005)), 2),
                    "close":       round(theo, 2),
                    "volume":      volume,
                    "operations":  max(0, volume // 3),
                    "oi_total":    oi_total,
                    "T_true":      T,
                    "iv_true":     iv,
                })

    return pd.DataFrame(rows)


def generate_history(
    n_days: int = N_DAYS,
    n_snapshots_per_day: int = N_SNAPSHOTS,
    spot_drift: float = 0.0002,
    spot_vol: float = 0.008,
    iv_drift: float = -0.0002,
    iv_vol: float = 0.010,
) -> pd.DataFrame:
    """
    Generate N_DAYS × N_SNAPSHOTS of intraday option snapshots.
    Used for the AR(1) mean-reversion tab.
    """
    today = datetime.now(timezone.utc).replace(hour=10, minute=30, second=0, microsecond=0)
    base_time = today - timedelta(days=n_days)
    all_dfs = []

    spot = SPOT
    atm_iv_state = {dte: p["atm_iv"] for dte, p in SABR_PARAMS.items()}

    # VENCIMIENTOS FIJOS PARA TODO EL HISTÓRICO
    ref_now = datetime.now(timezone.utc)
    fixed_expiries = {dte: _make_expiry_ts(ref_now, dte) for dte in SABR_PARAMS.keys()}

    for day in range(n_days):
        day_offset = timedelta(days=day)
        for snap_i in range(n_snapshots_per_day):
            ts = base_time + day_offset + timedelta(minutes=snap_i)

            # Random walk spot
            spot = spot * np.exp(
                (spot_drift - 0.5 * spot_vol**2) + spot_vol * RNG.normal()
            )
            spot = float(np.clip(spot, SPOT * 0.5, SPOT * 2.0))

            # Mean-reverting IV level
            for dte in atm_iv_state:
                atm_iv_state[dte] = float(np.clip(
                    atm_iv_state[dte]
                    + iv_drift * (atm_iv_state[dte] - SABR_PARAMS[dte]["atm_iv"])
                    + iv_vol * RNG.normal(),
                    0.20, 1.50
                ))

            for dte, p in SABR_PARAMS.items():
                exp_ts = fixed_expiries[dte]
                T = max((exp_ts - ts).total_seconds() / (365 * 24 * 3600), 1e-4)
                F = spot * np.exp((R - Q) * T)

                p_now = dict(
                    atm_iv=atm_iv_state[dte],
                    skew=p["skew"] + RNG.normal(0, 0.005),
                    smile=p["smile"] + RNG.normal(0, 0.003),
                )

                for K in STRIKES_GRID:
                    for cp in ["C", "P"]:
                        iv = float(np.clip(
                            _smile_iv(K, F, T, **p_now) + RNG.normal(0, 0.012),
                            0.04, 3.0
                        ))
                        theo = bs_price(spot, K, T, R, Q, iv, cp)
                        if not (np.isfinite(theo) and theo > 0):
                            continue

                        mny = K / spot
                        spread_pct = 0.04 + 0.10 * abs(mny - 1.0)
                        half = theo * spread_pct / 2
                        bid = max(0.01, theo - half)
                        ask = theo + half

                        exp_label = exp_ts.strftime("%b%Y").lower()
                        symbol = f"GGAL{exp_label}{int(K)}{cp}"

                        all_dfs.append({
                            "symbol":      symbol,
                            "cp":          cp,
                            "strike":      float(K),
                            "expiration":  exp_ts,
                            "snapshot_ts": ts,
                            "spot_mid":    spot,
                            "spot_last":   spot,
                            "bid":         round(bid, 2),
                            "ask":         round(ask, 2),
                            "mid":         round((bid + ask) / 2, 2),
                            "last":        round(theo, 2),
                            "close":       round(theo, 2),
                            "volume":      int(max(0, RNG.poisson(5))),
                            "operations":  int(max(0, RNG.poisson(2))),
                            "oi_total":    int(max(0, RNG.poisson(50))),
                        })

    return pd.DataFrame(all_dfs)
