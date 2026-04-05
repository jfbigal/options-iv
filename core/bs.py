"""
Black-Scholes pricing, Greeks, and implied volatility.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ── d1 / d2 ────────────────────────────────────────────────────────────────

def _d1d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ── Pricing ─────────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, q, sigma, cp):
    cp = str(cp).upper()
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if cp == "C":
        return disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
    if cp == "P":
        return disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)
    return np.nan


# ── Greeks ──────────────────────────────────────────────────────────────────

def bs_delta(S, K, T, r, q, sigma, cp):
    cp = str(cp).upper()
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    disc_q = np.exp(-q * T)
    return disc_q * norm.cdf(d1) if cp == "C" else disc_q * (norm.cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    return np.exp(-q * T) * S * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, T, r, q, sigma, cp):
    cp = str(cp).upper()
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    term1 = -(disc_q * S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if cp == "C":
        return term1 - q * disc_q * S * norm.cdf(d1) + r * disc_r * K * norm.cdf(d2)
    return term1 + q * disc_q * S * norm.cdf(-d1) - r * disc_r * K * norm.cdf(-d2)


def bs_charm(S, K, T, r, q, sigma, cp):
    cp = str(cp).upper()
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1):
        return np.nan
    disc_q = np.exp(-q * T)
    denom = 2.0 * T * sigma * np.sqrt(T)
    term = (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / denom
    if cp == "C":
        return -q * disc_q * norm.cdf(d1) - disc_q * norm.pdf(d1) * term
    return -q * disc_q * (norm.cdf(d1) - 1.0) - disc_q * norm.pdf(d1) * term


def bs_vanna(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d1) or sigma <= 0 or T <= 0:
        return np.nan
    return np.exp(-q * T) * norm.pdf(d1) * (np.sqrt(T) - d1 / sigma)


# ── Implied Volatility ───────────────────────────────────────────────────────

def implied_vol(price, S, K, T, r, q, cp):
    """Brentq root-finding for implied volatility."""
    if any(np.isnan(v) for v in [price, S, K, T]) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    cp = str(cp).upper()
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    intrinsic = max(0.0, disc_q * S - disc_r * K) if cp == "C" else max(0.0, disc_r * K - disc_q * S)
    if price < intrinsic - 1e-10:
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, q, sig, cp) - price

    try:
        lo, hi = 1e-6, 5.0
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            return np.nan
        if flo * fhi > 0:
            for hi2 in [8.0, 12.0]:
                if f(lo) * f(hi2) <= 0:
                    hi = hi2
                    break
            else:
                return np.nan
        return brentq(f, lo, hi, maxiter=200)
    except Exception:
        return np.nan


# ── Prob ITM (risk-neutral) ──────────────────────────────────────────────────

def prob_itm(S, K, T, r, q, sigma, cp):
    _, d2 = _d1d2(S, K, T, r, q, sigma)
    if not np.isfinite(d2):
        return np.nan
    return norm.cdf(d2) if str(cp).upper() == "C" else norm.cdf(-d2)
