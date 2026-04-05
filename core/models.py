"""
Volatility surface models: SVI (raw), SABR (Hagan), Quadratic.
All models fit on log-moneyness k = ln(K/F).
"""
import numpy as np
from scipy.optimize import minimize


# ── SVI (raw parametrization) ────────────────────────────────────────────────

def svi_total_var(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    if b <= 0 or sigma <= 0 or not (-0.999 < rho < 0.999):
        return np.full_like(k, np.nan)
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def svi_iv(k, T, a, b, rho, m, sigma):
    w = svi_total_var(k, a, b, rho, m, sigma)
    if not (np.isfinite(T) and T > 0):
        return np.full_like(np.asarray(k, float), np.nan)
    w = np.where(np.isfinite(w), np.maximum(w, 1e-12), np.nan)
    return np.sqrt(w / T)


def fit_svi(k, iv, T, weights=None):
    """
    Fit SVI raw on (k, iv) with optional vega weights.
    Returns dict of params or None.
    """
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    w_mkt = iv**2 * T
    msk = np.isfinite(k) & np.isfinite(w_mkt) & (w_mkt > 0)
    if weights is not None:
        wt = np.asarray(weights, float)
        msk = msk & np.isfinite(wt) & (wt > 0)
    else:
        wt = np.ones_like(iv)

    k, w_mkt, wt = k[msk], w_mkt[msk], wt[msk]
    if k.size < 7:
        return None

    m0     = float(np.median(k))
    sig0   = float(np.clip(np.std(k) * 0.5, 1e-3, 2.0))
    b0     = float(np.clip(np.std(w_mkt) / (np.std(k) + 1e-6), 1e-3, 5.0))
    a0     = float(np.clip(np.min(w_mkt) - b0 * sig0, 1e-6, np.max(w_mkt)))
    x0     = [a0, b0, 0.0, m0, sig0]
    kmin, kmax = float(np.min(k)), float(np.max(k))
    wmax   = float(np.max(w_mkt))
    bounds = [(0, max(5 * wmax, 1e-3)), (1e-8, max(10 * wmax, 0.01)),
              (-0.999, 0.999), (kmin - 1, kmax + 1), (1e-8, 5.0)]

    def obj(x):
        a, b, rho, m, s = x
        if a < 0 or b <= 0 or s <= 0 or not (-0.999 < rho < 0.999):
            return 1e18
        wm = svi_total_var(k, a, b, rho, m, s)
        if not np.all(np.isfinite(wm)) or np.any(wm <= 0):
            return 1e16
        return float(np.sum(wt * (wm - w_mkt)**2))

    res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
    if not res.success:
        return None
    a, b, rho, m, s = res.x
    return dict(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(s))


# ── SABR (Hagan lognormal approximation) ─────────────────────────────────────

def sabr_iv_hagan(F, K, T, alpha, beta, rho, nu):
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0 or nu <= 0:
        return np.nan
    if not (0 <= beta <= 1) or not (-0.999 < rho < 0.999):
        return np.nan

    if abs(F - K) < 1e-12:
        FK = F
        t1 = alpha / (FK**(1 - beta))
        t2 = (((1-beta)**2/24) * alpha**2 / (FK**(2-2*beta))
              + (rho*beta*nu*alpha) / (4 * FK**(1-beta))
              + ((2 - 3*rho**2)/24) * nu**2) * T
        return t1 * (1 + t2)

    logFK = np.log(F / K)
    FK_b  = (F * K)**((1-beta)/2)
    z     = (nu / alpha) * FK_b * logFK
    xz    = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))
    if abs(xz) < 1e-12:
        return np.nan

    denom = 1 + ((1-beta)**2/24)*logFK**2 + ((1-beta)**4/1920)*logFK**4
    tA    = alpha / (FK_b * denom)
    tB    = z / xz
    tC    = 1 + (((1-beta)**2/24) * alpha**2 / ((F*K)**(1-beta))
                 + (rho*beta*nu*alpha) / (4*(F*K)**((1-beta)/2))
                 + ((2-3*rho**2)/24) * nu**2) * T
    return tA * tB * tC


def fit_sabr(F, strikes, iv, T, beta=1.0, weights=None):
    strikes = np.asarray(strikes, float); iv = np.asarray(iv, float)
    msk = np.isfinite(strikes) & np.isfinite(iv) & (strikes > 0) & (iv > 0)
    if weights is not None:
        wt = np.asarray(weights, float)
        msk = msk & np.isfinite(wt) & (wt > 0)
    else:
        wt = np.ones_like(iv)

    Kf, ivm, wf = strikes[msk], iv[msk], wt[msk]
    if len(Kf) < 6:
        return None

    alpha0 = float(np.clip(np.median(ivm) * F**(1-beta), 1e-6, 10.0))

    def obj(x):
        alpha, rho, nu = x
        if alpha <= 0 or nu <= 0 or not (-0.999 < rho < 0.999):
            return 1e9
        iv_m = np.array([sabr_iv_hagan(F, float(K), T, alpha, beta, rho, nu) for K in Kf])
        if not np.all(np.isfinite(iv_m)):
            return 1e9
        return float(np.sum(wf * (iv_m - ivm)**2))

    res = minimize(obj, [alpha0, 0.0, 0.8], bounds=[(1e-8, 50), (-0.999, 0.999), (1e-6, 50)],
                   method="L-BFGS-B")
    if not res.success:
        return None
    alpha, rho, nu = res.x
    return dict(alpha=float(alpha), beta=float(beta), rho=float(rho), nu=float(nu), T=T)


# ── Quadratic (WLS) ──────────────────────────────────────────────────────────

def fit_quadratic(k, iv, weights=None):
    k = np.asarray(k, float); iv = np.asarray(iv, float)
    msk = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    if weights is not None:
        wt = np.asarray(weights, float)
        msk = msk & np.isfinite(wt) & (wt > 0)
    else:
        wt = np.ones_like(iv)

    k, iv, wt = k[msk], iv[msk], wt[msk]
    if len(k) < 5:
        return None

    W = np.diag(wt)
    X = np.column_stack([np.ones_like(k), k, k**2])
    XtW = X.T @ W
    try:
        c = np.linalg.lstsq(XtW @ X, XtW @ iv, rcond=None)[0]
        return dict(a0=float(c[0]), a1=float(c[1]), a2=float(c[2]))
    except Exception:
        return None


def quad_iv(k, a0, a1, a2):
    return np.clip(a0 + a1 * np.asarray(k, float) + a2 * np.asarray(k, float)**2, 1e-6, None)


# ── AR(1) helpers ────────────────────────────────────────────────────────────

def fit_ar1(arr):
    """OLS AR(1) on a 1-D array. Returns phi, half_life, r2."""
    s = np.asarray(arr, float)
    s = s[np.isfinite(s)]
    out = dict(phi=np.nan, half_life=np.nan, r2=np.nan, n=len(s))
    if len(s) < 15:
        return out
    y = s[1:] - s[1:].mean()
    x = s[:-1] - s[:-1].mean()
    vx = float(np.dot(x, x))
    if vx < 1e-18:
        return out
    phi = float(np.dot(x, y) / vx)
    res = y - phi * x
    ss_res = float(np.dot(res, res))
    ss_tot = float(np.dot(y, y))
    r2  = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan
    hl  = np.log(0.5) / np.log(abs(phi)) if 0 < abs(phi) < 1.0 else np.nan
    out.update(phi=phi, half_life=float(hl) if np.isfinite(hl) else np.nan, r2=float(r2))
    return out
