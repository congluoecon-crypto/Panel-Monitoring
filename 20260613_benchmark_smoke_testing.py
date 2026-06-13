"""
20260613_smoke_testing.py
==========================
Smoke test for the *revised* Brownian Bridge structural break monitoring
framework discussed in the design conversation. This script:

  1) Implements the DGP for N = 10 (Table 3.1, Case 2) and N = 30
     (Table 3.1, Case 3), and the four common factors (Table 3.2).
  2) Replaces the O(T^2) python-loop fractional-differencing filter with an
     O(T log T) FFT-based convolution (scipy.signal.fftconvolve).
  3) Implements the closed-form reduction of the "average over all N(N-1)
     ordered-pair pooled-OLS residuals" to
         bar_e_t = (1-beta_hat) * (mean_i hat_e_{i,t} - mu)
     which is estimated in O(N*M) instead of O(N^2*M).
  4) Implements the CUSUM-of-squares statistic T_n and the Brownian-Bridge
     boundary g(n; a_alpha^2) for alpha in {10%,5%,1%} (unchanged).
  5) Implements:
       - Scenario A (delta = 0): k is a *post-hoc scoring window* M+kM;
         one simulated path is scored against all 6 values of k.
       - Scenario B (delta in {1,2,3}): k determines the *break position*
         tau(k) = M + kM; records overall alarm, premature-alarm (before
         tau), post-break detection (after tau), and detection delay.

This file does NOT run the full 1.9M-replication experiment. It only runs
a handful of replications per "heaviest" and "lightest" parameter
combination, prints sanity-check output, and reports per-replication
timing so that the total runtime of the full design can be extrapolated.
"""

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.signal import fftconvolve
from scipy.special import gammaln

# --------------------------------------------------------------------------
# 0. Fractional differencing via FFT (replaces the O(T^2) python loop)
# --------------------------------------------------------------------------
def fracdiff_coefs(d, n_max):
    """w_j = Gamma(j+d) / (Gamma(j+1) Gamma(d)), j = 0..n_max-1, via gammaln
    (numerically stable for large j, fully vectorized)."""
    if d == 0:
        w = np.zeros(n_max)
        w[0] = 1.0
        return w
    j = np.arange(n_max)
    log_w = gammaln(j + d) - gammaln(j + 1) - gammaln(d)
    return np.exp(log_w)


def fracdiff_filter(Y, d):
    """X_t = sum_{j=0}^{t} w_j * Y_{t-j}  ==  (Y * w)[:T]  (full convolution)."""
    n = len(Y)
    if d == 0:
        return Y.copy()
    w = fracdiff_coefs(d, n)
    return fftconvolve(Y, w)[:n]


def gen_series(ar, d, ma, nsample):
    """Generate one ARFIMA(ar, d, ma) series. ar/ma are scalars (single AR/MA
    coefficient); d is the fractional-integration order. ar=0 & ma=0 with
    d=0 -> i.i.d. N(0,1)."""
    if ar == 0 and ma == 0 and d == 0:
        return np.random.normal(0, 1, nsample)
    ar_poly = np.array([1.0, -ar]) if ar != 0 else np.array([1.0])
    ma_poly = np.array([1.0, ma]) if ma != 0 else np.array([1.0])
    Y = ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=nsample)
    return fracdiff_filter(Y, d)


# --------------------------------------------------------------------------
# 1. DGP -- Table 3.1
# --------------------------------------------------------------------------
def data_generation_N10(nsample=4000):
    """Case 2 (N=10): columns (ar, d, ma) per Table 3.1."""
    specs = [
        (0.8, 0.0, 0.5),  # s1
        (0.8, 0.0, 0.5),  # s2
        (0.5, 0.0, 0.3),  # s3
        (0.5, 0.0, 0.3),  # s4
        (0.0, 0.4, 0.0),  # s5
        (0.0, 0.4, 0.0),  # s6
        (0.1, 0.3, 0.8),  # s7
        (0.1, 0.3, 0.8),  # s8
        (0.7, 0.4, 0.0),  # s9
        (0.7, 0.4, 0.0),  # s10
    ]
    data = {f"s{i+1}": gen_series(ar, d, ma, nsample)
            for i, (ar, d, ma) in enumerate(specs)}
    return pd.DataFrame(data)


def data_generation_N30(nsample=4000):
    """Case 3 (N=30): random-coefficient panel per Table 3.1.
    Coefficients are redrawn for *every* call (i.e. every simulation
    replication), per the agreed design."""
    cols = {}
    for i in range(1, 16):           # 1-15: ARMA(1,1), ar~U(0,0.5), ma=0.6
        ar = np.random.uniform(0.0, 0.5)
        cols[f"s{i}"] = gen_series(ar, 0.0, 0.6, nsample)
    for i in range(16, 21):          # 16-20: ARMA(1,1), ar~U(0.7,0.9), ma=0.6
        ar = np.random.uniform(0.7, 0.9)
        cols[f"s{i}"] = gen_series(ar, 0.0, 0.6, nsample)
    for i in range(21, 26):          # 21-25: ARFIMA(0,d,0), d~U(0,0.5)
        d = np.random.uniform(0.0, 0.5)
        cols[f"s{i}"] = gen_series(0.0, d, 0.0, nsample)
    for i in range(26, 31):          # 26-30: ARFIMA(1,d,1)
        ar = np.random.uniform(0.0, 0.4)
        d = np.random.uniform(0.0, 0.5)
        ma = np.random.uniform(0.0, 0.3)
        cols[f"s{i}"] = gen_series(ar, d, ma, nsample)
    return pd.DataFrame(cols)


def common_factor_generation(nsample=4000):
    """Table 3.2: factor1=IID, factor2=AR(1,ar=0.99), factor3=ARFIMA(0,0.2,0),
    factor4=ARFIMA(0,0.45,0)."""
    f1 = np.random.normal(0, 1, nsample)
    f2 = gen_series(0.99, 0.0, 0.0, nsample)
    f3 = gen_series(0.0, 0.2, 0.0, nsample)
    f4 = gen_series(0.0, 0.45, 0.0, nsample)
    return pd.DataFrame({"factor1": f1, "factor2": f2, "factor3": f3, "factor4": f4})


# --------------------------------------------------------------------------
# 2. AR(p) fixed-coefficient filtering (unchanged from original code)
# --------------------------------------------------------------------------
def rolling_ar_residuals_fixed(series, M):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = max(3, int((M - 1) // 100) * 3)
    if M <= p:
        raise ValueError(f"M={M} too small for AR order p={p}")
    y_cal = series[p:M]
    X_cal = np.column_stack([series[p - i - 1:M - i - 1] for i in range(p)])
    beta, *_ = np.linalg.lstsq(X_cal, y_cal, rcond=None)
    residuals = np.full(n, np.nan)
    for t in range(8, n + 1):
        if t <= p:
            continue
        last_obs = series[t - p - 1:t - 1][::-1]
        residuals[t - 1] = series[t - 1] - beta @ last_obs
    return residuals


# --------------------------------------------------------------------------
# 3. Closed-form pooled cross-sectional residual mean  bar_e_t
#
#    bar_e_t = (1/(N(N-1))) * sum_{i!=j} U_{i,j,t}
#            = (1 - beta_hat) * (mean_i hat_e_{i,t} - mu)
#
#    where, on the calibration sample t=1..M, with d_{i,t}=hat_e_{i,t}-mu,
#    D_t = sum_i d_{i,t}, Q'_t = sum_i d_{i,t}^2:
#       beta_hat = [sum_t (D_t^2 - Q'_t)] / [(N-1) sum_t Q'_t]
#       mu       = grand mean of hat_e_{i,t} over i=1..N, t=1..M
# --------------------------------------------------------------------------
def pooled_cross_section_mean(resid_matrix, M):
    N = resid_matrix.shape[1]
    e_bar_full = resid_matrix.mean(axis=1)              # mean_i hat_e_{i,t}, full sample
    mu = e_bar_full[:M].mean()                          # grand mean, calibration only
    d_cal = resid_matrix[:M, :] - mu
    D = d_cal.sum(axis=1)
    Qp = (d_cal ** 2).sum(axis=1)
    num = np.sum(D ** 2 - Qp)
    den = (N - 1) * np.sum(Qp)
    beta_hat = num / den
    e_t = (1 - beta_hat) * (e_bar_full - mu)
    return e_t, beta_hat, mu


# --------------------------------------------------------------------------
# 4. CUSUM-of-squares statistic T_n and Brownian-Bridge boundary g(n;a^2)
# --------------------------------------------------------------------------
A2 = {"10%": 6.2514, "5%": 7.8147, "1%": 11.3449}


def cusum_alarm_times(e_t, M):
    """Returns (Tn, ns, alarms) where alarms[level] = first n at which
    |T_n| >= g(n; a^2_level), or np.inf if never."""
    T = len(e_t)
    cumsum_sq = np.cumsum(e_t ** 2)
    Cm = cumsum_sq[M - 1]
    ns = np.arange(M + 1, T + 1, dtype=float)
    Cn = cumsum_sq[M:T]
    Dn = Cn / Cm - (ns - 2) / (M - 2)
    Tn = np.sqrt(M / 2.0) * Dn

    ratio = (ns - M) / M
    inner = ns / (ns - M)

    alarms = {}
    for level, a2 in A2.items():
        g = ratio * np.sqrt(inner * (a2 + np.log(inner)))
        idx = np.where(np.abs(Tn) >= g)[0]
        alarms[level] = ns[idx[0]] if len(idx) > 0 else np.inf
    return Tn, ns, alarms


# --------------------------------------------------------------------------
# 5. Shared pipeline: DGP -> factor loading -> AR filter -> bar_e_t
# --------------------------------------------------------------------------
def build_panel(N, factor, T_total):
    df_raw = data_generation_N10() if N == 10 else data_generation_N30()
    cf = common_factor_generation()
    fcol = cf[f"factor{factor}"].values
    df = df_raw + 0.5 * fcol[:, None]
    return df.iloc[:T_total].reset_index(drop=True)


def ar_filter_panel(df, M):
    resid = np.column_stack(
        [rolling_ar_residuals_fixed(df[c].values, M) for c in df.columns]
    )
    valid = ~np.isnan(resid).any(axis=1)
    n_dropped = int(np.argmax(valid))   # index of first valid row (initial NaN block)
    return resid[valid], n_dropped


# --------------------------------------------------------------------------
# 6. Scenario A (delta = 0): k = post-hoc scoring window M + kM
# --------------------------------------------------------------------------
K_LIST = [0.2, 0.3, 0.5, 1, 2, 3]


def simulation_A(N, m, factor, T_total=None):
    if T_total is None:
        T_total = 10 * m + 10
    df = build_panel(N, factor, T_total)
    resid, _ = ar_filter_panel(df, m)
    e_t, beta_hat, mu = pooled_cross_section_mean(resid, m)
    Tn, ns, alarms = cusum_alarm_times(e_t, m)

    out = {}
    for level, tau_hat in alarms.items():
        for k in K_LIST:
            out[(level, k)] = int(tau_hat <= m + k * m)
    return out, beta_hat


# --------------------------------------------------------------------------
# 7. Scenario B (delta in {1,2,3}): k determines break position tau(k)=M+kM
# --------------------------------------------------------------------------
def inject_break(df, tau, delta):
    df2 = df.copy()
    df2.iloc[tau:] = df2.iloc[tau:] + delta
    return df2


def simulation_B(N, delta, k, m, factor, T_total=None):
    if T_total is None:
        T_total = 10 * m + 10
    df = build_panel(N, factor, T_total)

    tau_pos = m + int(k * m)
    df = inject_break(df, tau_pos, delta)

    resid, n_dropped = ar_filter_panel(df, m)
    tau_adj = tau_pos - n_dropped

    e_t, beta_hat, mu = pooled_cross_section_mean(resid, m)
    Tn, ns, alarms = cusum_alarm_times(e_t, m)

    out = {}
    for level, tau_hat in alarms.items():
        if np.isinf(tau_hat):
            out[level] = dict(alarm=0, premature=0, detect=0, delay=np.nan)
        else:
            premature = int(tau_hat <= tau_adj)
            detect = int(tau_hat > tau_adj)
            delay = (tau_hat - tau_adj) if detect else np.nan
            out[level] = dict(alarm=1, premature=premature, detect=detect, delay=delay)
    return out, beta_hat, tau_adj


# --------------------------------------------------------------------------
# 8. Smoke test driver
# --------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(100)

    print("=" * 70)
    print("Scenario A | N=10, m=80, factor=1 | R=5")
    t0 = time.time()
    for r in range(5):
        out, beta_hat = simulation_A(10, 80, 1)
        print(f"  r={r}  beta_hat={beta_hat:.4f}  size@5%:",
              {k: out[("5%", k)] for k in K_LIST})
    print(f"  avg time/run: {(time.time()-t0)/5*1000:.1f} ms")

    print("=" * 70)
    print("Scenario A | N=30, m=300, factor=4 (heaviest) | R=3")
    t0 = time.time()
    for r in range(3):
        out, beta_hat = simulation_A(30, 300, 4)
        print(f"  r={r}  beta_hat={beta_hat:.4f}  size@5%:",
              {k: out[("5%", k)] for k in K_LIST})
    t_A_heavy = (time.time() - t0) / 3
    print(f"  avg time/run: {t_A_heavy*1000:.1f} ms")

    print("=" * 70)
    print("Scenario B | N=10, delta=2, k=1, m=80, factor=2 | R=5")
    t0 = time.time()
    for r in range(5):
        out, beta_hat, tau_adj = simulation_B(10, 2, 1, 80, 2)
        print(f"  r={r}  beta_hat={beta_hat:.4f}  tau_adj={tau_adj}  5%:", out["5%"])
    print(f"  avg time/run: {(time.time()-t0)/5*1000:.1f} ms")

    print("=" * 70)
    print("Scenario B | N=30, delta=3, k=3, m=300, factor=3 (heaviest) | R=3")
    t0 = time.time()
    for r in range(3):
        out, beta_hat, tau_adj = simulation_B(30, 3, 3, 300, 3)
        print(f"  r={r}  beta_hat={beta_hat:.4f}  tau_adj={tau_adj}  5%:", out["5%"])
    t_B_heavy = (time.time() - t0) / 3
    print(f"  avg time/run: {t_B_heavy*1000:.1f} ms")

    # ---------------------------------------------------------------
    # 9. Extrapolated full-design run count and rough time estimate
    # ---------------------------------------------------------------
    n_A = 2 * 5 * 4 * 2500          # N x m x factor x R
    n_B = 2 * 3 * 6 * 5 * 4 * 2500  # N x delta x k x m x factor x R
    print("=" * 70)
    print(f"Scenario A total replications: {n_A:,}")
    print(f"Scenario B total replications: {n_B:,}")
    print(f"Rough single-thread estimate (using HEAVIEST-case timings as upper bound):")
    est_A = n_A * t_A_heavy
    est_B = n_B * t_B_heavy
    print(f"  Scenario A: {est_A/60:.1f} min  ({est_A/3600:.2f} h)")
    print(f"  Scenario B: {est_B/60:.1f} min  ({est_B/3600:.2f} h)")
    print(f"  Total (single thread, upper bound): {(est_A+est_B)/3600:.2f} h")
    print("  -> divide by number of parallel processes for wall-clock estimate.")
