"""
simulation_revised.py
======================
Power-focused Monte Carlo simulation for the revised Brownian Bridge
structural break monitoring framework.

Experiments:
  Exp1  - Factor 4 volatility break: residual std 1 -> 3, factor=4 only
  Exp2  - Factor loading break: gamma 0.1 -> 0.8, all 4 factors separately
  Exp3.1 - Weak factor panel, volatility break (std 1->3), subgroup reporting
  Exp3.2 - Weak factor panel, loading break per subgroup, subgroup reporting

k in {0.2, 0.3, 0.5} only (break at tau* = (1+k)*M).
R = 2500 replications per combo. N_WORKERS parallel processes.
Checkpoint every CHECKPOINT_SECONDS to progress_latest.xlsx.
"""

import os
import time
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.signal import fftconvolve
from scipy.special import gammaln
from tqdm import tqdm
import multiprocessing as mp

# ============================================================
# CONFIG
# ============================================================
TEST_MODE = True
R_FULL = 2500
R_TEST = 5
N_WORKERS = 8
CHECKPOINT_SECONDS = 3600
OUTPUT_DIR = "."
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_latest.xlsx")

N_LIST = [10, 30]
M_LIST = [50, 80, 100, 200, 300]
FACTOR_LIST = [1, 2, 3, 4]
K_LIST = [0.2, 0.3, 0.5]   # break position tau* = (1+k)*M; also monitoring window

# Critical values a^2 from Wang & Hsiao (2013)
A2 = {"10%": 6.251389, "5%": 7.814728, "1%": 11.344867}
LEVELS = ["10%", "5%", "1%"]

# Exp3 subgroup definitions
# G_j: 0-based column indices of sequences influenced by factor j
EXP3_GROUPS_N10 = {2: [0,1,2], 3: [3,4,5], 4: [6,7,8,9]}
EXP3_GROUPS_N30 = {2: list(range(0,3)), 3: list(range(3,12)), 4: list(range(12,30))}

# Exp3.2 loadings
EXP32_GAMMA_PRE  = {2: 0.10, 3: 0.25, 4: 0.30}
EXP32_GAMMA_POST = {2: 0.60, 3: 0.85, 4: 0.90}


# ============================================================
# 0. Fractional differencing
# ============================================================
def fracdiff_coefs(d, n_max):
    if d == 0:
        w = np.zeros(n_max); w[0] = 1.0; return w
    j = np.arange(n_max)
    log_w = gammaln(j + d) - gammaln(j + 1) - gammaln(d)
    return np.exp(log_w)

def fracdiff_filter(Y, d):
    n = len(Y)
    if d == 0: return Y.copy()
    w = fracdiff_coefs(d, n)
    return fftconvolve(Y, w)[:n]

def gen_series(ar, d, ma, nsample):
    if ar == 0 and ma == 0 and d == 0:
        return np.random.normal(0, 1, nsample)
    ar_poly = np.array([1.0, -ar]) if ar != 0 else np.array([1.0])
    ma_poly = np.array([1.0,  ma]) if ma != 0 else np.array([1.0])
    Y = ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=nsample)
    return fracdiff_filter(Y, d)


# ============================================================
# 1. DGP
# ============================================================
def data_generation_N10(nsample=4000):
    specs = [
        (0.8,0.0,0.5),(0.8,0.0,0.5),(0.5,0.0,0.3),(0.5,0.0,0.3),
        (0.0,0.4,0.0),(0.0,0.4,0.0),(0.1,0.3,0.8),(0.1,0.3,0.8),
        (0.7,0.4,0.0),(0.7,0.4,0.0),
    ]
    return pd.DataFrame(
        {f"s{i+1}": gen_series(ar,d,ma,nsample) for i,(ar,d,ma) in enumerate(specs)}
    )

def data_generation_N30(nsample=4000):
    cols = {}
    for i in range(1,16):
        cols[f"s{i}"] = gen_series(np.random.uniform(0.0,0.5), 0.0, 0.6, nsample)
    for i in range(16,21):
        cols[f"s{i}"] = gen_series(np.random.uniform(0.7,0.9), 0.0, 0.6, nsample)
    for i in range(21,26):
        cols[f"s{i}"] = gen_series(0.0, np.random.uniform(0.0,0.5), 0.0, nsample)
    for i in range(26,31):
        cols[f"s{i}"] = gen_series(
            np.random.uniform(0.0,0.4), np.random.uniform(0.0,0.5),
            np.random.uniform(0.0,0.3), nsample)
    return pd.DataFrame(cols)


# ============================================================
# 2. Factor generation (with optional post-break noise std)
# ============================================================
def gen_factor(factor_id, nsample, tau_break=None, post_std=1.0):
    """
    Generate a single factor series.
    If tau_break is not None, the driving noise std changes from 1.0 to
    post_std at position tau_break (0-based index).
    For AR-based factors the noise injection is done by filtering the
    heteroskedastic noise through the AR recursion manually.
    """
    if tau_break is not None:
        n_pre  = tau_break
        n_post = nsample - tau_break
        noise_pre  = np.random.normal(0, 1.0,     n_pre)  if n_pre  > 0 else np.array([])
        noise_post = np.random.normal(0, post_std, n_post) if n_post > 0 else np.array([])
        noise = np.concatenate([noise_pre, noise_post])
    else:
        noise = np.random.normal(0, 1.0, nsample)

    if factor_id == 1:
        return noise

    elif factor_id == 2:
        # AR(1) with phi=0.99 driven by `noise` via direct recursion
        phi = 0.99
        Y = np.empty(nsample)
        Y[0] = noise[0]
        for t in range(1, nsample):
            Y[t] = phi * Y[t-1] + noise[t]
        return Y

    elif factor_id == 3:
        return fracdiff_filter(noise, 0.2)

    else:  # factor_id == 4
        return fracdiff_filter(noise, 0.45)


# ============================================================
# 3. Panel builder helpers
# ============================================================
def base_panel(N, T_total):
    """Raw idiosyncratic panel (no factor added)."""
    df = data_generation_N10(T_total) if N == 10 else data_generation_N30(T_total)
    return df.iloc[:T_total].reset_index(drop=True).values  # shape (T, N)

def add_factor_uniform(panel, factor_vals, gamma):
    """Add gamma * f_t to ALL columns."""
    return panel + gamma * factor_vals[:, None]

def add_factor_by_group(panel, factor_vals_dict, gamma_dict, groups):
    """
    Add factor contributions per subgroup.
    factor_vals_dict: {factor_id: array of length T}
    gamma_dict: {factor_id: scalar gamma}
    groups: {factor_id: list of col indices}
    """
    panel = panel.copy()
    for fid, cols in groups.items():
        panel[:, cols] += gamma_dict[fid] * factor_vals_dict[fid][:, None]
    return panel


# ============================================================
# 4. AR(p) filtering
# ============================================================
def rolling_ar_residuals_fixed(series, M):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = max(3, int((M - 1) // 100) * 3)
    if M <= p:
        raise ValueError(f"M={M} too small for AR order p={p}")
    y_cal  = series[p:M]
    X_cal  = np.column_stack([series[p-i-1:M-i-1] for i in range(p)])
    beta, *_ = np.linalg.lstsq(X_cal, y_cal, rcond=None)
    residuals = np.full(n, np.nan)
    idx = np.arange(7, n)
    X = np.column_stack([series[idx-1-i] for i in range(p)])
    residuals[idx] = series[idx] - X @ beta
    return residuals

def ar_filter_panel(panel_array, M):
    """panel_array: (T, N) ndarray. Returns (resid, n_dropped)."""
    resid = np.column_stack(
        [rolling_ar_residuals_fixed(panel_array[:, i], M) for i in range(panel_array.shape[1])]
    )
    valid = ~np.isnan(resid).any(axis=1)
    n_dropped = int(np.argmax(valid))
    return resid[valid], n_dropped


# ============================================================
# 5. Unpooled pairwise (i, j) regressions, closed-form aggregate
# ============================================================
def pairwise_cross_section_mean(resid_matrix, M):
    """
    For every ordered pair (i, j), i != j, fit a SEPARATE calibration-period
    OLS regression e_i,t = alpha_ij + beta_ij * e_j,t + u_ij,t  (t = 1..M).
    No stacking/pooling across pairs: each pair gets its own (alpha_ij,
    beta_ij). These N(N-1) fixed pairs of coefficients are estimated once
    from the calibration sample and then applied UNCHANGED to every t in
    the full sample (calibration + monitoring) -- exactly like the AR
    coefficients in Section 4 above. The signal fed into the CUSQ test is
    the simple average, at each t, of the N(N-1) pairwise residuals
    U_ij,t = e_i,t - alpha_ij - beta_ij * e_j,t.

    Naively this requires materializing N(N-1) residual series. The
    following closed form gives the exact average without ever forming
    them:

        mu_i     = calibration-period mean of series i                 (N,)
        f_{i,t}  = e_{i,t} - mu_i                       (centered, ALL t)
        S        = D^T D,  D = f restricted to t = 1..M                (N,N)
        beta_ij  = S[i,j] / S[j,j]                  (single-pair OLS slope)
        alpha_ij = mu_i - beta_ij * mu_j
        U_ij,t   = e_i,t - alpha_ij - beta_ij*e_j,t = f_i,t - beta_ij*f_j,t

        ebar_t = (1/(N(N-1))) * sum_{i!=j} U_ij,t
               = [ (N-1)*F_t - sum_j c_j * f_j,t ] / (N(N-1))

      where F_t = sum_i f_i,t and c_j = sum_{i!=j} beta_ij is the column
      sum of the beta matrix (diagonal excluded). This is exact (not an
      approximation) and costs O(N^2 M) once for S plus O(NT) for the
      time-series part, instead of O(N^2 T) for literal pairwise looping.

    Works for any N >= 2 columns of resid_matrix, including the N=3..18
    subgroup sizes used by cusum_subgroup() in Experiment 3 below.

    Returns
    -------
    e_t  : (T,) aggregate residual series used by the CUSQ test
    beta : (N, N) matrix of FIXED pairwise slopes, beta[i, j] = beta_ij
           (diagonal is 0 and unused; pair (i, i) does not exist)
    """
    N = resid_matrix.shape[1]
    mu = resid_matrix[:M, :].mean(axis=0)        # (N,) per-series calibration mean
    f = resid_matrix - mu                        # (T, N) centered with FIXED mu
    D = f[:M, :]                                 # (M, N) calibration-period centered
    S = D.T @ D                                  # (N, N)
    diag_S = np.diag(S).copy()
    if np.any(diag_S <= 0):
        raise ValueError("Degenerate (zero-variance) calibration series.")
    beta = S / diag_S[None, :]                   # beta[i, j] = S[i, j] / S[j, j]
    np.fill_diagonal(beta, 0.0)                  # (i, i) is not a valid pair

    c = beta.sum(axis=0)                         # (N,) c_j = sum_{i!=j} beta_ij
    F = f.sum(axis=1)                            # (T,) sum_i f_i,t
    e_t = ((N - 1) * F - f @ c) / (N * (N - 1))
    return e_t, beta


def _pairwise_cross_section_mean_bruteforce(resid_matrix, M):
    """Reference (slow, O(N^2) explicit OLS fits) implementation used only
    to validate the closed form above. Not used in the production pipeline."""
    T, N = resid_matrix.shape
    e_bar = np.zeros(T)
    npairs = 0
    for j in range(N):
        ej_cal = resid_matrix[:M, j]
        xj = np.column_stack([np.ones(M), ej_cal])
        for i in range(N):
            if i == j:
                continue
            ei_cal = resid_matrix[:M, i]
            (alpha_ij, beta_ij), *_ = np.linalg.lstsq(xj, ei_cal, rcond=None)
            e_bar += resid_matrix[:, i] - alpha_ij - beta_ij * resid_matrix[:, j]
            npairs += 1
    return e_bar / npairs


# ============================================================
# 6. CUSUM-of-squares + Brownian Bridge boundary
# ============================================================
def cusum_alarm_times(e_t, M):
    T = len(e_t)
    cumsum_sq = np.cumsum(e_t ** 2)
    Cm = cumsum_sq[M - 1]
    ns = np.arange(M + 1, T + 1, dtype=float)
    Cn = cumsum_sq[M:T]
    Dn = Cn / Cm - ns / M
    Tn = np.sqrt(M / 2.0) * Dn
    tau = ns / M
    alarms = {}
    for level, a2 in A2.items():
        g = np.sqrt(tau * (a2 + np.log(tau)))
        idx = np.where(np.abs(Tn) >= g)[0]
        alarms[level] = float(ns[idx[0]]) if len(idx) > 0 else np.inf
    return alarms


# ============================================================
# 7. Outcome metrics from a single alarm time
# ============================================================
def compute_metrics(tau_hat, tau_star):
    """
    Returns dict with alarm, premature, detect, delay.
    tau_hat and tau_star are in absolute time indices (>M).
    """
    if np.isinf(tau_hat):
        return dict(alarm=0, premature=0, detect=0, delay=np.nan)
    premature = int(tau_hat <= tau_star)
    detect    = int(tau_hat >  tau_star)
    delay     = float(tau_hat - tau_star) if detect else np.nan
    return dict(alarm=1, premature=premature, detect=detect, delay=delay)


def aggregate_metrics(results_list, R, levels=LEVELS):
    """
    results_list: list of R dicts, each keyed by level -> {alarm,premature,detect,delay}
    Returns flat dict of summary statistics per level.
    """
    out = {}
    for lvl in levels:
        suf = lvl.rstrip("%")
        alarm_r     = [r[lvl]["alarm"]     for r in results_list]
        premature_r = [r[lvl]["premature"] for r in results_list]
        detect_r    = [r[lvl]["detect"]    for r in results_list]
        delays      = np.array([r[lvl]["delay"] for r in results_list], dtype=float)
        delays      = delays[~np.isnan(delays)]

        out[f"AR_{suf}"]  = np.mean(alarm_r)
        out[f"PAR_{suf}"] = np.mean(premature_r)
        out[f"DR_{suf}"]  = np.mean(detect_r)
        if len(delays) > 0:
            out[f"delay_mean_{suf}"]   = delays.mean()
            out[f"delay_std_{suf}"]    = delays.std(ddof=1) if len(delays)>1 else np.nan
            out[f"delay_median_{suf}"] = np.median(delays)
            out[f"delay_q25_{suf}"]    = np.quantile(delays, 0.25)
            out[f"delay_q75_{suf}"]    = np.quantile(delays, 0.75)
        else:
            for s in ["mean","std","median","q25","q75"]:
                out[f"delay_{s}_{suf}"] = np.nan
    return out


# ============================================================
# 8. Experiment 1: Factor 4 volatility break
#    - All N series use factor 4 with gamma=0.5
#    - Break: factor4 noise std 1 -> 3 at tau* = (1+k)*M
# ============================================================
def run_exp1_single(N, m, k, R):
    tau_star_rel = int((1 + k) * m)
    T_total = 10 * m + 10

    results = []
    for _ in range(R):
        panel = base_panel(N, T_total)
        # Factor 4 with volatility break at tau_star_rel
        f4 = gen_factor(4, T_total, tau_break=tau_star_rel, post_std=np.sqrt(3.0))
        panel_obs = add_factor_uniform(panel, f4, gamma=0.5)

        resid, n_dropped = ar_filter_panel(panel_obs, m)
        tau_adj = tau_star_rel - n_dropped

        e_t, _ = pairwise_cross_section_mean(resid, m)
        alarms = cusum_alarm_times(e_t, m)

        rep = {lvl: compute_metrics(alarms[lvl], tau_adj) for lvl in LEVELS}
        results.append(rep)
    return results

def worker_exp1(combo):
    N, m, k = combo
    R = R_TEST if TEST_MODE else R_FULL
    results = run_exp1_single(N, m, k, R)
    row = {"N": N, "m": m, "k": k}
    row.update(aggregate_metrics(results, R))
    return ("exp1", combo, [row])


# ============================================================
# 9. Experiment 2: Factor loading break
#    - gamma: 0.1 -> 0.8 at tau* = (1+k)*M
#    - All N series, each factor j in {1,2,3,4} separately
# ============================================================
def run_exp2_single(N, m, k, factor_id, R):
    tau_star_rel = int((1 + k) * m)
    T_total = 10 * m + 10

    results = []
    for _ in range(R):
        panel = base_panel(N, T_total)
        f = gen_factor(factor_id, T_total)  # no noise break; loading changes instead

        # Build panel: pre-break gamma=0.1, post-break gamma=0.8
        panel_obs = panel.copy()
        panel_obs[:tau_star_rel, :]  += 0.1 * f[:tau_star_rel,  None]
        panel_obs[tau_star_rel:,  :] += 0.8 * f[tau_star_rel:,  None]

        resid, n_dropped = ar_filter_panel(panel_obs, m)
        tau_adj = tau_star_rel - n_dropped

        e_t, _ = pairwise_cross_section_mean(resid, m)
        alarms = cusum_alarm_times(e_t, m)

        rep = {lvl: compute_metrics(alarms[lvl], tau_adj) for lvl in LEVELS}
        results.append(rep)
    return results

def worker_exp2(combo):
    N, m, k, factor_id = combo
    R = R_TEST if TEST_MODE else R_FULL
    results = run_exp2_single(N, m, k, factor_id, R)
    row = {"N": N, "m": m, "k": k, "factor": factor_id}
    row.update(aggregate_metrics(results, R))
    return ("exp2", combo, [row])


# ============================================================
# 10. Experiment 3 helpers: subgroup CUSUM
# ============================================================
def cusum_subgroup(resid_full, m, col_indices):
    """Run pooled CUSUM on a subset of columns of resid_full."""
    sub = resid_full[:, col_indices]
    e_t, _ = pairwise_cross_section_mean(sub, m)
    return cusum_alarm_times(e_t, m)

def groups_for_N(N):
    return EXP3_GROUPS_N10 if N == 10 else EXP3_GROUPS_N30


# ============================================================
# 11. Experiment 3.1: Volatility break, subgroup reporting
#     gamma=0.5 uniform, break at tau*=1.2M (k=0.2 fixed as break position)
#     k in {0.2,0.3,0.5} as monitoring window
# ============================================================
def run_exp31_single(N, m, k, R):
    tau_star_rel = int((1 + k) * m)   # break position varies with k
    T_total = 10 * m + 10
    groups = groups_for_N(N)

    results_full = []
    results_sub  = {fid: [] for fid in [2,3,4]}

    for _ in range(R):
        panel = base_panel(N, T_total)
        panel_obs = panel.copy()
        for fid, cols in groups.items():
            f = gen_factor(fid, T_total, tau_break=tau_star_rel, post_std=np.sqrt(3.0))
            panel_obs[:, cols] += 0.5 * f[:, None]

        resid, n_dropped = ar_filter_panel(panel_obs, m)
        tau_adj = tau_star_rel - n_dropped

        e_t, _ = pairwise_cross_section_mean(resid, m)
        alarms_full = cusum_alarm_times(e_t, m)
        rep_full = {lvl: compute_metrics(alarms_full[lvl], tau_adj) for lvl in LEVELS}
        results_full.append(rep_full)

        for fid, cols in groups.items():
            alarms_sub = cusum_subgroup(resid, m, cols)
            rep_sub = {lvl: compute_metrics(alarms_sub[lvl], tau_adj) for lvl in LEVELS}
            results_sub[fid].append(rep_sub)

    row = {"N": N, "m": m, "k": k}
    mfull = aggregate_metrics(results_full, R)
    row.update({f"full_{kk}": vv for kk, vv in mfull.items()})
    for fid in [2,3,4]:
        msub = aggregate_metrics(results_sub[fid], R)
        row.update({f"g{fid}_{kk}": vv for kk, vv in msub.items()})
    for (fa, fb) in [(2,3),(3,4),(2,4)]:
        row[f"DeltaDR_g{fa}_g{fb}_5"] = row[f"g{fa}_DR_5"] - row[f"g{fb}_DR_5"]
    return [row]

def worker_exp31(combo):
    N, m, k = combo
    R = R_TEST if TEST_MODE else R_FULL
    rows = run_exp31_single(N, m, k, R)
    return ("exp31", combo, rows)


# ============================================================
# 12. Experiment 3.2: Loading break, subgroup reporting
#     break at tau*=1.2M (k=0.2 fixed); k in {0.2,0.3,0.5} monitoring window
#     pre-break: gamma per group = {2:0.1, 3:0.25, 4:0.3}
#     post-break: gamma per group = {2:0.6, 3:0.85, 4:0.9}
# ============================================================
def run_exp32_single(N, m, k, R):
    tau_star_rel = int((1 + k) * m)   # break position varies with k
    T_total = 10 * m + 10
    groups = groups_for_N(N)

    results_full = []
    results_sub  = {fid: [] for fid in [2,3,4]}

    for _ in range(R):
        panel = base_panel(N, T_total)
        panel_obs = panel.copy()

        for fid, cols in groups.items():
            f = gen_factor(fid, T_total)
            g_pre  = EXP32_GAMMA_PRE[fid]
            g_post = EXP32_GAMMA_POST[fid]
            panel_obs[:tau_star_rel, cols] += g_pre  * f[:tau_star_rel, None]
            panel_obs[tau_star_rel:, cols] += g_post * f[tau_star_rel:, None]

        resid, n_dropped = ar_filter_panel(panel_obs, m)
        tau_adj = tau_star_rel - n_dropped

        e_t, _ = pairwise_cross_section_mean(resid, m)
        alarms_full = cusum_alarm_times(e_t, m)
        rep_full = {lvl: compute_metrics(alarms_full[lvl], tau_adj) for lvl in LEVELS}
        results_full.append(rep_full)

        for fid, cols in groups.items():
            alarms_sub = cusum_subgroup(resid, m, cols)
            rep_sub = {lvl: compute_metrics(alarms_sub[lvl], tau_adj) for lvl in LEVELS}
            results_sub[fid].append(rep_sub)

    row = {"N": N, "m": m, "k": k}
    mfull = aggregate_metrics(results_full, R)
    row.update({f"full_{kk}": vv for kk, vv in mfull.items()})
    for fid in [2,3,4]:
        msub = aggregate_metrics(results_sub[fid], R)
        row.update({f"g{fid}_{kk}": vv for kk, vv in msub.items()})
    for (fa, fb) in [(2,3),(3,4),(2,4)]:
        row[f"DeltaDR_g{fa}_g{fb}_5"] = row[f"g{fa}_DR_5"] - row[f"g{fb}_DR_5"]
    return [row]

def worker_exp32(combo):
    N, m, k = combo
    R = R_TEST if TEST_MODE else R_FULL
    rows = run_exp32_single(N, m, k, R)
    return ("exp32", combo, rows)


# ============================================================
# 13. Checkpoint writer
# ============================================================
def write_progress(rows_dict):
    sheet_map = {
        "exp1":  "exp1_factor4_vol",
        "exp2":  "exp2_loading",
        "exp31": "exp31_weak_vol",
        "exp32": "exp32_weak_loading",
    }
    with pd.ExcelWriter(PROGRESS_FILE, engine="openpyxl") as writer:
        for key, sheet in sheet_map.items():
            df = pd.DataFrame(rows_dict.get(key, []))
            df.to_excel(writer, sheet_name=sheet, index=False)


# ============================================================
# 14. Main driver
# ============================================================
def _run_task(task):
    kind, combo = task
    if kind == "exp1":  return worker_exp1(combo)
    if kind == "exp2":  return worker_exp2(combo)
    if kind == "exp31": return worker_exp31(combo)
    if kind == "exp32": return worker_exp32(combo)

def worker_init():
    seed = int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)

def main():
    combos_exp1  = list(itertools.product(N_LIST, M_LIST, K_LIST))
    combos_exp2  = list(itertools.product(N_LIST, M_LIST, K_LIST, FACTOR_LIST))
    combos_exp31 = list(itertools.product(N_LIST, M_LIST, K_LIST))
    combos_exp32 = list(itertools.product(N_LIST, M_LIST, K_LIST))

    if TEST_MODE:
        combos_exp1  = combos_exp1[:2]
        combos_exp2  = combos_exp2[:2]
        combos_exp31 = combos_exp31[:2]
        combos_exp32 = combos_exp32[:2]

    tasks = (
        [("exp1",  c) for c in combos_exp1]  +
        [("exp2",  c) for c in combos_exp2]  +
        [("exp31", c) for c in combos_exp31] +
        [("exp32", c) for c in combos_exp32]
    )
    total = len(tasks)
    R = R_TEST if TEST_MODE else R_FULL
    print(f"Total tasks: {total}  R={R}  TEST_MODE={TEST_MODE}")
    print(f"  exp1:{len(combos_exp1)}  exp2:{len(combos_exp2)}"
          f"  exp31:{len(combos_exp31)}  exp32:{len(combos_exp32)}")

    rows = {"exp1": [], "exp2": [], "exp31": [], "exp32": []}
    last_checkpoint = time.time()

    with mp.Pool(processes=N_WORKERS, initializer=worker_init) as pool:
        results_iter = pool.imap_unordered(_run_task, tasks, chunksize=1)
        with tqdm(total=total, desc="combos") as pbar:
            for kind, combo, new_rows in results_iter:
                rows[kind].extend(new_rows)
                pbar.update(1)
                print(f"  [done] {kind}  combo={combo}")

                now = time.time()
                if now - last_checkpoint >= CHECKPOINT_SECONDS:
                    write_progress(rows)
                    print(f"  [checkpoint] {datetime.now():%Y-%m-%d %H:%M:%S}")
                    last_checkpoint = now

    write_progress(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(OUTPUT_DIR, f"final_results_{ts}.xlsx")
    sheet_map = {
        "exp1_factor4_vol": rows["exp1"],
        "exp2_loading":     rows["exp2"],
        "exp31_weak_vol":   rows["exp31"],
        "exp32_weak_loading": rows["exp32"],
    }
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        for sheet, data in sheet_map.items():
            pd.DataFrame(data).to_excel(writer, sheet_name=sheet, index=False)
    print(f"Done. Results -> {final_path}")


if __name__ == "__main__":
    main()
