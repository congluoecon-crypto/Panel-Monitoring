"""
20260617_pairwise_bentest.py
============================
Implementation of recursive (rolling) pairwise/pooled OLS residual aggregation
for Wang & Hsiao (2013) CUSUM-of-squares structural break monitoring.

Two strategies (Pairwise OLS, Pooled OLS) are computed simultaneously per
simulation path. Both sequential (ever-alarm) and pointwise (at kM)
statistics are reported for Scenario A (size) and Scenario B (power).

Set TEST_MODE=True for a quick smoke run.
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
R_TEST = 3
N_WORKERS = 8
CHECKPOINT_SECONDS = 3600
OUTPUT_DIR = "."
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_latest.xlsx")

N_LIST = [10, 30]
M_LIST = [50, 80, 100, 200, 300]
FACTOR_LIST = [1, 2, 3, 4]
DELTA_LIST = [1, 2, 3]
K_LIST = [0.2, 0.3, 0.5, 1, 2, 3]

A2 = {"10%": 6.2514, "5%": 7.8147, "1%": 11.3449}
LEVELS = ["10%", "5%", "1%"]

# ============================================================
# 0. Fractional differencing & DGP (unchanged from original)
# ============================================================
def fracdiff_coefs(d, n_max):
    if d == 0:
        w = np.zeros(n_max); w[0] = 1.0; return w
    j = np.arange(n_max)
    return np.exp(gammaln(j + d) - gammaln(j + 1) - gammaln(d))

def fracdiff_filter(Y, d):
    if d == 0: return Y.copy()
    return fftconvolve(Y, fracdiff_coefs(d, len(Y)))[:len(Y)]

def gen_series(ar, d, ma, nsample):
    if ar == 0 and ma == 0 and d == 0:
        return np.random.normal(0, 1, nsample)
    ar_poly = np.array([1.0, -ar]) if ar != 0 else np.array([1.0])
    ma_poly = np.array([1.0, ma]) if ma != 0 else np.array([1.0])
    Y = ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=nsample)
    return fracdiff_filter(Y, d)

def data_generation_N10(nsample=4000):
    specs = [(0.8,0.0,0.5), (0.8,0.0,0.5), (0.5,0.0,0.3), (0.5,0.0,0.3),
             (0.0,0.4,0.0), (0.0,0.4,0.0), (0.1,0.3,0.8), (0.1,0.3,0.8),
             (0.7,0.4,0.0), (0.7,0.4,0.0)]
    return pd.DataFrame({f"s{i+1}": gen_series(ar,d,ma,nsample) for i,(ar,d,ma) in enumerate(specs)})

def data_generation_N30(nsample=4000):
    cols = {}
    for i in range(1, 16):
        cols[f"s{i}"] = gen_series(np.random.uniform(0.0,0.5), 0.0, 0.6, nsample)
    for i in range(16, 21):
        cols[f"s{i}"] = gen_series(np.random.uniform(0.7,0.9), 0.0, 0.6, nsample)
    for i in range(21, 26):
        cols[f"s{i}"] = gen_series(0.0, np.random.uniform(0.0,0.5), 0.0, nsample)
    for i in range(26, 31):
        cols[f"s{i}"] = gen_series(np.random.uniform(0.0,0.4), np.random.uniform(0.0,0.5),
                                   np.random.uniform(0.0,0.3), nsample)
    return pd.DataFrame(cols)

def common_factor_generation(nsample=4000):
    return pd.DataFrame({
        "factor1": np.random.normal(0,1,nsample),
        "factor2": gen_series(0.99,0.0,0.0,nsample),
        "factor3": gen_series(0.0,0.2,0.0,nsample),
        "factor4": gen_series(0.0,0.45,0.0,nsample)
    })

# ============================================================
# 1. Fixed-coefficient AR filtering (unchanged)
# ============================================================
def rolling_ar_residuals_fixed(series, M):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = max(3, int((M - 1) // 100) * 3)
    if M <= p: raise ValueError(f"M={M} too small for AR order p={p}")
    y_cal = series[p:M]
    X_cal = np.column_stack([series[p-i-1:M-i-1] for i in range(p)])
    beta, *_ = np.linalg.lstsq(X_cal, y_cal, rcond=None)
    residuals = np.full(n, np.nan)
    idx = np.arange(max(7, p), n)
    X = np.column_stack([series[idx-1-i] for i in range(p)])
    residuals[idx] = series[idx] - X @ beta
    return residuals

# ============================================================
# 2. Core: calibration-fixed beta (-> Cm) + monitoring-recursive beta
#    (t>M, expanding window 1..t), separately for Pairwise and Pooled OLS
# ============================================================
def compute_pairwise_cusq_inputs(resid, M):
    """
    Pairwise OLS. Calibration period (t=1..M): ONE fixed beta_ij, estimated
    once on the M calibration observations, gives the calibration aggregate
    residual series e_1,...,e_M and hence Cm = sum_{t=1}^M e_t^2.
    Monitoring period (t=M+1..T): beta_ij(t) is RE-ESTIMATED at every
    period using the expanding window 1..t (true recursive OLS), giving the
    monitoring aggregate residual e_t for t=M+1,...,T (NaN for t<=M).
    Returns (Cm, e_mon).
    """
    T, N = resid.shape

    # --- calibration-only fixed beta ---
    resid_cal = resid[:M, :]
    mu_cal = resid_cal.mean(axis=0)
    d_cal = resid_cal - mu_cal
    S_cal = d_cal.T @ d_cal
    diagS_cal = np.diag(S_cal)
    beta_cal = S_cal / diagS_cal[None, :]
    np.fill_diagonal(beta_cal, 0.0)
    cj_cal = beta_cal.sum(axis=0)
    F_cal = d_cal.sum(axis=1)
    e_cal = ((N - 1) * F_cal - d_cal @ cj_cal) / (N * (N - 1))
    Cm = float(np.sum(e_cal ** 2))

    # --- recursive monitoring beta(t), expanding window 1..t, t=M+1..T ---
    e_mon = np.full(T, np.nan)
    sum_x = resid[:M, :].sum(axis=0)
    sum_xx = resid[:M, :].T @ resid[:M, :]
    for t in range(M + 1, T + 1):
        x_t = resid[t - 1, :]
        sum_x = sum_x + x_t
        sum_xx = sum_xx + np.outer(x_t, x_t)
        mu_t = sum_x / t
        S_t = sum_xx - t * np.outer(mu_t, mu_t)
        diag_t = np.diag(S_t)
        beta_t = S_t / diag_t[None, :]
        np.fill_diagonal(beta_t, 0.0)
        c_t = beta_t.sum(axis=0)
        f_t = x_t - mu_t
        F_t = f_t.sum()
        e_mon[t - 1] = ((N - 1) * F_t - f_t @ c_t) / (N * (N - 1))

    return Cm, e_mon

def compute_pooled_cusq_inputs(resid, M):
    """Pooled OLS. Same calibration-fixed / monitoring-recursive split as
    compute_pairwise_cusq_inputs, but using the common-slope pooled
    estimator beta_pool = sum_{i!=j} S_ij / [(N-1)*tr(S)] at every step
    (calibration: fitted once on 1..M; monitoring: re-fitted at every t
    using the expanding window 1..t)."""
    T, N = resid.shape

    resid_cal = resid[:M, :]
    mu_cal = resid_cal.mean(axis=0)
    d_cal = resid_cal - mu_cal
    S_cal = d_cal.T @ d_cal
    diagS_cal = np.diag(S_cal)
    num_cal = S_cal.sum() - np.trace(S_cal)
    den_cal = (N - 1) * diagS_cal.sum()
    beta_pool_cal = num_cal / den_cal if den_cal > 1e-12 else 0.0
    c_pool_cal = (N - 1) * beta_pool_cal
    F_cal = d_cal.sum(axis=1)
    e_cal = ((N - 1) * F_cal - c_pool_cal * F_cal) / (N * (N - 1))
    Cm = float(np.sum(e_cal ** 2))

    e_mon = np.full(T, np.nan)
    sum_x = resid[:M, :].sum(axis=0)
    sum_xx = resid[:M, :].T @ resid[:M, :]
    for t in range(M + 1, T + 1):
        x_t = resid[t - 1, :]
        sum_x = sum_x + x_t
        sum_xx = sum_xx + np.outer(x_t, x_t)
        mu_t = sum_x / t
        S_t = sum_xx - t * np.outer(mu_t, mu_t)
        diag_t = np.diag(S_t)
        num_t = S_t.sum() - np.trace(S_t)
        den_t = (N - 1) * diag_t.sum()
        beta_pool_t = num_t / den_t if den_t > 1e-12 else 0.0
        c_pool_t = (N - 1) * beta_pool_t
        f_t = x_t - mu_t
        F_t = f_t.sum()
        e_mon[t - 1] = ((N - 1) * F_t - c_pool_t * F_t) / (N * (N - 1))

    return Cm, e_mon

# ============================================================
# 3. CUSUM-of-squares statistic (Wang & Hsiao, 2013)
# ============================================================
def cusum_alarm_times(Cm, e_mon, M):
    T = len(e_mon)
    e_sq = np.where(np.isnan(e_mon), 0.0, e_mon ** 2)
    cumsum_mon = np.cumsum(e_sq)        # 0 contribution for t<=M
    Cn = Cm + cumsum_mon                # C_n = C_m + sum_{t=m+1}^n e_t^2
    ns = np.arange(M+1, T+1, dtype=float)
    Cn_used = Cn[M:T]
    Dn = Cn_used / Cm - (ns - 2) / (M - 2)
    Tn = np.sqrt(M / 2.0) * Dn
    ratio = (ns - M) / M
    inner = ns / (ns - M)

    alarms = {}
    for level, a2 in A2.items():
        g = ratio * np.sqrt(inner * (a2 + np.log(inner)))
        idx = np.where(np.abs(Tn) >= g)[0]
        alarms[level] = ns[idx[0]] if len(idx) > 0 else np.inf
    return alarms

# ============================================================
# 4. Pipeline: build panel, AR filter, compute both e_t
# ============================================================
def build_panel(N, factor, T_total):
    df_raw = data_generation_N10() if N == 10 else data_generation_N30()
    cf = common_factor_generation()
    df = df_raw + 0.5 * cf[f"factor{factor}"].values[:, None]
    return df.iloc[:T_total].reset_index(drop=True)

def ar_filter_panel(df, M):
    resid = np.column_stack([rolling_ar_residuals_fixed(df[c].values, M) for c in df.columns])
    valid = ~np.isnan(resid).any(axis=1)
    n_dropped = int(np.argmax(valid))
    return resid[valid], n_dropped

def inject_break(df, tau, delta):
    df2 = df.copy()
    df2.iloc[tau:] = df2.iloc[tau:] + delta
    return df2

# ============================================================
# 5. Scenario A (delta=0) and Scenario B (delta>0)
# ============================================================
def simulate_one_path(N, m, factor, delta, k, T_total=None):
    if T_total is None: T_total = 10 * m + 10
    df = build_panel(N, factor, T_total)

    if delta > 0:
        tau_pos = m + int(k * m)
        df = inject_break(df, tau_pos, delta)

    resid, n_dropped = ar_filter_panel(df, m)
    Cm_pw, e_mon_pw = compute_pairwise_cusq_inputs(resid, m)
    Cm_pl, e_mon_pl = compute_pooled_cusq_inputs(resid, m)

    # Adjust tau position for dropped initial observations (if any)
    tau_adj = (m + int(k * m) if delta > 0 else np.inf) - n_dropped

    alarms_pairwise = cusum_alarm_times(Cm_pw, e_mon_pw, m)
    alarms_pooled   = cusum_alarm_times(Cm_pl, e_mon_pl, m)

    # Helper to extract results for a given alarms dict
    def extract_results(alarms, is_B=False):
        out_seq = {}
        out_point = {}
        for level, tau_hat in alarms.items():
            # Sequential (ever-alarm over full horizon)
            if np.isinf(tau_hat):
                out_seq[level] = {'alarm':0, 'premature':0, 'detect':0, 'delay':np.nan}
            else:
                if is_B:
                    premature = int(tau_hat <= tau_adj)
                    detect = int(tau_hat > tau_adj)
                    delay = (tau_hat - tau_adj) if detect else np.nan
                    out_seq[level] = {'alarm':1, 'premature':premature, 'detect':detect, 'delay':delay}
                else:
                    out_seq[level] = {'alarm': int(np.isfinite(tau_hat))}

            # Pointwise (at each k in K_LIST, has alarm occurred by m + k*m?)
            for k0 in K_LIST:
                out_point[(level, k0)] = int(np.isfinite(tau_hat) and tau_hat <= m + k0*m)
        return out_seq, out_point

    res_pairwise_seq, res_pairwise_point = extract_results(alarms_pairwise, is_B=(delta>0))
    res_pooled_seq, res_pooled_point = extract_results(alarms_pooled, is_B=(delta>0))

    return {
        'pairwise_seq': res_pairwise_seq,
        'pairwise_point': res_pairwise_point,
        'pooled_seq': res_pooled_seq,
        'pooled_point': res_pooled_point
    }

# ============================================================
# 6. Worker functions for parallel execution
# ============================================================
def worker_init():
    seed = int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)

def run_combo(combo):
    kind, N, m, factor, delta, k = combo
    R = R_TEST if TEST_MODE else R_FULL

    # Accumulators for Scenario A (delta=0)
    if delta == 0:
        sum_pairwise_seq = {lvl: 0 for lvl in LEVELS}
        sum_pairwise_point = {(lvl, k0): 0 for lvl in LEVELS for k0 in K_LIST}
        sum_pooled_seq = {lvl: 0 for lvl in LEVELS}
        sum_pooled_point = {(lvl, k0): 0 for lvl in LEVELS for k0 in K_LIST}
        for _ in range(R):
            out = simulate_one_path(N, m, factor, 0, None)
            for lvl in LEVELS:
                sum_pairwise_seq[lvl] += out['pairwise_seq'][lvl]['alarm']
                sum_pooled_seq[lvl] += out['pooled_seq'][lvl]['alarm']
                for k0 in K_LIST:
                    sum_pairwise_point[(lvl, k0)] += out['pairwise_point'][(lvl, k0)]
                    sum_pooled_point[(lvl, k0)] += out['pooled_point'][(lvl, k0)]

        rows = {}
        # Sequential
        row_seq = {"N": N, "m": m, "factor": factor}
        for lvl in LEVELS:
            row_seq[f"size_{lvl.rstrip('%')}"] = sum_pairwise_seq[lvl] / R
        rows['A_pairwise_seq'] = [row_seq]
        row_seq = {"N": N, "m": m, "factor": factor}
        for lvl in LEVELS:
            row_seq[f"size_{lvl.rstrip('%')}"] = sum_pooled_seq[lvl] / R
        rows['A_pooled_seq'] = [row_seq]

        # Pointwise
        rows['A_pairwise_pointwise'] = []
        rows['A_pooled_pointwise'] = []
        for k0 in K_LIST:
            rw_pw = {"N": N, "m": m, "factor": factor, "k": k0}
            rw_pl = {"N": N, "m": m, "factor": factor, "k": k0}
            for lvl in LEVELS:
                rw_pw[f"point_alarm_{lvl.rstrip('%')}"] = sum_pairwise_point[(lvl, k0)] / R
                rw_pl[f"point_alarm_{lvl.rstrip('%')}"] = sum_pooled_point[(lvl, k0)] / R
            rows['A_pairwise_pointwise'].append(rw_pw)
            rows['A_pooled_pointwise'].append(rw_pl)
        return ('A', combo, rows)

    # Scenario B (delta > 0)
    else:
        agg_pw_seq = {lvl: {'alarm':0, 'premature':0, 'detect':0, 'delays':[]} for lvl in LEVELS}
        agg_pl_seq = {lvl: {'alarm':0, 'premature':0, 'detect':0, 'delays':[]} for lvl in LEVELS}
        agg_pw_point = {(lvl, k0): 0 for lvl in LEVELS for k0 in K_LIST}
        agg_pl_point = {(lvl, k0): 0 for lvl in LEVELS for k0 in K_LIST}

        for _ in range(R):
            out = simulate_one_path(N, m, factor, delta, k)
            for lvl in LEVELS:
                # Pairwise sequential
                d_pw = out['pairwise_seq'][lvl]
                agg_pw_seq[lvl]['alarm'] += d_pw['alarm']
                agg_pw_seq[lvl]['premature'] += d_pw['premature']
                agg_pw_seq[lvl]['detect'] += d_pw['detect']
                if d_pw['detect'] == 1:
                    agg_pw_seq[lvl]['delays'].append(d_pw['delay'])
                # Pooled sequential
                d_pl = out['pooled_seq'][lvl]
                agg_pl_seq[lvl]['alarm'] += d_pl['alarm']
                agg_pl_seq[lvl]['premature'] += d_pl['premature']
                agg_pl_seq[lvl]['detect'] += d_pl['detect']
                if d_pl['detect'] == 1:
                    agg_pl_seq[lvl]['delays'].append(d_pl['delay'])
                # Pointwise (for both, at all k0)
                for k0 in K_LIST:
                    agg_pw_point[(lvl, k0)] += out['pairwise_point'][(lvl, k0)]
                    agg_pl_point[(lvl, k0)] += out['pooled_point'][(lvl, k0)]

        rows = {}
        # Sequential
        for suffix, agg_seq in [('pairwise', agg_pw_seq), ('pooled', agg_pl_seq)]:
            row = {"N": N, "delta": delta, "k": k, "m": m, "factor": factor}
            for lvl in LEVELS:
                s = lvl.rstrip('%')
                d = agg_seq[lvl]
                row[f"alarmrate_{s}"] = d['alarm'] / R
                row[f"prematurerate_{s}"] = d['premature'] / R
                row[f"detectrate_{s}"] = d['detect'] / R
                delays = np.array(d['delays'], dtype=float)
                if len(delays) > 0:
                    row[f"delay_mean_{s}"] = delays.mean()
                    row[f"delay_std_{s}"] = delays.std(ddof=1) if len(delays)>1 else np.nan
                    row[f"delay_median_{s}"] = np.median(delays)
                    row[f"delay_q25_{s}"] = np.quantile(delays, 0.25)
                    row[f"delay_q75_{s}"] = np.quantile(delays, 0.75)
                else:
                    for stat in ['mean','std','median','q25','q75']:
                        row[f"delay_{stat}_{s}"] = np.nan
            rows[f'B_{suffix}_seq'] = [row]

        # Pointwise (only the k corresponding to break position, but we output all k0 for completeness)
        for suffix, agg_point in [('pairwise', agg_pw_point), ('pooled', agg_pl_point)]:
            rows[f'B_{suffix}_pointwise'] = []
            for k0 in K_LIST:
                row = {"N": N, "delta": delta, "k_break": k, "k": k0, "m": m, "factor": factor}
                for lvl in LEVELS:
                    row[f"point_alarm_{lvl.rstrip('%')}"] = agg_point[(lvl, k0)] / R
                rows[f'B_{suffix}_pointwise'].append(row)

        return ('B', combo, rows)

# ============================================================
# 7. Checkpoint & Main
# ============================================================
def write_progress(rows_dict):
    with pd.ExcelWriter(PROGRESS_FILE, engine='openpyxl') as writer:
        for sheet, rows in rows_dict.items():
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

def main():
    combos_A = [(N, m, factor, 0, None) for N in N_LIST for m in M_LIST for factor in FACTOR_LIST]
    combos_B = [(N, m, factor, delta, k) for N in N_LIST for m in M_LIST for factor in FACTOR_LIST
                for delta in DELTA_LIST for k in K_LIST]
    if TEST_MODE:
        combos_A = combos_A[:4]
        combos_B = combos_B[:4]

    tasks = [('A',) + c for c in combos_A] + [('B',) + c for c in combos_B]
    total = len(tasks)
    R = R_TEST if TEST_MODE else R_FULL
    print(f"Total combos: {total} (A: {len(combos_A)}, B: {len(combos_B)}), R={R}")

    # We'll collect results per sheet
    all_rows = {
        'A_pairwise_seq': [], 'A_pairwise_pointwise': [],
        'A_pooled_seq': [], 'A_pooled_pointwise': [],
        'B_pairwise_seq': [], 'B_pairwise_pointwise': [],
        'B_pooled_seq': [], 'B_pooled_pointwise': []
    }
    last_checkpoint = time.time()

    with mp.Pool(processes=N_WORKERS, initializer=worker_init) as pool:
        results = pool.imap_unordered(_run_task_wrapper, tasks, chunksize=1)
        with tqdm(total=total, desc="combos") as pbar:
            for result in results:
                kind, combo, rows = result
                for sheet, rows_sheet in rows.items():
                    all_rows[sheet].extend(rows_sheet)
                pbar.update(1)
                print(f"  [done] {kind} combo={combo}")

                if time.time() - last_checkpoint >= CHECKPOINT_SECONDS:
                    write_progress(all_rows)
                    print(f"  [checkpoint] wrote {PROGRESS_FILE}")
                    last_checkpoint = time.time()

    write_progress(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(OUTPUT_DIR, f"final_results_{ts}.xlsx")
    with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
        for sheet, rows in all_rows.items():
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
    print(f"Done. Final: {final_path}")

def _run_task_wrapper(task):
    kind, N, m, factor, delta, k = task
    return run_combo((kind, N, m, factor, delta, k))

if __name__ == "__main__":
    main()