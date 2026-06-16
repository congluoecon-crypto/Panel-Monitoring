"""
20260613_full_simulation.py
============================
Full production run of the revised Brownian Bridge structural break
monitoring framework (Scenario A = size, Scenario B = power).

- Scenario A: delta = 0. For each (N, m, factor) combo, run R=2500
  replications; k in {0.2,0.3,0.5,1,2,3} is a post-hoc scoring window
  M+kM, scored from a single simulated path.
- Scenario B: delta in {1,2,3}, k determines the break position
  tau(k)=M+kM. For each (N, delta, k, m, factor) combo, run R=2500
  replications, recording overall alarm / premature alarm (before tau) /
  post-break detection (after tau) / detection delay (conditional on
  detection).

Execution:
  - 8 worker processes (multiprocessing.Pool), each combo = 1 task.
  - Global tqdm progress bar over all combos (Scenario A: 40, Scenario B:
    720 -> 760 total), plus a print line each time a combo finishes.
  - Every 3600 seconds, overwrite `progress_latest.xlsx` with the results
    of all FULLY COMPLETED combos so far (two sheets: scenario_A,
    scenario_B). Partially-completed combos are not included.
  - At the end, write a timestamped final results file (not overwritten)
    in addition to the last progress_latest.xlsx.

Set TEST_MODE=True below for a quick end-to-end smoke run (tiny R and a
reduced combo grid) to validate the pipeline before committing to the
full run.
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
TEST_MODE = True          # <-- set to False for the real 2500-rep run
R_FULL = 2500
R_TEST = 3
N_WORKERS = 8
CHECKPOINT_SECONDS = 3600
OUTPUT_DIR = "."           # change to your desired output folder
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress_latest.xlsx")

N_LIST = [10, 30]
M_LIST = [50, 80, 100, 200, 300]
FACTOR_LIST = [1, 2, 3, 4]
DELTA_LIST = [1, 2, 3]
K_LIST = [0.2, 0.3, 0.5, 1, 2, 3]

# Critical values a^2 from Wang & Hsiao (2013) formula [5], p.6.
# Paper states a^2_{5%} = 7.78. Exact numerical solutions below.
A2 = {"10%": 6.2514, "5%": 7.8147, "1%": 11.3449}
LEVELS = ["10%", "5%", "1%"]


# ============================================================
# 0. Fractional differencing via FFT
# ============================================================
def fracdiff_coefs(d, n_max):
    if d == 0:
        w = np.zeros(n_max)
        w[0] = 1.0
        return w
    j = np.arange(n_max)
    log_w = gammaln(j + d) - gammaln(j + 1) - gammaln(d)
    return np.exp(log_w)


def fracdiff_filter(Y, d):
    n = len(Y)
    if d == 0:
        return Y.copy()
    w = fracdiff_coefs(d, n)
    return fftconvolve(Y, w)[:n]


def gen_series(ar, d, ma, nsample):
    if ar == 0 and ma == 0 and d == 0:
        return np.random.normal(0, 1, nsample)
    ar_poly = np.array([1.0, -ar]) if ar != 0 else np.array([1.0])
    ma_poly = np.array([1.0, ma]) if ma != 0 else np.array([1.0])
    Y = ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=nsample)
    return fracdiff_filter(Y, d)


# ============================================================
# 1. DGP -- Table 3.1 / Table 3.2
# ============================================================
def data_generation_N10(nsample=4000):
    specs = [
        (0.8, 0.0, 0.5), (0.8, 0.0, 0.5),
        (0.5, 0.0, 0.3), (0.5, 0.0, 0.3),
        (0.0, 0.4, 0.0), (0.0, 0.4, 0.0),
        (0.1, 0.3, 0.8), (0.1, 0.3, 0.8),
        (0.7, 0.4, 0.0), (0.7, 0.4, 0.0),
    ]
    data = {f"s{i+1}": gen_series(ar, d, ma, nsample)
            for i, (ar, d, ma) in enumerate(specs)}
    return pd.DataFrame(data)


def data_generation_N30(nsample=4000):
    cols = {}
    for i in range(1, 16):
        ar = np.random.uniform(0.0, 0.5)
        cols[f"s{i}"] = gen_series(ar, 0.0, 0.6, nsample)
    for i in range(16, 21):
        ar = np.random.uniform(0.7, 0.9)
        cols[f"s{i}"] = gen_series(ar, 0.0, 0.6, nsample)
    for i in range(21, 26):
        d = np.random.uniform(0.0, 0.5)
        cols[f"s{i}"] = gen_series(0.0, d, 0.0, nsample)
    for i in range(26, 31):
        ar = np.random.uniform(0.0, 0.4)
        d = np.random.uniform(0.0, 0.5)
        ma = np.random.uniform(0.0, 0.3)
        cols[f"s{i}"] = gen_series(ar, d, ma, nsample)
    return pd.DataFrame(cols)


def common_factor_generation(nsample=4000):
    f1 = np.random.normal(0, 1, nsample)
    f2 = gen_series(0.99, 0.0, 0.0, nsample)
    f3 = gen_series(0.0, 0.2, 0.0, nsample)
    f4 = gen_series(0.0, 0.45, 0.0, nsample)
    return pd.DataFrame({"factor1": f1, "factor2": f2, "factor3": f3, "factor4": f4})


# ============================================================
# 2. AR(p) fixed-coefficient filtering (vectorized)
# ============================================================
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
    idx = np.arange(7, n)
    X = np.column_stack([series[idx - 1 - i] for i in range(p)])
    residuals[idx] = series[idx] - X @ beta
    return residuals


# ============================================================
# 3. Unpooled pairwise (i, j) regressions, closed-form aggregate
# ============================================================
def pairwise_cross_section_mean(resid_matrix, M):
    """
    For every ordered pair (i, j), i != j, fit a SEPARATE calibration-period
    OLS regression e_i,t = alpha_ij + beta_ij * e_j,t + u_ij,t  (t = 1..M).
    No stacking/pooling across pairs: each pair gets its own (alpha_ij,
    beta_ij). These N(N-1) fixed pairs of coefficients are estimated once
    from the calibration sample and then applied UNCHANGED to every t in
    the full sample (calibration + monitoring) -- exactly like the AR
    coefficients in Section 3 of the methodology note. The signal fed into
    the CUSQ test is the simple average, at each t, of the N(N-1) pairwise
    residuals U_ij,t = e_i,t - alpha_ij - beta_ij * e_j,t.

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
# 4. CUSUM-of-squares statistic and Brownian-Bridge boundary
# ============================================================
def cusum_alarm_times(e_t, M):
    """
    Strictly follows Wang & Hsiao (2013) JTSE, pp.5-6.

    Statistic (p.5):
        C_l  = sum_{t=1}^{l} e_t^2
        D_n  = C_n / C_m  -  (n-2) / (m-2)      [Wang & Hsiao eq. on p.5]
        T_n  = sqrt(m/2) * D_n                   [= m^{-1/2} * S_n]

    Boundary (p.6), obtained by discretising lambda as n/m:
        g(n/m) = ((n-m)/m) * [(n/(n-m)) * (a^2 + ln(n/(n-m)))]^{1/2}
                                                  [Wang & Hsiao eq. bottom of p.6]

    Rejection rule (double-sided):  |T_n| >= g(n/m)
    """
    T = len(e_t)
    cumsum_sq = np.cumsum(e_t ** 2)
    Cm = cumsum_sq[M - 1]
    ns = np.arange(M + 1, T + 1, dtype=float)
    Cn = cumsum_sq[M:T]

    # Wang & Hsiao (2013) p.5: D_n = C_n/C_m - (n-2)/(m-2)
    Dn = Cn / Cm - (ns - 2) / (M - 2)
    Tn = np.sqrt(M / 2.0) * Dn

    # Wang & Hsiao (2013) p.6 boundary: g(n/m) = ratio * sqrt(inner*(a^2+ln inner))
    ratio = (ns - M) / M          # (n-m)/m
    inner = ns / (ns - M)         # n/(n-m)

    alarms = {}
    for level, a2 in A2.items():
        g = ratio * np.sqrt(inner * (a2 + np.log(inner)))
        idx = np.where(np.abs(Tn) >= g)[0]
        alarms[level] = ns[idx[0]] if len(idx) > 0 else np.inf
    return alarms


# ============================================================
# 5. Shared pipeline
# ============================================================
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
    n_dropped = int(np.argmax(valid))
    return resid[valid], n_dropped


# ============================================================
# 6. Scenario A (delta = 0)
# ============================================================
def simulation_A(N, m, factor, T_total=None):
    if T_total is None:
        T_total = 10 * m + 10
    df = build_panel(N, factor, T_total)
    resid, _ = ar_filter_panel(df, m)
    e_t, beta = pairwise_cross_section_mean(resid, m)
    alarms = cusum_alarm_times(e_t, m)

    out = {}
    for level, tau_hat in alarms.items():
        for k in K_LIST:
            out[(level, k)] = int(tau_hat <= m + k * m)
    return out


# ============================================================
# 7. Scenario B (delta in {1,2,3})
# ============================================================
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

    e_t, beta = pairwise_cross_section_mean(resid, m)
    alarms = cusum_alarm_times(e_t, m)

    out = {}
    for level, tau_hat in alarms.items():
        if np.isinf(tau_hat):
            out[level] = dict(alarm=0, premature=0, detect=0, delay=np.nan)
        else:
            premature = int(tau_hat <= tau_adj)
            detect = int(tau_hat > tau_adj)
            delay = (tau_hat - tau_adj) if detect else np.nan
            out[level] = dict(alarm=1, premature=premature, detect=detect, delay=delay)
    return out


# ============================================================
# 8. Worker functions (one combo = 2500 reps, run in a worker process)
# ============================================================
def worker_init():
    # Each process gets an independent random stream, seeded from OS entropy
    # via SeedSequence (statistically independent across processes).
    seed = int.from_bytes(os.urandom(4), "little")
    np.random.seed(seed)


def run_combo_A(combo):
    N, m, factor = combo
    R = R_TEST if TEST_MODE else R_FULL
    sums = {(lvl, k): 0 for lvl in LEVELS for k in K_LIST}
    for _ in range(R):
        out = simulation_A(N, m, factor)
        for key, val in out.items():
            sums[key] += val
    rows = []
    for k in K_LIST:
        row = {"N": N, "m": m, "factor": factor, "k": k}
        for lvl in LEVELS:
            row[f"size_{lvl.rstrip('%')}"] = sums[(lvl, k)] / R
        rows.append(row)
    return ("A", combo, rows)


def run_combo_B(combo):
    N, delta, k, m, factor = combo
    R = R_TEST if TEST_MODE else R_FULL
    agg = {lvl: {"premature": 0, "detect": 0, "alarm": 0, "delays": []} for lvl in LEVELS}
    for _ in range(R):
        out = simulation_B(N, delta, k, m, factor)
        for lvl, d in out.items():
            agg[lvl]["premature"] += d["premature"]
            agg[lvl]["detect"] += d["detect"]
            agg[lvl]["alarm"] += d["alarm"]
            if d["detect"] == 1:
                agg[lvl]["delays"].append(d["delay"])

    row = {"N": N, "delta": delta, "k": k, "m": m, "factor": factor}
    for lvl in LEVELS:
        suf = lvl.rstrip("%")
        row[f"alarmrate_{suf}"] = agg[lvl]["alarm"] / R
        row[f"prematurerate_{suf}"] = agg[lvl]["premature"] / R
        row[f"detectrate_{suf}"] = agg[lvl]["detect"] / R
        delays = np.array(agg[lvl]["delays"], dtype=float)
        if len(delays) > 0:
            row[f"delay_mean_{suf}"] = delays.mean()
            row[f"delay_std_{suf}"] = delays.std(ddof=1) if len(delays) > 1 else np.nan
            row[f"delay_median_{suf}"] = np.median(delays)
            row[f"delay_q25_{suf}"] = np.quantile(delays, 0.25)
            row[f"delay_q75_{suf}"] = np.quantile(delays, 0.75)
        else:
            for stat in ["mean", "std", "median", "q25", "q75"]:
                row[f"delay_{stat}_{suf}"] = np.nan
    return ("B", combo, [row])


# ============================================================
# 9. Checkpoint writer
# ============================================================
def write_progress(rows_A, rows_B):
    df_A = pd.DataFrame(rows_A) if rows_A else pd.DataFrame()
    df_B = pd.DataFrame(rows_B) if rows_B else pd.DataFrame()
    with pd.ExcelWriter(PROGRESS_FILE, engine="openpyxl") as writer:
        df_A.to_excel(writer, sheet_name="scenario_A", index=False)
        df_B.to_excel(writer, sheet_name="scenario_B", index=False)


# ============================================================
# 10. Main driver
# ============================================================
def main():
    combos_A = list(itertools.product(N_LIST, M_LIST, FACTOR_LIST))
    combos_B = list(itertools.product(N_LIST, DELTA_LIST, K_LIST, M_LIST, FACTOR_LIST))

    if TEST_MODE:
        # reduced grid for a quick end-to-end check
        combos_A = combos_A[:4]
        combos_B = combos_B[:4]

    tasks = [("A", c) for c in combos_A] + [("B", c) for c in combos_B]
    total = len(tasks)
    R = R_TEST if TEST_MODE else R_FULL
    print(f"Total combos: {total}  (A: {len(combos_A)}, B: {len(combos_B)}), R={R}")

    rows_A, rows_B = [], []
    last_checkpoint = time.time()

    with mp.Pool(processes=N_WORKERS, initializer=worker_init) as pool:
        def dispatch(kind, combo):
            return run_combo_A(combo) if kind == "A" else run_combo_B(combo)

        results_iter = pool.imap_unordered(
            _run_task, tasks, chunksize=1
        )
        with tqdm(total=total, desc="combos completed") as pbar:
            for kind, combo, rows in results_iter:
                if kind == "A":
                    rows_A.extend(rows)
                else:
                    rows_B.extend(rows)
                pbar.update(1)
                print(f"  [done] scenario {kind}  combo={combo}")

                now = time.time()
                if now - last_checkpoint >= CHECKPOINT_SECONDS:
                    write_progress(rows_A, rows_B)
                    print(f"  [checkpoint] wrote {PROGRESS_FILE} "
                          f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    last_checkpoint = now

    # final write: overwrite progress file + a timestamped final copy
    write_progress(rows_A, rows_B)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = os.path.join(OUTPUT_DIR, f"final_results_{ts}.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        pd.DataFrame(rows_A).to_excel(writer, sheet_name="scenario_A", index=False)
        pd.DataFrame(rows_B).to_excel(writer, sheet_name="scenario_B", index=False)
    print(f"Done. Final results written to {final_path} (and {PROGRESS_FILE}).")


# top-level wrapper required for Pool with spawn (must be picklable)
def _run_task(task):
    kind, combo = task
    return run_combo_A(combo) if kind == "A" else run_combo_B(combo)


if __name__ == "__main__":
    main()
