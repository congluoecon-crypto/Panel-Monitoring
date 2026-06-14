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

# Critical values a^2 solved numerically from Wang & Hsiao (2013) formula [5]:
#   P(sup_{tau>=1} |W^0(tau)|/sqrt(tau) >= a) = 2[1-Phi(a)] + 2*a*phi(a) = alpha
# Paper quotes a^2_{5%} = 7.78 (2 d.p. rounded); exact values used below.
A2 = {"10%": 6.251389, "5%": 7.814728, "1%": 11.344867}
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
# 3. Closed-form pooled cross-sectional residual mean bar_e_t
# ============================================================
def pooled_cross_section_mean(resid_matrix, M):
    N = resid_matrix.shape[1]
    e_bar_full = resid_matrix.mean(axis=1)
    mu = e_bar_full[:M].mean()
    d_cal = resid_matrix[:M, :] - mu
    D = d_cal.sum(axis=1)
    Qp = (d_cal ** 2).sum(axis=1)
    num = np.sum(D ** 2 - Qp)
    den = (N - 1) * np.sum(Qp)
    beta_hat = num / den
    e_t = (1 - beta_hat) * (e_bar_full - mu)
    return e_t, beta_hat, mu


# ============================================================
# 4. CUSUM-of-squares statistic and Brownian-Bridge boundary
# ============================================================
def cusum_alarm_times(e_t, M):
    """
    Strictly follows Wang & Hsiao (2013) JTSЕ:

    Statistic (p.5):
        C_k  = sum_{t=1}^{k} e_t^2
        D_n  = C_n / C_m  -  n / m          (NOT (n-2)/(m-2))
        T_n  = sqrt(m/2) * D_n

    Boundary (p.6):
        tau  = n / m  >= 1
        g(tau) = sqrt( tau * (a^2 + ln(tau)) )   (NOT Chu et al. 1996 form)

    Rejection rule:  |T_n| >= g(tau)
    """
    T = len(e_t)
    cumsum_sq = np.cumsum(e_t ** 2)
    Cm = cumsum_sq[M - 1]
    ns = np.arange(M + 1, T + 1, dtype=float)
    Cn = cumsum_sq[M:T]

    # Wang & Hsiao (2013) eq. on p.5
    Dn = Cn / Cm - ns / M
    Tn = np.sqrt(M / 2.0) * Dn

    # Wang & Hsiao (2013) boundary on p.6: g(tau) = sqrt(tau*(a^2 + ln tau))
    tau = ns / M

    alarms = {}
    for level, a2 in A2.items():
        g = np.sqrt(tau * (a2 + np.log(tau)))
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
    e_t, beta_hat, mu = pooled_cross_section_mean(resid, m)
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

    e_t, beta_hat, mu = pooled_cross_section_mean(resid, m)
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
