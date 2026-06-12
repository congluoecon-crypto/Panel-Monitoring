"""
1) 检验size：weak factor，1/2/3使用random shock， 4/5/6使用AR（1），7/8/9/10使用long memory，Pool OLS直接使用
2）检验power：1.2m时刻1/2/3被common factor1 影响，1.2m之后4/5/6被common factor 2 影响，2.4m之后7/8/9被common factor影响。
3）记录单次simulation的时间，作为证据去说明。
4)propose currency market pattern by advanced econometrics model
5)simutaneously trading and break monitoirng
6

Brownian Bridge Structural Break Monitor (CUSUM of Squares)
===========================================================
Detect a change in the cross-sectional mean of AR-filtered residuals.
Statistic: T_n = sqrt(M) * (C_n / C_m - (n-2)/(M-2))
Under H0, T_n converges to a Brownian bridge.
Critical values (sup |W^0|): 1.22 (10%), 1.36 (5%), 1.63 (1%).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
from itertools import permutations
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ------------------- 数据生成函数（不变） -------------------
def simulate_arma(ar_coef, ma_coef, nsample=100, scale=1.0):
    arma = ArmaProcess(ar_coef, ma_coef)
    return arma.generate_sample(nsample=nsample, scale=scale)

def fracdiff_coefs(d, n_max):
    from math import gamma
    from scipy.special import gammaln
    coefs = np.zeros(n_max)
    for j in range(n_max):
        if j < 100:
            coefs[j] = gamma(j + d) / (gamma(j + 1) * gamma(d))
        else:
            coefs[j] = np.exp(gammaln(j + d) - gammaln(j + 1)) / gamma(d)
    return coefs

def simulate_arfima(d, ar=None, ma=None, nsample=100, scale=1.0):
    if ar is not None and ma is not None:
        Y = ArmaProcess(ar, ma).generate_sample(nsample=nsample, scale=scale)
    else:
        Y = np.random.normal(0, scale, nsample)
    w = fracdiff_coefs(d, nsample)
    X = np.zeros(nsample)
    for t in range(nsample):
        X[t] = np.sum(w[:t+1] * np.flip(Y[:t+1]))
    return X

def data_generation(nsample=4000):
    """生成10条异质序列，取最后3500期"""
    ar1 = np.array([1, -0.8])
    ma1 = np.array([1,  0.5])
    s1 = simulate_arma(ar1, ma1, nsample)
    s2 = simulate_arma(ar1, ma1, nsample)

    ar2 = np.array([1, -0.5])
    ma2 = np.array([1,  0.3])
    s3 = simulate_arma(ar2, ma2, nsample)
    s4 = simulate_arma(ar2, ma2, nsample)

    s5 = simulate_arfima(0.4, None, None, nsample)
    s6 = simulate_arfima(0.4, None, None, nsample)

    ar4 = np.array([1, -0.1])
    ma4 = np.array([1,  0.8])
    s7 = simulate_arfima(0.3, ar4, ma4, nsample)
    s8 = simulate_arfima(0.3, ar4, ma4, nsample)

    s9 = simulate_arfima(0.4, ar4, None, nsample)
    s10 = simulate_arfima(0.4, ar4, None, nsample)

    df = pd.DataFrame({
        's1': s1, 's2': s2, 's3': s3, 's4': s4,
        's5': s5, 's6': s6, 's7': s7, 's8': s8,
        's9': s9, 's10': s10
    })
    return df.iloc[-3500:]

def common_factor_generation(nsample=4000, scale=1.0):
    s1 = np.random.normal(0, scale, nsample)
    ar = np.array([1, -0.99])
    ma = np.array([1, 0])
    s2 = simulate_arma(ar, ma, nsample, scale=scale)
    s3 = simulate_arfima(0.2, None, None, nsample, scale=scale)
    s4 = simulate_arfima(0.45, None, None, nsample, scale=scale)
    df = pd.DataFrame({'N01': s1, 'AR99': s2, 'ARF02': s3, 'ARF045': s4})
    return df.iloc[-3500:]

# ------------------- AR 滤波（固定系数） -------------------
def rolling_ar_residuals_fixed(series, M):
    series = np.asarray(series, dtype=float)
    n = len(series)
    p = max(3, int((M - 1) // 100) * 3)
    if M <= p:
        raise ValueError(f"M={M} 太小，无法在阶数p={p}下估计校准期AR系数")
    y_cal = series[p:M]
    X_cal = np.column_stack([series[p-i-1:M-i-1] for i in range(p)])
    beta, *_ = np.linalg.lstsq(X_cal, y_cal, rcond=None)
    residuals = np.full(n, np.nan)
    for t in range(8, n+1):
        if t <= p:
            continue
        last_obs = series[t-p-1:t-1][::-1]
        residuals[t-1] = series[t-1] - beta @ last_obs
    return residuals

def inject_break(df, M, break_frac=0.5, break_size=0.0):
    if break_size == 0:
        return df, None
    T_raw = len(df)
    tau = M + int(break_frac * (T_raw - M))
    df_out = df.copy()
    df_out.iloc[tau:] = df_out.iloc[tau:] + break_size
    return df_out, tau

# ------------------- 单次模拟（增加 factor 参数） -------------------
def simulation_once(M=80, break_size=0.0, break_frac=0.5, factor='ARF045'):
    """
    单次蒙特卡洛模拟，支持选择不同的公共因子。

    factor: str, 可选 'N01', 'AR99', 'ARF02', 'ARF045'
    """
    # Step 0: 生成原始数据 + 指定公共因子（乘以0.5）
    df_raw = data_generation()
    common_factor = common_factor_generation()
    # 提取指定因子列，乘以0.5后广播加到每条序列
    factor_col = common_factor[factor].values  # (3500,)
    df = df_raw + factor_col[:, np.newaxis] * 0.5
    df = df.iloc[:10*M+10].reset_index(drop=True)

    # 注入突变
    df, break_point = inject_break(df, M, break_frac=break_frac, break_size=break_size)

    # Step 1: 固定系数AR滤波
    df_resid = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        df_resid[col] = rolling_ar_residuals_fixed(df[col].values, M)
    df_resid = df_resid.dropna().astype(float)
    data = df_resid.values

    # Step 2: Pool OLS 与横截面均值
    N_series = 10
    pairs = list(permutations(range(N_series), 2))
    N_pairs = len(pairs)

    cal_data = data[:M, :]
    Y_pool = np.concatenate([cal_data[:, i] for (i, j) in pairs])
    X_pool = np.concatenate([
        np.column_stack([np.ones(M), cal_data[:, j]])
        for (i, j) in pairs
    ])
    coef, *_ = np.linalg.lstsq(X_pool, Y_pool, rcond=None)
    alpha_hat, beta_hat = coef[0], coef[1]

    i_arr = np.array([i for (i, j) in pairs])
    j_arr = np.array([j for (i, j) in pairs])
    U = data[:, i_arr] - alpha_hat - beta_hat * data[:, j_arr]
    e_mean = U.mean(axis=1)

    # Step 3: 统计量与边界
    A2 = {'10%': 6.2514, '5%': 7.8147, '1%': 11.3449}
    T = len(e_mean)
    cumsum_sq = np.cumsum(e_mean**2)
    Cm = cumsum_sq[M-1]
    ns = np.arange(M+1, T+1, dtype=float)
    Cn_arr = cumsum_sq[M:T]
    Dn_arr = Cn_arr / Cm - (ns - 2) / (M - 2)
    Tn_arr = np.sqrt(M / 2.0) * Dn_arr

    ratio = (ns - M) / M
    inner = ns / (ns - M)
    def boundary(a2):
        return ratio * np.sqrt(inner * (a2 + np.log(inner)))

    g_10 = boundary(A2['10%'])
    g_5  = boundary(A2['5%'])
    g_1  = boundary(A2['1%'])

    reject_10 = (np.abs(Tn_arr) >= g_10).astype(int)
    reject_5  = (np.abs(Tn_arr) >= g_5).astype(int)
    reject_1  = (np.abs(Tn_arr) >= g_1).astype(int)

    index = ns.astype(int)
    reject_10_mat = np.tile(reject_10[:, None], (1, N_pairs))
    reject_5_mat  = np.tile(reject_5[:, None],  (1, N_pairs))
    reject_1_mat  = np.tile(reject_1[:, None],  (1, N_pairs))

    col_names = [f"β_{i+1}-{j+1}" for (i,j) in pairs]
    cstath0_10 = pd.DataFrame(reject_10_mat, index=index, columns=col_names)
    cstath0_5  = pd.DataFrame(reject_5_mat,  index=index, columns=col_names)
    cstath0_1  = pd.DataFrame(reject_1_mat,  index=index, columns=col_names)

    def first_one(series):
        ones = series[series == 1]
        return ones.index[0] if not ones.empty else np.nan

    first_10 = first_one(cstath0_10.iloc[:, 0])
    first_5  = first_one(cstath0_5.iloc[:, 0])
    first_1  = first_one(cstath0_1.iloc[:, 0])

    RF_stats_10 = pd.DataFrame([[first_10]*N_pairs], columns=col_names, index=['RL'])
    RF_stats_5  = pd.DataFrame([[first_5]*N_pairs],  columns=col_names, index=['RL'])
    RF_stats_1  = pd.DataFrame([[first_1]*N_pairs],  columns=col_names, index=['RL'])

    return cstath0_10, cstath0_5, cstath0_1, RF_stats_10, RF_stats_5, RF_stats_1, break_point

# ------------------- 汇总函数（不变） -------------------
def summarize_RF_stats(RF_stats_list):
    cols = RF_stats_list[0].columns
    col_values = {col: [] for col in cols}
    for df in RF_stats_list:
        for col in cols:
            val = df.loc['RL', col]
            col_values[col].append(val)
    summary_data = {}
    for col in cols:
        series = pd.Series(col_values[col])
        valid = series.dropna()
        q1  = valid.quantile(0.25) if not valid.empty else np.nan
        med = valid.median() if not valid.empty else np.nan
        q3  = valid.quantile(0.75) if not valid.empty else np.nan
        mean= valid.mean() if not valid.empty else np.nan
        std = valid.std(ddof=1) if len(valid) > 1 else np.nan
        mx  = valid.max() if not valid.empty else np.nan
        prob = len(valid) / len(series)
        summary_data[col] = [q1, med, q3, mean, std, mx, prob]
    summary_index = ["1/4 quantile","median","3/4 quantile","mean","std","max","probability"]
    summary_df = pd.DataFrame(summary_data, index=summary_index)
    return summary_df

def compute_size_power(flag_list_10, flag_list_5, flag_list_1,
                       rf_list_10, rf_list_5, rf_list_1, m):
    K_targets = [0.2*m, 0.3*m, 0.5*m, m, 2*m, 3*m, 4*m]
    K_labels  = ['0.2m','0.3m','0.5m','m','2m','3m','4m']

    avg_flag_10 = pd.concat(flag_list_10, axis=0).groupby(level=0).mean()
    avg_flag_5  = pd.concat(flag_list_5,  axis=0).groupby(level=0).mean()
    avg_flag_1  = pd.concat(flag_list_1,  axis=0).groupby(level=0).mean()
    for avg in [avg_flag_10, avg_flag_5, avg_flag_1]:
        avg['mean'] = avg.mean(axis=1)

    rel_idx = avg_flag_10.index - m
    rate_10, rate_5, rate_1 = [], [], []
    for k in K_targets:
        pos = np.argmin(np.abs(rel_idx - k))
        rate_10.append(avg_flag_10['mean'].iloc[pos])
        rate_5.append(avg_flag_5['mean'].iloc[pos])
        rate_1.append(avg_flag_1['mean'].iloc[pos])

    rate_df = pd.DataFrame({'10%': rate_10, '5%': rate_5, '1%': rate_1},
                           index=K_labels)

    def rl_summary(rf_list):
        rls = pd.Series([df.iloc[0, 0] for df in rf_list])
        valid = rls.dropna()
        return {
            'median':      valid.median() if not valid.empty else np.nan,
            'mean':        valid.mean()   if not valid.empty else np.nan,
            'std':         valid.std(ddof=1) if len(valid)>1 else np.nan,
            'q25':         valid.quantile(0.25) if not valid.empty else np.nan,
            'q75':         valid.quantile(0.75) if not valid.empty else np.nan,
            'prob_alarm':  len(valid) / len(rls)
        }

    rl_df = pd.DataFrame({
        '10%': rl_summary(rf_list_10),
        '5%':  rl_summary(rf_list_5),
        '1%':  rl_summary(rf_list_1)
    })
    return rate_df, rl_df

# ------------------- 主程序：对所有四个因子分别进行蒙特卡洛模拟 -------------------
if __name__ == "__main__":
    np.random.seed(100)                      # 固定种子，确保可重复性
    num_simulations = 100                    # 模拟次数增加到100
    m_size = [50, 80, 100]
    break_sizes = [1.0, 2.0, 3.0]
    break_fracs = [0.1, 0.5, 0.9]
    factors = ['N01', 'AR99', 'ARF02', 'ARF045']   # 四个公共因子

    # 存储所有结果（将附加 factor 列）
    all_size_rows = []
    all_power_rows = []
    all_rl_rows = []

    for factor in factors:
        print("\n" + "="*70)
        print(f"开始模拟公共因子: {factor}")
        print("="*70)

        # ======== H0：size模拟 ========
        print("\nH0 SIZE 模拟（break_size=0）")
        for m in m_size:
            print(f"\nM = {m} ...")
            fl10, fl5, fl1, rl10, rl5, rl1 = [], [], [], [], [], []
            for _ in tqdm(range(num_simulations), desc=f"H0 M={m}, factor={factor}"):
                f10, f5, f1, r10, r5, r1, _ = simulation_once(
                    M=m, break_size=0.0, factor=factor
                )
                fl10.append(f10); fl5.append(f5); fl1.append(f1)
                rl10.append(r10); rl5.append(r5); rl1.append(r1)
            rate_df, rl_df = compute_size_power(fl10, fl5, fl1, rl10, rl5, rl1, m)
            # 记录 size 结果（按 horizon）
            for horizon in rate_df.index:
                all_size_rows.append({
                    'factor': factor,
                    'M': m, 'horizon': horizon, 'type': 'size',
                    'break_size': 0.0, 'break_frac': None,
                    'rate_10': rate_df.loc[horizon,'10%'],
                    'rate_5':  rate_df.loc[horizon,'5%'],
                    'rate_1':  rate_df.loc[horizon,'1%'],
                })
            # 记录 size 的 RL 统计
            rl_entry = {'factor': factor, 'M': m, 'type': 'size', 'break_size': 0.0, 'break_frac': None}
            for col in ['10%','5%','1%']:
                for stat in rl_df.index:
                    rl_entry[f'{stat}_{col}'] = rl_df.loc[stat, col]
            all_rl_rows.append(rl_entry)
            print(f"  Size@5%: {rate_df['5%'].values.round(3)}")

        # ======== H1：power模拟 ========
        print("\nH1 POWER 模拟（9个场景）")
        for bs in break_sizes:
            for bf in break_fracs:
                print(f"\n  break_size={bs}, break_frac={bf}")
                for m in m_size:
                    fl10, fl5, fl1, rl10, rl5, rl1 = [], [], [], [], [], []
                    for _ in tqdm(range(num_simulations),
                                  desc=f"H1 M={m} bs={bs} bf={bf} factor={factor}"):
                        f10, f5, f1, r10, r5, r1, _ = simulation_once(
                            M=m, break_size=bs, break_frac=bf, factor=factor
                        )
                        fl10.append(f10); fl5.append(f5); fl1.append(f1)
                        rl10.append(r10); rl5.append(r5); rl1.append(r1)
                    rate_df, rl_df = compute_size_power(
                        fl10, fl5, fl1, rl10, rl5, rl1, m
                    )
                    # 记录 power 结果
                    for horizon in rate_df.index:
                        all_power_rows.append({
                            'factor': factor,
                            'M': m, 'horizon': horizon, 'type': 'power',
                            'break_size': bs, 'break_frac': bf,
                            'rate_10': rate_df.loc[horizon,'10%'],
                            'rate_5':  rate_df.loc[horizon,'5%'],
                            'rate_1':  rate_df.loc[horizon,'1%'],
                        })
                    # 记录 power 的 RL 统计
                    rl_entry = {'factor': factor, 'M': m, 'type': 'power',
                                'break_size': bs, 'break_frac': bf}
                    for col in ['10%','5%','1%']:
                        for stat in rl_df.index:
                            rl_entry[f'{stat}_{col}'] = rl_df.loc[stat, col]
                    all_rl_rows.append(rl_entry)
                    print(f"    M={m}: Power@5%={rate_df['5%'].values.round(3)}")

    # 保存所有结果到 CSV
    pd.DataFrame(all_size_rows + all_power_rows).to_csv("size_power_results_by_factor.csv", index=False)
    pd.DataFrame(all_rl_rows).to_csv("rl_results_by_factor.csv", index=False)
    print("\n所有结果已保存至 size_power_results_by_factor.csv 和 rl_results_by_factor.csv")