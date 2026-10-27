# -*- coding: utf-8 -*-
"""因子A（中证1000 深度学习量价因子）独立评估: IC / RankIC / ICIR / 分层 / 状态

独立口径（不依赖项目里的 label 列）：
  - 前瞻收益由收盘价自算: fwd_H(t) = ln(close[t+H]/close[t])
  - 信号 = 64 通道等权均值（与下游用法一致）；另做 MA5 平滑对照
  - 重叠问题: 日频重叠 IC 的 t 会被高估 √H 倍, 故同时给"非重叠采样"(每 H 天取 1)
    与非重叠样本的逐年 IC 序列 -> ICIR = mean(IC)/std(IC), t = ICIR·√年数

用法: python factor_a_ic.py
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
SIGDIR = os.path.join(BASE, "output", "signals")
ANN = 252
HORIZONS = [1, 2, 3, 5, 10, 20]
MAIN_H = 10
START = "2019-01-01"   # 2017-2018 信号受换手率缺陷污染, 结论只取 2019+


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


def load_price():
    fp = glob.glob(os.path.join(INPUT, "*中证1000*.csv"))[0]
    df = read_any(fp)
    d = [c for c in df.columns if "日期" in c][0]
    c = [c for c in df.columns if "收盘" in c][0]
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values,
                  index=pd.to_datetime(df[d]), name="close")
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_matrix(fn):
    df = pd.read_csv(fn, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    df = df.drop_duplicates("日期").set_index("日期").sort_index()
    return df[cols]


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.nan
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def spearman(x, y):
    x = pd.Series(np.asarray(x, float)); y = pd.Series(np.asarray(y, float))
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return np.nan
    return pearson(x[m].rank().values, y[m].rank().values)


def ic_stats(ics):
    ics = pd.Series(ics).dropna()
    if len(ics) < 3:
        return np.nan, np.nan, np.nan, len(ics)
    mu, sd = ics.mean(), ics.std(ddof=1)
    icir = mu / sd if sd > 0 else np.nan
    return mu, icir, icir * np.sqrt(len(ics)), len(ics)


def main():
    close = load_price()
    fn = glob.glob(os.path.join(SIGDIR, "*中证1000*_pv_signals.csv"))[0]
    mat = load_matrix(fn)
    raw = mat.mean(axis=1)
    print(f"信号: {os.path.basename(fn)} | 64 通道均值 | {raw.index.min().date()}~{raw.index.max().date()} ({len(raw)} 日)")
    print(f"价格: {close.index.min().date()}~{close.index.max().date()} ({len(close)} 日) | 评估起点 {START}\n")

    sigs = {"raw(原始)": raw,
            "MA5(平滑5日)": raw.rolling(5, min_periods=1).mean(),
            "rz250(滚动z)": (raw - raw.rolling(250, min_periods=60).mean())
                            / raw.rolling(250, min_periods=60).std()}

    def frame(name, H):
        fwd = np.log(close.shift(-H) / close)
        df = pd.DataFrame({"s": sigs[name], "f": fwd}).dropna()
        return df[df.index >= START]

    # ---------- 1. 多持有期 IC / RankIC（重叠 + 非重叠）----------
    print("=" * 100)
    print("1) 多持有期 IC / RankIC（2019+；重叠=日频全样本，非重叠=每H天取1个）")
    print("=" * 100)
    for name in sigs:
        print(f"\n--- {name} ---")
        print(f"  {'H':>3} {'重叠IC':>8}{'重叠RankIC':>11}{'非重叠IC':>10}{'非重叠RankIC':>13}"
              f"{'独立观测':>9}{'年度ICIR':>10}{'t':>7}")
        for H in HORIZONS:
            df = frame(name, H)
            if len(df) < 100:
                continue
            ic_ov = pearson(df["s"].values, df["f"].values)
            ric_ov = spearman(df["s"].values, df["f"].values)
            sub = df.iloc[::H]
            ic_nv = pearson(sub["s"].values, sub["f"].values)
            ric_nv = spearman(sub["s"].values, sub["f"].values)
            yearly = []
            for y, g in sub.groupby(sub.index.year):
                if len(g) >= 8:
                    yearly.append(pearson(g["s"].values, g["f"].values))
            mu, icir, t, n = ic_stats(yearly)
            print(f"  {H:>3} {ic_ov:>8.4f}{ric_ov:>11.4f}{ic_nv:>10.4f}{ric_nv:>13.4f}"
                  f"{len(sub):>9}{icir:>10.2f}{t:>7.2f}")

    # ---------- 2. 逐年 IC（H=10）----------
    H = MAIN_H
    print("\n" + "=" * 100)
    print(f"2) 逐年 IC（H={H}；列1=日频重叠 列2=年内非重叠采样）")
    print("=" * 100)
    for name in ["raw(原始)", "MA5(平滑5日)"]:
        df = pd.DataFrame({"s": sigs[name], "f": np.log(close.shift(-H) / close)}).dropna()
        df = df[df.index >= "2017-01-01"]
        print(f"\n--- {name} ---")
        print(f"  {'年份':<6}{'IC(重叠)':>10}{'RankIC':>9}{'IC(非重叠)':>11}{'n(非重叠)':>10}")
        for y, g in df.groupby(df.index.year):
            ic_ov = pearson(g["s"].values, g["f"].values)
            ric = spearman(g["s"].values, g["f"].values)
            sub = g.iloc[::H]
            ic_nv = pearson(sub["s"].values, sub["f"].values)
            flag = "  ← 换手率缺陷段" if y <= 2018 else ""
            print(f"  {y:<6}{ic_ov:>10.4f}{ric:>9.4f}{ic_nv:>11.4f}{len(sub):>10}{flag}")

    # ---------- 3. 月度/半年度 IC 与 ICIR（最保守口径）----------
    print("\n" + "=" * 100)
    print(f"3) 分频 ICIR（H={H}, 2019+；ICIR = 各期 IC 的 mean/std）")
    print("=" * 100)
    for name in ["raw(原始)", "MA5(平滑5日)"]:
        df = frame(name, H)
        for freq, ann_f in [("M", 12), ("Q", 4), ("A", 1)]:
            ics = []
            for _, g in df.groupby(df.index.to_period(freq)):
                if len(g) >= 10:
                    ics.append(pearson(g["s"].values, g["f"].values))
            mu, icir, t, n = ic_stats(ics)
            if np.isfinite(mu):
                print(f"  {name:<12} 频率{freq:<2} IC均值{mu:>+.4f}  ICIR{icir:>+.2f}  "
                      f"年化ICIR{icir*np.sqrt(ann_f):>+.2f}  t={t:>5.2f}  "
                      f"正比例{(np.array(ics)>0).mean():>4.0%}  期数{n}")

    # ---------- 4. 分层前瞻收益（非重叠）----------
    print("\n" + "=" * 100)
    print(f"4) 分 5 层前瞻 {H} 日收益（非重叠，每 {H} 天取 1 个）")
    print("=" * 100)
    for name in ["raw(原始)", "MA5(平滑5日)"]:
        df = frame(name, H).iloc[::H]
        q = pd.qcut(df["s"], 5, labels=False, duplicates="drop")
        print(f"\n--- {name}（{len(df)} 个独立观测）---")
        means = []
        for k in sorted(pd.Series(q).dropna().unique()):
            g = df[q == k]
            means.append(g["f"].mean())
            print(f"  Q{int(k)+1}  信号均值{g['s'].mean():>+8.3f}  前瞻{H}日{g['f'].mean()*100:>+6.2f}%  "
                  f"胜率{(g['f']>0).mean():>4.0%}")
        ls = means[-1] - means[0]
        print(f"  多空 Q5-Q1 {ls*100:+.2f}%/{H}日 → 年化约 {ls*ANN/H*100:+.1f}%  |  "
              f"单调性(秩相关): {spearman(range(len(means)), means):+.2f}")
        print(f"  买持同期 {df['f'].mean()*100:+.2f}%/{H}日")

    # ---------- 5. 分状态 IC ----------
    print("\n" + "=" * 100)
    print(f"5) 分状态 IC（H={H}, 日频重叠, 2019+）")
    print("=" * 100)
    df = pd.DataFrame({"s": sigs["MA5(平滑5日)"], "f": np.log(close.shift(-H) / close),
                       "rv": np.log(close).diff().rolling(20).std() * np.sqrt(ANN),
                       "ma200": close.rolling(200).mean(), "px": close,
                       "r20": close.pct_change(20)}).dropna()
    df = df[df.index >= START]
    med = df["rv"].median()
    for lbl, mask in [("高波动", df["rv"] > med), ("低波动", df["rv"] <= med),
                      ("牛市(>200日均线)", df["px"] > df["ma200"]),
                      ("熊市(<200日均线)", df["px"] <= df["ma200"]),
                      ("近20日上涨", df["r20"] > 0), ("近20日下跌", df["r20"] <= 0)]:
        g = df[mask]
        print(f"  {lbl:<18} IC={pearson(g['s'].values, g['f'].values):+.4f}  "
              f"RankIC={spearman(g['s'].values, g['f'].values):+.4f}  n={len(g)}")

    # ---------- 6. 64 通道是否有差异 ----------
    print("\n" + "=" * 100)
    print(f"6) 64 个通道各自 IC（H={H}, 2019+）—— 检验通道结构是否真的分散")
    print("=" * 100)
    fw = np.log(close.shift(-H) / close)
    ics = []
    for c in mat.columns:
        s = mat[c]
        d = pd.DataFrame({"s": s, "f": fw.reindex(s.index)}).dropna()
        d = d[d.index >= START]
        ics.append(pearson(d["s"].values, d["f"].values))
    ics = np.array([x for x in ics if np.isfinite(x)])
    print(f"  通道数 {len(ics)} | IC 均值 {ics.mean():+.4f} | 中位 {np.median(ics):+.4f} | 标准差 {ics.std():.4f}")
    print(f"  IC>0: {(ics>0).mean():.0%} | IC>0.05: {(ics>0.05).mean():.0%} | "
          f"范围 [{ics.min():+.4f}, {ics.max():+.4f}]")
    d = pd.DataFrame({"s": raw, "f": fw.reindex(raw.index)}).dropna()
    d = d[d.index >= START]
    print(f"  64 通道等权均值后的 IC = {pearson(d['s'].values, d['f'].values):+.4f}")
    corr_mat = mat.loc[mat.index >= START].corr()
    off = corr_mat.values[np.triu_indices_from(corr_mat.values, 1)]
    print(f"  通道间相关性: 均值 {off.mean():+.3f} | 中位 {np.median(off):+.3f} | "
          f"<0.5 的比例 {(off<0.5).mean():.0%}")

    # ---------- 7. 与资金面因子(slow)的关系 ----------
    print("\n" + "=" * 100)
    print("7) 与资金面因子(slow)的相关性与增量 IC")
    print("=" * 100)
    fs = glob.glob(os.path.join(SIGDIR, "*中证1000*_slow_signals.csv"))[0]
    sl = load_matrix(fs).mean(axis=1)
    both = pd.DataFrame({"pv": raw, "sl": sl}).dropna()
    both = both[both.index >= START]
    print(f"  原始相关 {pearson(both['pv'].values, both['sl'].values):+.3f} | "
          f"分位后相关 {pearson(both['pv'].rank().values, both['sl'].rank().values):+.3f} (n={len(both)})")
    beta = np.polyfit(both["sl"].values, both["pv"].values, 1)
    resid = pd.Series(both["pv"].values - np.polyval(beta, both["sl"].values), index=both.index)
    f2 = fw.reindex(both.index)
    print(f"  IC(pv)            = {pearson(both['pv'].values, f2.values):+.4f}")
    print(f"  IC(pv 对 slow 正交) = {pearson(resid.values, f2.values):+.4f}")
    print(f"  IC(slow)          = {pearson(both['sl'].values, f2.values):+.4f}")

    # ---------- 8. 前后半段 ----------
    print("\n" + "=" * 100)
    print("8) 前后半段稳定性（非重叠）")
    print("=" * 100)
    for name in ["raw(原始)", "MA5(平滑5日)"]:
        df = frame(name, H).iloc[::H]
        for a, b, lbl in [("2019-01-01", "2022-12-31", "2019-2022"),
                          ("2023-01-01", "2099-12-31", "2023-2026")]:
            g = df[(df.index >= a) & (df.index <= b)]
            print(f"  {name:<12} {lbl}  IC={pearson(g['s'].values, g['f'].values):+.4f}  "
                  f"RankIC={spearman(g['s'].values, g['f'].values):+.4f}  n={len(g)}")


if __name__ == "__main__":
    main()
