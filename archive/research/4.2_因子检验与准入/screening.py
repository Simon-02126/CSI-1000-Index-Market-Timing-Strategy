# -*- coding: utf-8 -*-
"""量价因子准入选拔：所有候选过同一套门槛，选出最终一个用于后续合成

门槛（7 项，任一不过即淘汰）:
  S1 因子层     : 2019-24 IC(10日) t ≥ 2.0 且 ICIR ≥ 1.0
  S2 尾部区分度 : 分箱多空(H=10, 非重叠) ≥ 1.5%
  S3 阈值覆盖   : 分位 q90 ≥ 0.85 且 满仓占比 ≥ 8%（否则三档打不到满仓）
  S4 参数平台   : W∈{150,300,750} × k∈{3,5,10,15} 策略夏普均值 ≥ 0.30 且 最小 ≥ 0
  S5 真样本外   : 2025 IC > 0 且 2026 IC > 0（seed 平均信号）
  S6 种子稳定性 : 与 slow 组合后，多 seed 夏普最小值 ≥ slow 单因子基准
  S7 正交性     : |corr(pv, slow)| < 0.3
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
TRAIN = _P.TRAIN
ANN = 252


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


fp = glob.glob(os.path.join(BASE, "input", "*中证1000*.csv"))[0]
df = read_any(fp)
dc = [c for c in df.columns if "日期" in c][0]
cc = [c for c in df.columns if "收盘" in c][0]
close = pd.Series(pd.to_numeric(df[cc], errors="coerce").values, index=pd.to_datetime(df[dc]))
close = close[~close.index.duplicated(keep="last")].sort_index()
fwd10 = np.log(close.shift(-10) / close)
fwd20 = np.log(close.shift(-20) / close)


def load_tag(tag):
    parts = []
    for p in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def load_dir(view):
    p = glob.glob(os.path.join(BASE, "output", "signals", f"*中证1000*_{view}_signals.csv"))[0]
    x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
    x = x.drop_duplicates("日期").set_index("日期").sort_index()
    return x[cols].mean(axis=1)


def pe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


def roll_pct(s, win, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    mp = min(minp, max(60, win // 3))
    for i in range(len(v)):
        if i < mp: continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


def tgt3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def pos_mode(t, k=5):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w): continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def bt(pos, start="2019-01-01", end="2099-12-31", cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    if len(d) < 100: return np.nan
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return ann / vol if vol else np.nan


slow = load_dir("slow")
pct_slow = roll_pct(slow, 750)
SLOW_ONLY = bt(pos_mode(tgt3(pct_slow)))
print(f"基准（slow 单因子）2019-2026 夏普 = {SLOW_ONLY:.3f}\n")

# ---- 候选与它的 seed 池 ----
CANDS = {
    "单10日(MSE)": [f"s{sd}" for sd in range(6)],
    "双期5+10 MSE": [f"d5_10_mse_s{sd}" for sd in range(5)],
    "双期5+10 IC": ["d5_10_ic"],
    "三期5+10+20 IC": ["d5_10_20_ic"],
}
pools = {}
for nm, tags in CANDS.items():
    sigs = []
    for t in tags:
        s = load_tag(t)
        if s is not None and len(s) > 1000:
            sigs.append((t, s))
    if sigs:
        pools[nm] = sigs

print("=" * 118)
print("量价因子准入选拔（7 项门槛）")
print("=" * 118)
hdr = (f"  {'候选':<16}{'S1 IC/t':>13}{'S2 多空':>9}{'S3 q90/满仓':>13}{'S4 网格均值/最小':>17}"
       f"{'S5 25IC/26IC':>14}{'S6 组合min':>11}{'S7 相关':>9}{'结论':>8}")
print(hdr)
rows = []
for nm, sigs in pools.items():
    avg = pd.concat([s for _, s in sigs], axis=1).mean(axis=1).dropna()

    # S1
    d = pd.DataFrame({"s": avg, "f": fwd10.reindex(avg.index)}).dropna()
    d = d[(d.index >= "2019-01-01") & (d.index <= "2024-12-31")]
    ics = np.array([pe(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)])
    mu, sdv = ics.mean(), ics.std(ddof=1)
    s1 = (mu / sdv * np.sqrt(len(ics))) if sdv > 0 else np.nan
    s1_pass = (s1 >= 2.0) and (mu / sdv >= 1.0)

    # S2
    dd = pd.DataFrame({"s": roll_pct(avg, 750), "f": fwd10.reindex(avg.index)}).dropna()
    dd = dd[(dd.index >= "2019-01-01")].iloc[::10]
    q = pd.qcut(dd["s"], 5, labels=False, duplicates="drop")
    m = [dd["f"][q == kk].mean() * 100 for kk in range(5)]
    s2 = m[4] - m[0]
    s2_pass = s2 >= 1.5

    # S3
    p = roll_pct(avg, 750).dropna()
    q90 = p.quantile(.9); full = (p >= .80).mean()
    s3_pass = (q90 >= 0.85) and (full >= 0.08)

    # S4
    grid = []
    for W in (150, 300, 750):
        for k in (3, 5, 10, 15):
            grid.append(bt(pos_mode(tgt3(roll_pct(avg, W)), k)))
    g = np.array(grid, dtype=float)
    s4_pass = (np.nanmean(g) >= 0.30) and (np.nanmin(g) >= 0)

    # S5
    d25 = pd.DataFrame({"s": avg, "f": fwd10.reindex(avg.index)}).dropna()
    ic25 = pe(d25[d25.index.year == 2025]["s"].values, d25[d25.index.year == 2025]["f"].values)
    ic26 = pe(d25[d25.index.year == 2026]["s"].values, d25[d25.index.year == 2026]["f"].values)
    s5_pass = (ic25 > 0) and (ic26 > 0)

    # S6 + S7
    combo, corrs = [], []
    for t, s in sigs:
        pp = roll_pct(s, 750)
        mm = pd.DataFrame({"pv": pp, "sl": pct_slow}).dropna()
        corrs.append(pe(mm["pv"].values, mm["sl"].values))
        pct = 0.5 * mm["pv"] + 0.5 * mm["sl"]
        combo.append(bt(pos_mode(tgt3(pct))))
    combo = np.array(combo, dtype=float)
    s6 = np.nanmin(combo) if len(combo) > 1 else np.nan
    s6_pass = (len(combo) >= 5) and (s6 >= SLOW_ONLY)
    s7 = np.mean(corrs)
    s7_pass = abs(s7) < 0.3

    flags = [s1_pass, s2_pass, s3_pass, s4_pass, s5_pass, s6_pass, s7_pass]
    verdict = "通过" if all(flags) else ("淘汰" if not s6_pass and len(combo) < 5 else "淘汰")
    rows.append((nm, flags, s1, s2, q90, full, np.nanmean(g), np.nanmin(g), ic25, ic26,
                 s6, s7, len(combo), verdict))
    mark = lambda b: "✓" if b else "✗"
    print(f"  {nm:<16}{mu:>6.3f}/{s1:>5.2f}{mark(s1_pass)} "
          f"{s2:>7.2f}{mark(s2_pass)} {q90:>5.2f}/{full:>5.0%}{mark(s3_pass)} "
          f"{np.nanmean(g):>7.2f}/{np.nanmin(g):>7.2f}{mark(s4_pass)} "
          f"{ic25:>6.3f}/{ic26:>6.3f}{mark(s5_pass)} "
          f"{s6:>9.2f}{mark(s6_pass)} {s7:>8.3f}{mark(s7_pass)} {verdict:>8}")

print("\n" + "=" * 118)
print("逐项明细")
print("=" * 118)
print(f"  {'候选':<16}{'seed数':>6}{'组合夏普(各seed)':>42}{'组合均值':>9}{'组合最小':>9}")
for nm, flags, s1, s2, q90, full, gm, gmin, ic25, ic26, s6, s7, ns, v in rows:
    sigs = pools[nm]
    cs = []
    for t, s in sigs:
        pp = roll_pct(s, 750)
        mm = pd.DataFrame({"pv": pp, "sl": pct_slow}).dropna()
        cs.append(bt(pos_mode(tgt3(0.5 * mm["pv"] + 0.5 * mm["sl"]))))
    cs = np.array(cs, dtype=float)
    print(f"  {nm:<16}{ns:>6}" + "".join(f"{c:>7.2f}" for c in cs[:6])
          + f"{'':>{42-min(6,len(cs))*7}}" + f"{np.nanmean(cs):>9.2f}{np.nanmin(cs):>9.2f}")
print(f"\n  slow 单因子基准 = {SLOW_ONLY:.3f}（S6 要求组合最小值 ≥ 此值）")
