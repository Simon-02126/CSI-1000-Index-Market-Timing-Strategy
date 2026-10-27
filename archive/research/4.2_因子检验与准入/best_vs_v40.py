# -*- coding: utf-8 -*-
"""对比: 「现在最好的信号」vs「v4.0 的量价因子信号」

A. 信号一致性: 受控重训 h10(64通道/10日标签) 与 v4.0 的 base pv 是不是同一个信号?
B. 信号质量: IC / ICIR (共同窗口 2019-2024)
C. 策略层: base pv / h10 pv / 双因子(base slow)
D. 平滑窗口 k 网格: 确认「众数10」是否尖峰
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
OUT = os.path.join(BASE, "output")
ANN, START = 252, "2019-01-01"
COMMON_END = "2024-12-31"      # 受控重训只到 2024


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


def load_price():
    fp = glob.glob(os.path.join(BASE, "input", "*中证1000*.csv"))[0]
    df = read_any(fp)
    d = [c for c in df.columns if "日期" in c][0]
    c = [c for c in df.columns if "收盘" in c][0]
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=pd.to_datetime(df[d]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_sig_dir(d, view):
    pats = glob.glob(os.path.join(OUT, d, f"*中证1000*_{view}_signals.csv"))
    if not pats: return None
    df = pd.read_csv(sorted(pats)[0], encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    df = df.drop_duplicates("日期").set_index("日期").sort_index()
    return df[cols].mean(axis=1)


def load_train(tag):
    parts = []
    for fp in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        df = pd.read_csv(fp, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(df[cols].mean(axis=1).values, index=df["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


close = load_price()
base_pv = load_sig_dir("signals", "pv")
base_sl = load_sig_dir("signals", "slow")
h10_pv = load_train("h10")
h10_sl = None

S = {"v4.0 pv（base，2015-2026）": base_pv,
     "受控重训 h10 pv（2019-2024）": h10_pv,
     "v4.0 slow（base）": base_sl}

print("=" * 96)
print("A. 信号一致性：受控重训的 h10 pv 与 v4.0 的 base pv 是同一个信号吗？")
print("=" * 96)
a = base_pv[(base_pv.index >= START) & (base_pv.index <= COMMON_END)]
b = h10_pv[(h10_pv.index >= START) & (h10_pv.index <= COMMON_END)]
d = pd.DataFrame({"base": a, "h10": b}).dropna()
print(f"  共同窗口 {d.index.min().date()}~{d.index.max().date()}  n={len(d)}")
print(f"  原始信号相关性      : {pearson(d['base'].values, d['h10'].values):+.4f}")
print(f"  分位化后相关性      : {pearson(d['base'].rank().values, d['h10'].rank().values):+.4f}")
print(f"  一阶差分相关性      : {pearson(d['base'].diff().dropna().values, d['h10'].diff().dropna().values):+.4f}")
yr = []
for y, g in d.groupby(d.index.year):
    yr.append(f"{y}:{pearson(g['base'].values, g['h10'].values):+.3f}")
print("  逐年相关性          : " + "  ".join(yr))


def year_ic(s, h, start=START, end="2099-12-31"):
    fwd = np.log(close.shift(-h) / close)
    d = pd.DataFrame({"s": s, "f": fwd.reindex(s.index)}).dropna()
    d = d[(d.index >= start) & (d.index <= end)]
    ics = []
    for _, g in d.groupby(d.index.year):
        ic = pearson(g["s"].values, g["f"].values)
        if np.isfinite(ic): ics.append(ic)
    if len(ics) < 2: return np.nan, np.nan, np.nan, len(ics)
    arr = np.array(ics)
    mu, sd = arr.mean(), arr.std(ddof=1)
    return mu, mu / sd, mu / sd * np.sqrt(len(arr)), len(arr)


print("\n" + "=" * 96)
print(f"B. 信号质量（共同窗口 {START} ~ {COMMON_END}，逐年 IC 的均值/ICIR/t）")
print("=" * 96)
print(f"  {'信号':<30}{'IC(H=10)':>11}{'ICIR':>7}{'t':>7}{'IC(H=20)':>11}{'ICIR':>7}{'t':>7}")
for nm in ["v4.0 pv（base，2015-2026）", "受控重训 h10 pv（2019-2024）", "v4.0 slow（base）"]:
    s = S[nm]
    r10 = year_ic(s, 10, end=COMMON_END)
    r20 = year_ic(s, 20, end=COMMON_END)
    print(f"  {nm:<30}{r10[0]:>11.4f}{r10[1]:>7.2f}{r10[2]:>7.2f}{r20[0]:>11.4f}{r20[1]:>7.2f}{r20[2]:>7.2f}")


def roll_pct(s, win=750, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp: continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


def tgt3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def pos_mode(t, k):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w): continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def bt(pos, start=START, end="2099-12-31", cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= start) & (d.index <= end)]
    if len(d) < 150: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN))


print("\n" + "=" * 96)
print(f"C. 策略层对比（三档 0.30/0.80 + 离散平滑；共同窗口 {START}~{COMMON_END}）")
print("=" * 96)
combos = {
    "v4.0 pv 单独": (0.5 * roll_pct(base_pv) + 0.5 * roll_pct(base_pv), None),
    "h10 pv 单独（重训）": (0.5 * roll_pct(h10_pv) + 0.5 * roll_pct(h10_pv), None),
    "v4.0 双因子 pv+slow 0.5/0.5": (0.5 * roll_pct(base_pv), roll_pct(base_sl)),
    "h10 pv + v4.0 slow 0.5/0.5": (0.5 * roll_pct(h10_pv), roll_pct(base_sl)),
}
for k in (5, 10, 15):
    print(f"\n--- 离散平滑 = 众数{k} ---")
    print(f"  {'信号':<32}{'夏普':>7}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}{'换仓':>7}")
    for nm, (a, b) in combos.items():
        pct = a + b if b is not None else a
        pos = pos_mode(tgt3(pct), k)
        r = bt(pos, end=COMMON_END)
        if r is None: continue
        print(f"  {nm:<32}{r['sharpe']:>7.2f}{r['ann']:>8.1%}{r['mdd']:>8.1%}"
              f"{r['calmar']:>8.2f}{r['avg']:>6.0%}{r['chg']:>7.1f}")

print("\n" + "=" * 96)
print(f"D. 平滑窗口网格：确认「众数10」不是尖峰（{START}~{COMMON_END}）")
print("=" * 96)
print(f"  {'信号':<32}" + "".join(f"{'k='+str(k):>13}" for k in (3, 5, 7, 10, 12, 15, 20)))
for nm, (a, b) in combos.items():
    pct = a + b if b is not None else a
    t3 = tgt3(pct)
    row = []
    for k in (3, 5, 7, 10, 12, 15, 20):
        r = bt(pos_mode(t3, k), end=COMMON_END)
        row.append(f"{r['sharpe']:>6.2f}/{r['ann']:>5.1%}")
    print(f"  {nm:<32}" + "".join(f"{x:>13}" for x in row))
print("  (格式: 夏普/年化)")

print("\n" + "=" * 96)
print(f"E. 全样本对比（base 系信号覆盖到 2026，公平展示现方案上限）")
print("=" * 96)
full = {
    "v4.0 双因子 众数5": pos_mode(tgt3(0.5 * roll_pct(base_pv) + 0.5 * roll_pct(base_sl)), 5),
    "v4.0 双因子 众数10": pos_mode(tgt3(0.5 * roll_pct(base_pv) + 0.5 * roll_pct(base_sl)), 10),
}
for nm, pos in full.items():
    for a, b, lbl in [(START, "2099-12-31", "2019-2026"), ("2023-01-01", "2099-12-31", "2023-2026留出")]:
        r = bt(pos, a, b)
        print(f"  {nm:<24}{lbl:<16}夏普{r['sharpe']:>5.2f} 年化{r['ann']:>7.1%} "
              f"回撤{r['mdd']:>7.1%} Calmar{r['calmar']:>5.2f} 均仓{r['avg']:>4.0%}")
