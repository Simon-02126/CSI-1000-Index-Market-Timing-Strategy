# -*- coding: utf-8 -*-
"""label 期限对因子A(量价)的影响: 10日标签 vs 20日标签

对照设计:
  base    = output/signals      (64通道, 10日标签)  ← 现用
  h10     = output/signals_h10  (16通道, 10日标签)
  h20     = output/signals_h20  (16通道, 20日标签)
  v2      = output/signals_v2   (16通道, 5/10/20多期限 + IC损失)
  注: TM_H>0 时 SIG_DIM=16, 故 h10 vs h20 是"只改label"的干净对照;
      base vs h10 只差通道数(64 vs 16)

评估:
  1) 自算前瞻收益 H=10 / H=20 的 IC / RankIC (非重叠)
  2) 逐年 IC (H=20) + 年度 ICIR
  3) 5 分箱前瞻 20 日收益 (20 个 offset 稳健性)
  4) 策略层: 滚动分位(750/250) -> 三档(0.3/0.8) + 离散平滑 -> 回测
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
INPUT, OUT = os.path.join(BASE, "input"), os.path.join(BASE, "output")
ANN, START = 252, "2019-01-01"
SETS = {"base(10日,64通道)": "signals",
        "h10(10日,16通道)": "signals_h10",
        "h20(20日,16通道)": "signals_h20",
        "v2(多期限,16通道)": "signals_v2"}


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
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=pd.to_datetime(df[d]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_sig(d, view="pv"):
    pats = glob.glob(os.path.join(OUT, d, f"*中证1000*_{view}_signals.csv"))
    if not pats:
        return None, None
    fp = sorted(pats)[0]
    df = pd.read_csv(fp, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    df = df.drop_duplicates("日期").set_index("日期").sort_index()
    return df[cols].mean(axis=1), df["label"]


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


def spearman(x, y):
    a = pd.Series(np.asarray(x, float)); b = pd.Series(np.asarray(y, float))
    m = a.notna() & b.notna()
    if m.sum() < 10: return np.nan
    return pearson(a[m].rank().values, b[m].rank().values)


close = load_price()
sig = {}
for nm, d in SETS.items():
    s, lab = load_sig(d)
    if s is None:
        print(f"[跳过] {nm}: 文件不存在")
        continue
    sig[nm] = s
    print(f"{nm:<20} {s.index.min().date()}~{s.index.max().date()}  {len(s)} 日")

print("\n" + "=" * 100)
print("1) 自算前瞻收益的 IC / RankIC（非重叠；2019+）")
print("=" * 100)
print(f"  {'信号':<20}{'H=10 IC':>9}{'H=10 RankIC':>12}{'H=20 IC':>9}{'H=20 RankIC':>12}{'观测(H20)':>10}")
for nm, s in sig.items():
    row = []
    for H in (10, 20):
        fwd = np.log(close.shift(-H) / close)
        d = pd.DataFrame({"s": s, "f": fwd}).dropna()
        d = d[d.index >= START].iloc[::H]
        row += [pearson(d["s"].values, d["f"].values), spearman(d["s"].values, d["f"].values)]
        n20 = len(d)
    print(f"  {nm:<20}{row[0]:>9.4f}{row[1]:>12.4f}{row[2]:>9.4f}{row[3]:>12.4f}{n20:>10}")

print("\n" + "=" * 100)
print("2) H=20 逐年 IC（非重叠）与年度 ICIR")
print("=" * 100)
H = 20
fwd20 = np.log(close.shift(-H) / close)
for nm, s in sig.items():
    d = pd.DataFrame({"s": s, "f": fwd20}).dropna()
    d = d[d.index >= START].iloc[::H]
    ys, ics = [], []
    for y, g in d.groupby(d.index.year):
        ic = pearson(g["s"].values, g["f"].values)
        if np.isfinite(ic):
            ys.append(f"{y}:{ic:+.3f}"); ics.append(ic)
    ics = np.array(ics)
    mu, sd = ics.mean(), ics.std(ddof=1)
    print(f"  {nm:<20} {'  '.join(ys)}")
    print(f"  {'':<20} 均值{mu:+.4f} ICIR{mu/sd:+.2f} t={mu/sd*np.sqrt(len(ics)):+.2f} "
          f"正年比例{(ics>0).mean():.0%} ({len(ics)}年)")

print("\n" + "=" * 100)
print("3) 5 分箱前瞻 20 日收益（20 个 offset 的均值与稳定性，单位 %）")
print("=" * 100)
print(f"  {'信号':<20}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}{'Q5-Q4':>8}{'跨offset符号稳定性':>20}")
for nm, s in sig.items():
    d = pd.DataFrame({"s": s, "f": fwd20}).dropna()
    d = d[d.index >= START]
    A = []
    for off in range(H):
        sub = d.iloc[off::H]
        if len(sub) < 60: continue
        q = pd.qcut(sub["s"], 5, labels=False, duplicates="drop")
        if q.nunique() < 5: continue
        A.append([sub["f"][q == k].mean() * 100 for k in range(5)])
    if not A:
        continue
    A = np.array(A)
    m = A.mean(0)
    stab = np.mean(A[:, 4] < A[:, 3])
    print(f"  {nm:<20}" + "".join(f"{v:>8.2f}" for v in m) + f"{m[4]-m[3]:>8.2f}{stab:>19.0%}")

print("\n" + "=" * 100)
print("4) 策略层: 滚动分位(750/250) -> 三档(0.30/0.80) + 离散平滑 -> 回测 3bp")
print("=" * 100)


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


def bt(pos, start, cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[d.index >= start]
    if len(d) < 150: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN),
                byyear=st.groupby(st.index.year).sum())


for k, lbl in [(5, "众数5"), (10, "众数10"), (20, "众数20")]:
    print(f"\n--- 离散平滑 = {lbl} ---")
    print(f"  {'信号':<20}{'2019+夏普':>10}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}"
          f"{'2023+夏普':>10}{'2023+年化':>10}")
    for nm, s in sig.items():
        p = tgt3(roll_pct(s))
        pos = pos_mode(p, k)
        a, b = bt(pos, START), bt(pos, "2023-01-01")
        if a is None or b is None: continue
        print(f"  {nm:<20}{a['sharpe']:>10.2f}{a['ann']:>8.1%}{a['mdd']:>8.1%}"
              f"{a['calmar']:>8.2f}{a['avg']:>6.0%}{b['sharpe']:>10.2f}{b['ann']:>10.1%}")

print("\n" + "=" * 100)
print("5) 20日标签的对冲组合验证: h20 pv + slow 双因子（分位加权）")
print("=" * 100)
h20_pv, _ = load_sig("signals_h20", "pv")
h20_sl, _ = load_sig("signals_h20", "slow")
base_sl, _ = load_sig("signals", "slow")
combos = [("h20 pv+slow 0.5/0.5", h20_pv, h20_sl, 0.5),
          ("h20 pv+slow 0.25/0.75", h20_pv, h20_sl, 0.25),
          ("base pv + base slow 0.5/0.5", sig.get("base(10日,64通道)"), base_sl, 0.5)]
for nm, a_, b_, w in combos:
    if a_ is None or b_ is None: continue
    pct = w * roll_pct(a_) + (1 - w) * roll_pct(b_)
    pos = pos_mode(tgt3(pct), 5)
    A, B = bt(pos, START), bt(pos, "2023-01-01")
    print(f"  {nm:<30} 2019+ 夏普{A['sharpe']:.2f} 年化{A['ann']:.1%} 回撤{A['mdd']:.1%} "
          f"| 2023+ 夏普{B['sharpe']:.2f} 年化{B['ann']:.1%}")
