# -*- coding: utf-8 -*-
"""训练随机性评估：单10日 64通道 量价因子，逐 seed 对比

输出：
  A. 各 seed 的信号相似度（逐年相关）
  B. 各 seed 的因子层 IC / ICIR
  C. 各 seed 的策略表现（三档 0.30/0.80 + 众数5, 分位窗口 750）
  D. seed 平均信号 vs 单 seed
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
SEEDS = [0, 1, 2, 3, 4, 5]


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


def load_tag(tag):
    parts = []
    for p in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def pe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


def roll_pct(s, win=750, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp: continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


def pos_mode(t, k=5):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w): continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def tgt3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def bt(pos, start, end="2099-12-31", cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    if len(d) < 100: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd, avg=float(d["p"].mean()),
                cum=float(st.sum()), byyear=st.groupby(st.index.year).sum())


sig = {}
for sd in SEEDS:
    s = load_tag(f"s{sd}")
    if s is not None and len(s) > 1000:
        sig[sd] = s
print(f"可用 seed: {list(sig.keys())}")
for sd, s in sig.items():
    print(f"  seed{sd}: {s.index.min().date()}~{s.index.max().date()} {len(s)} 日")

avg2 = load_tag("h10")           # 现有基线 = seed 0/1 平均（2019-2026）
avgN = pd.concat(sig.values(), axis=1).mean(axis=1).dropna()

print("\n" + "=" * 96)
print("A. 各 seed 信号相似度（逐年 Pearson，原始信号）")
print("=" * 96)
years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
print(f"  {'seed对':<10}" + "".join(f"{y:>7}" for y in years) + f"{'全期':>8}")
pairs = []
for i in range(len(sig)):
    for j in range(i + 1, len(sig)):
        a, b = sig[SEEDS[i]], sig[SEEDS[j]]
        d = pd.DataFrame({"a": a, "b": b}).dropna()
        row = []
        for y in years:
            g = d[d.index.year == y]
            row.append(pe(g["a"].values, g["b"].values) if len(g) > 20 else np.nan)
        pairs.append(row + [pe(d["a"].values, d["b"].values)])
        print(f"  {str(SEEDS[i])+'-'+str(SEEDS[j]):<10}" + "".join(f"{v:>7.2f}" if np.isfinite(v) else f"{'-':>7}" for v in row)
              + f"{pairs[-1][-1]:>8.2f}")
P = np.array(pairs, dtype=float)
print(f"  {'均值':<10}" + "".join(f"{np.nanmean(P[:,k]):>7.2f}" for k in range(len(years))) + f"{np.nanmean(P[:,-1]):>8.2f}")
print(f"  {'最小值':<10}" + "".join(f"{np.nanmin(P[:,k]):>7.2f}" for k in range(len(years))) + f"{np.nanmin(P[:,-1]):>8.2f}")

print("\n" + "=" * 96)
print("B. 各 seed 的因子层表现（逐年 IC 均值 / ICIR / t）")
print("=" * 96)
print(f"  {'seed':<8}{'2019-24 IC':>12}{'ICIR':>7}{'t':>7}{'2025-26 IC':>12}{'2025IC':>8}{'2026IC':>8}")
ic_store = {}
for sd, s in sig.items():
    fwd = np.log(close.shift(-10) / close)
    d = pd.DataFrame({"s": s, "f": fwd.reindex(s.index)}).dropna()
    ics = {y: pe(g["s"].values, g["f"].values) for y, g in d.groupby(d.index.year)}
    a = np.array([ics[y] for y in years[:6] if y in ics and np.isfinite(ics[y])])
    b = np.array([ics[y] for y in years[6:] if y in ics and np.isfinite(ics[y])])
    ic_store[sd] = ics
    mu, sdv = a.mean(), a.std(ddof=1)
    print(f"  seed{sd:<4}{mu:>12.4f}{mu/sdv:>7.2f}{mu/sdv*np.sqrt(len(a)):>7.2f}"
          f"{b.mean():>12.4f}{ics.get(2025,np.nan):>8.3f}{ics.get(2026,np.nan):>8.3f}")
A = np.array([[ic_store[sd].get(y, np.nan) for y in years] for sd in sig])
print(f"  {'均值':<8}" + "".join(f"{np.nanmean(A[:,k]):>8.3f}" for k in range(len(years))))
print(f"  {'标准差':<8}" + "".join(f"{np.nanstd(A[:,k],ddof=1):>8.3f}" for k in range(len(years))))
print(f"  {'最差':<8}" + "".join(f"{np.nanmin(A[:,k]):>8.3f}" for k in range(len(years))))

print("\n" + "=" * 96)
print("C. 各 seed 的策略表现（三档0.30/0.80 + 众数5 + 分位窗口750）")
print("=" * 96)
print(f"  {'seed':<8}{'2019-26夏普':>12}{'年化':>8}{'回撤':>8}{'2019-24夏普':>12}{'年化':>8}{'2025夏普':>10}{'2026夏普':>10}{'均仓':>6}")
res = []
for sd, s in sig.items():
    pos = pos_mode(tgt3(roll_pct(s, 750)), 5)
    full = bt(pos, "2019-01-01")
    a = bt(pos, "2019-01-01", "2024-12-31")
    y25 = bt(pos, "2025-01-01", "2025-12-31")
    y26 = bt(pos, "2026-01-01")
    res.append((sd, full["sharpe"], full["ann"], full["mdd"], a["sharpe"], a["ann"],
                y25["sharpe"] if y25 else np.nan, y26["sharpe"] if y26 else np.nan, full["avg"]))
    print(f"  seed{sd:<4}{full['sharpe']:>12.2f}{full['ann']:>8.1%}{full['mdd']:>8.1%}"
          f"{a['sharpe']:>12.2f}{a['ann']:>8.1%}{res[-1][6]:>10.2f}{res[-1][7]:>10.2f}{full['avg']:>6.0%}")
R = np.array([[r[1], r[4], r[6], r[7]] for r in res], dtype=float)
lab = ["2019-26夏普", "2019-24夏普", "2025夏普", "2026夏普"]
print(f"  {'均值':<8}" + "".join(f"{np.nanmean(R[:,k]):>10.2f}" for k in range(4)))
print(f"  {'标准差':<8}" + "".join(f"{np.nanstd(R[:,k],ddof=1):>10.2f}" for k in range(4)))
print(f"  {'最小':<8}" + "".join(f"{np.nanmin(R[:,k]):>10.2f}" for k in range(4)))
print(f"  {'最大':<8}" + "".join(f"{np.nanmax(R[:,k]):>10.2f}" for k in range(4)))

print("\n" + "=" * 96)
print("D. 平均信号 vs 单 seed")
print("=" * 96)
print(f"  {'信号':<16}{'2019-26夏普':>12}{'年化':>8}{'回撤':>8}{'2019-24夏普':>12}{'2025夏普':>10}{'2026夏普':>10}")
for nm, s in [("单seed范围", None), (f"{len(sig)}seed平均", avgN), ("2seed平均(h10)", avg2)]:
    if s is None:
        print(f"  {nm:<16}{np.nanmin(R[:,0]):>6.2f}~{np.nanmax(R[:,0]):<5.2f}"
              f"{'':>8}{'':>8}{np.nanmin(R[:,1]):>6.2f}~{np.nanmax(R[:,1]):<5.2f}")
        continue
    pos = pos_mode(tgt3(roll_pct(s, 750)), 5)
    full = bt(pos, "2019-01-01"); a = bt(pos, "2019-01-01", "2024-12-31")
    y25 = bt(pos, "2025-01-01", "2025-12-31"); y26 = bt(pos, "2026-01-01")
    print(f"  {nm:<16}{full['sharpe']:>12.2f}{full['ann']:>8.1%}{full['mdd']:>8.1%}"
          f"{a['sharpe']:>12.2f}{y25['sharpe']:>10.2f}{y26['sharpe']:>10.2f}")

print("\n" + "=" * 96)
print("E. 逐年策略收益（各 seed, 众数5）")
print("=" * 96)
print(f"  {'seed':<8}" + "".join(f"{y:>8}" for y in years))
for sd, s in sig.items():
    pos = pos_mode(tgt3(roll_pct(s, 750)), 5)
    r = bt(pos, "2019-01-01")
    print(f"  seed{sd:<4}" + "".join(f"{r['byyear'].get(y,0):>8.1%}" for y in years))
posN = pos_mode(tgt3(roll_pct(avgN, 750)), 5)
rN = bt(posN, "2019-01-01")
print(f"  {'均值信号':<8}" + "".join(f"{rN['byyear'].get(y,0):>8.1%}" for y in years))
