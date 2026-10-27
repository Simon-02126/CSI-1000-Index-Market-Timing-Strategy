# -*- coding: utf-8 -*-
"""slow(资金面/两融+规模) 因子准入筛选：与量价因子同一套 7 项门槛

S1 因子层 IC/ICIR/t  S2 分箱多空  S3 阈值覆盖  S4 参数平台
S5 真样本外 2025-2026  S6 seed 稳定性  S7 与 pv 正交性
附加: 增量 IC、状态分层、信号结构
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
SEEDS = list(range(6))


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


def spearman(x, y):
    a = pd.Series(np.asarray(x, float)); b = pd.Series(np.asarray(y, float))
    m = a.notna() & b.notna()
    return pe(a[m].rank().values, b[m].rank().values) if m.sum() >= 10 else np.nan


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


def pos_mode(t, k):
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
    if len(d) < 60: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN),
                byyear=st.groupby(st.index.year).sum())


sig = {}
for sd in SEEDS:
    s = load_tag(f"slow_s{sd}")
    if s is not None and len(s) > 1000:
        sig[sd] = s
prod = load_dir("slow")
pv = load_dir("pv")
print(f"可用 slow seed: {list(sig.keys())}")
avg = pd.concat(sig.values(), axis=1).mean(axis=1).dropna() if sig else None
print(f"6seed平均 {avg.index.min().date()}~{avg.index.max().date()} {len(avg)} 日 | "
      f"生产版(2seed,{len(prod)}日) | pv({len(pv)}日)")

print("\n" + "=" * 104)
print("S1 因子层：逐年 IC 与 ICIR（2019-2024）")
print("=" * 104)
print(f"  {'信号':<14}{'IC(10日)':>10}{'ICIR':>7}{'t':>7}{'正年':>6}{'IC(20日)':>10}{'ICIR':>7}{'t':>7}{'RankIC10':>10}")
for lbl, s in [("6seed平均", avg), ("生产版2seed", prod)] + [(f"seed{sd}", v) for sd, v in sig.items()]:
    out = [f"  {lbl:<14}"]
    for h, fw in ((10, fwd10), (20, fwd20)):
        d = pd.DataFrame({"s": s, "f": fw.reindex(s.index)}).dropna()
        d = d[(d.index >= "2019-01-01") & (d.index <= "2024-12-31")]
        ics = np.array([pe(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)])
        mu, sdv = ics.mean(), ics.std(ddof=1)
        out.append(f"{mu:>10.4f}{mu/sdv:>7.2f}{mu/sdv*np.sqrt(len(ics)):>7.2f}{(ics>0).mean():>6.0%}")
        if h == 10:
            rics = [spearman(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)]
            out.append(f"{np.nanmean(rics):>10.4f}")
    print("".join(out))

print("\n" + "=" * 104)
print("S2 分箱前瞻 10 日收益（非重叠，单位 %）")
print("=" * 104)
print(f"  {'信号':<14}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}{'多空':>8}")
for lbl, s in [("6seed平均", avg), ("生产版", prod), ("seed0", sig.get(0))]:
    if s is None: continue
    d = pd.DataFrame({"s": roll_pct(s, 750), "f": fwd10.reindex(s.index)}).dropna()
    d = d[d.index >= "2019-01-01"].iloc[::10]
    q = pd.qcut(d["s"], 5, labels=False, duplicates="drop")
    m = [d["f"][q == kk].mean() * 100 for kk in range(5)]
    print(f"  {lbl:<14}" + "".join(f"{v:>8.2f}" for v in m) + f"{m[4]-m[0]:>8.2f}")

print("\n" + "=" * 104)
print("S3 阈值覆盖（分位窗口 750）")
print("=" * 104)
for lbl, s in [("6seed平均", avg), ("生产版", prod)]:
    p = roll_pct(s, 750).dropna()
    print(f"  {lbl:<14} q50={p.median():.2f} q80={p.quantile(.8):.2f} q90={p.quantile(.9):.2f} "
          f"满仓占比={(p>=.80).mean():.0%} 空仓={(p<.30).mean():.0%}")

print("\n" + "=" * 104)
print("S4 参数平台（W × k，夏普 2019-2026）")
print("=" * 104)
print(f"  {'信号':<14}{'W':>5}" + "".join(f"{'k='+str(k):>8}" for k in (3, 5, 7, 10, 15)) + f"{'均值':>8}{'最小':>8}")
for lbl, s in [("6seed平均", avg), ("生产版", prod)]:
    for W in (150, 300, 500, 750):
        vals = [bt(pos_mode(tgt3(roll_pct(s, W)), k))["sharpe"] for k in (3, 5, 7, 10, 15)]
        print(f"  {lbl:<14}{W:>5}" + "".join(f"{v:>8.2f}" for v in vals)
              + f"{np.mean(vals):>8.2f}{np.min(vals):>8.2f}")
    print()

print("=" * 104)
print("S5 真样本外（2025-2026）")
print("=" * 104)
print(f"  {'信号':<14}{'2025 IC':>9}{'2026 IC':>9}{'池化IC':>9}{'2025夏普':>10}{'2026夏普':>10}"
      f"{'2025收益':>10}{'2026收益':>10}")
for lbl, s in [("6seed平均", avg), ("生产版", prod)]:
    d = pd.DataFrame({"s": s, "f": fwd10.reindex(s.index)}).dropna()
    oos = d[d.index >= "2025-01-01"]
    pos = pos_mode(tgt3(roll_pct(s, 750)), 5)
    y25 = bt(pos, "2025-01-01", "2025-12-31"); y26 = bt(pos, "2026-01-01")
    print(f"  {lbl:<14}{pe(d[d.index.year==2025]['s'].values, d[d.index.year==2025]['f'].values):>9.3f}"
          f"{pe(d[d.index.year==2026]['s'].values, d[d.index.year==2026]['f'].values):>9.3f}"
          f"{pe(oos['s'].values, oos['f'].values):>9.3f}"
          f"{y25['sharpe']:>10.2f}{y26['sharpe']:>10.2f}"
          f"{y25['byyear'].get(2025,0):>10.1%}{y26['byyear'].get(2026,0):>10.1%}")

print("\n" + "=" * 104)
print("S6 种子稳定性")
print("=" * 104)
print(f"  {'配置':<22}" + "".join(f"{'s'+str(sd):>7}" for sd in sig) + f"{'均值':>8}{'std':>7}{'min':>7}")
for W, k in ((750, 5), (300, 5), (150, 5)):
    vals = []
    for sd, s in sig.items():
        vals.append(bt(pos_mode(tgt3(roll_pct(s, W)), k))["sharpe"])
    vals = np.array(vals)
    print(f"  {'W=%d,k=%d' % (W, k):<22}" + "".join(f"{v:>7.2f}" for v in vals)
          + f"{vals.mean():>8.2f}{vals.std(ddof=1):>7.2f}{vals.min():>7.2f}")
# 因子层多种子
for h, fw in ((10, fwd10),):
    vals = []
    for sd, s in sig.items():
        d = pd.DataFrame({"s": s, "f": fw.reindex(s.index)}).dropna()
        d = d[(d.index >= "2019-01-01") & (d.index <= "2024-12-31")]
        ics = np.array([pe(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)])
        vals.append(ics.mean())
    vals = np.array(vals)
    print(f"  {'因子层 IC(10日)':<22}" + "".join(f"{v:>7.3f}" for v in vals)
          + f"{vals.mean():>8.3f}{vals.std(ddof=1):>7.3f}{vals.min():>7.3f}")
print(f"  跨seed信号相关(全期): {np.mean([pe(pd.DataFrame({'a': sig[i], 'b': sig[j]}).dropna()['a'].values, pd.DataFrame({'a': sig[i], 'b': sig[j]}).dropna()['b'].values) for i in sig for j in sig if i < j]):.3f}")

print("\n" + "=" * 104)
print("S7 与 pv 的正交性 + 增量 IC")
print("=" * 104)
for lbl, s in [("6seed平均", avg), ("生产版", prod)]:
    m = pd.DataFrame({"a": roll_pct(s, 750), "b": roll_pct(pv, 750)}).dropna()
    m = m[m.index >= "2019-01-01"]
    f = fwd10.reindex(m.index)
    d1 = pd.DataFrame({"s": m["a"], "f": f}).dropna()
    beta = np.polyfit(m["b"].values, m["a"].values, 1)
    resid = pd.Series(m["a"].values - np.polyval(beta, m["b"].values), index=m.index)
    d2 = pd.DataFrame({"s": resid, "f": f}).dropna()
    print(f"  {lbl:<14} 与pv相关 {pe(m['a'].values, m['b'].values):+.3f} | "
          f"IC(自身)={pe(d1['s'].values, d1['f'].values):+.4f} | "
          f"IC(对pv正交化)={pe(d2['s'].values, d2['f'].values):+.4f}")

print("\n" + "=" * 104)
print("附加：状态分层 IC（H=10，用 6seed 平均）")
print("=" * 104)
rv = np.log(close).diff().rolling(20).std() * np.sqrt(ANN)
d = pd.DataFrame({"s": avg, "f": fwd10.reindex(avg.index), "rv": rv.reindex(avg.index),
                  "ma200": close.rolling(200).mean().reindex(avg.index),
                  "px": close.reindex(avg.index)}).dropna()
d = d[d.index >= "2019-01-01"]
med = d["rv"].median()
for lbl, mask in [("高波动", d["rv"] > med), ("低波动", d["rv"] <= med),
                  ("牛市(>200日均线)", d["px"] > d["ma200"]), ("熊市", d["px"] <= d["ma200"])]:
    g = d[mask]
    print(f"  {lbl:<18} IC={pe(g['s'].values, g['f'].values):+.4f}  "
          f"RankIC={spearman(g['s'].values, g['f'].values):+.4f}  n={len(g)}")

print("\n" + "=" * 104)
print("附加：信号结构")
print("=" * 104)
for lbl, s in [("6seed平均", avg), ("生产版", prod)]:
    v = s.values
    print(f"  {lbl:<14} std={v.std():.4f}  AR1={pe(v[1:], v[:-1]):.3f}  "
          f"逐std变异={np.std([g.std() for _, g in s.groupby(s.index.year)])/np.mean([g.std() for _, g in s.groupby(s.index.year)]):.2f}")
