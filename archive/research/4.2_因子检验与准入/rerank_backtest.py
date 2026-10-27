# -*- coding: utf-8 -*-
"""re-rank 完整回测：直接阈值 vs 合成后重新分位

对照维度: 权重 w_pv × 分位窗口 W × 平滑 k × 模式(直接/re-rank)
评估: 2019-2026 全期 / 2019-2024 / 2025-2026 真样本外，含多 seed 分布
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
    if len(d) < 100: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN),
                full=float((d["p"] >= 0.99).mean()), byyear=st.groupby(st.index.year).sum())


slow = load_dir("slow")
seeds = {sd: load_tag(f"s{sd}") for sd in range(6)}
seeds = {k: v for k, v in seeds.items() if v is not None}
FULL, TR, OOS = ("2019-01-01", "2099-12-31"), ("2019-01-01", "2024-12-31"), ("2025-01-01", "2099-12-31")


def run(pvS, slowS, W, w, k, rerank):
    a = roll_pct(pvS, W); b = roll_pct(slowS, W)
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    comp = w * m["a"] + (1 - w) * m["b"]
    if rerank:
        comp = roll_pct(comp, W)
    pos = pos_mode(tgt3(comp), k)
    return pos


print("=" * 112)
print("1) 权重 × 模式（W=750, k=5, pv 用 6seed 平均信号）")
print("=" * 112)
avg_pv = pd.concat(seeds.values(), axis=1).mean(axis=1).dropna()
print(f"  {'w_pv':<6}{'直接阈值 夏普':>14}{'re-rank 夏普':>14}{'直接 回撤':>11}{'re 回撤':>10}"
      f"{'直接 满仓%':>12}{'re 满仓%':>11}{'re 年化':>10}")
for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
    d1 = bt(run(avg_pv, slow, 750, w, 5, False), *FULL)
    d2 = bt(run(avg_pv, slow, 750, w, 5, True), *FULL)
    print(f"  {w:<6.1f}{d1['sharpe']:>14.3f}{d2['sharpe']:>14.3f}"
          f"{d1['mdd']:>10.1%}{d2['mdd']:>10.1%}{d1['full']:>12.0%}{d2['full']:>11.0%}{d2['ann']:>10.1%}")

print("\n" + "=" * 112)
print("2) 稳健性网格（w=0.5 与 w=0.4，各 (W,k) 组合；数值=夏普 2019-2026）")
print("=" * 112)
for w in (0.5, 0.4):
    print(f"\n  --- w_pv = {w} ---")
    print(f"  {'模式':<10}{'W':>5}" + "".join(f"{'k='+str(k):>9}" for k in (3, 5, 10, 15)) + f"{'均值':>8}{'最小':>8}")
    for rerank in (False, True):
        for W in (300, 500, 750):
            vals = []
            for k in (3, 5, 10, 15):
                r = bt(run(avg_pv, slow, W, w, k, rerank), *FULL)
                vals.append(r["sharpe"])
            nm = "re-rank" if rerank else "直接"
            print(f"  {nm:<10}{W:>5}" + "".join(f"{v:>9.3f}" for v in vals)
                  + f"{np.mean(vals):>8.3f}{np.min(vals):>8.3f}")
    print()

print("=" * 112)
print("3) 真样本外（W=750, k=5）：直接 vs re-rank")
print("=" * 112)
print(f"  {'w_pv':<6}{'模式':<10}{'2019-24夏普':>13}{'年化':>8}{'2025夏普':>10}{'2025收益':>10}"
      f"{'2026夏普':>10}{'2026收益':>10}")
for w in (0.4, 0.5):
    for rerank in (False, True):
        pos = run(avg_pv, slow, 750, w, 5, rerank)
        a = bt(pos, *TR); b = bt(pos, "2025-01-01", "2025-12-31"); c = bt(pos, "2026-01-01")
        nm = "re-rank" if rerank else "直接"
        print(f"  {w:<6.1f}{nm:<10}{a['sharpe']:>13.3f}{a['ann']:>8.1%}"
              f"{b['sharpe']:>10.2f}{b['byyear'].get(2025,0):>10.1%}"
              f"{c['sharpe']:>10.2f}{c['byyear'].get(2026,0):>10.1%}")

print("\n" + "=" * 112)
print("4) 多 seed 分布（w=0.5, W=750, k=5）：re-rank 是否降低了 seed 敏感度")
print("=" * 112)
for rerank in (False, True):
    vals = []
    for sd, s in seeds.items():
        vals.append(bt(run(s, slow, 750, 0.5, 5, rerank), *FULL)["sharpe"])
    vals = np.array(vals)
    nm = "re-rank" if rerank else "直接"
    print(f"  {nm:<10}" + " ".join(f"{v:.2f}" for v in vals)
          + f"  | 均值{vals.mean():.2f} 标准差{vals.std(ddof=1):.2f} 最小{vals.min():.2f}")

print("\n" + "=" * 112)
print("5) 逐年收益（w=0.5, W=750, k=5, 6seed 平均信号）")
print("=" * 112)
d1 = bt(run(avg_pv, slow, 750, 0.5, 5, False), *FULL)
d2 = bt(run(avg_pv, slow, 750, 0.5, 5, True), *FULL)
yrs = sorted(set(d1["byyear"].index) | set(d2["byyear"].index))
print("  年份      直接阈值    re-rank")
for y in yrs:
    print(f"  {y}      {d1['byyear'].get(y,0):>8.1%}    {d2['byyear'].get(y,0):>8.1%}")
print(f"  累计      {d1['byyear'].sum():>8.1%}    {d2['byyear'].sum():>8.1%}")
print(f"\n  指标对比: 直接 夏普{d1['sharpe']:.2f} 年化{d1['ann']:.1%} 回撤{d1['mdd']:.1%} "
      f"Calmar{d1['calmar']:.2f} 均仓{d1['avg']:.0%} 换仓{d1['chg']:.1f}/年")
print(f"            re-rank 夏普{d2['sharpe']:.2f} 年化{d2['ann']:.1%} 回撤{d2['mdd']:.1%} "
      f"Calmar{d2['calmar']:.2f} 均仓{d2['avg']:.0%} 换仓{d2['chg']:.1f}/年")

print("\n" + "=" * 112)
print("6) 仓位差异（直接 vs re-rank，w=0.5）")
print("=" * 112)
p1 = run(avg_pv, slow, 750, 0.5, 5, False)
p2 = run(avg_pv, slow, 750, 0.5, 5, True)
mm = pd.DataFrame({"a": p1, "b": p2}).dropna()
print(f"  仓位完全相同的天数占比: {(mm['a']==mm['b']).mean():.0%}")
print(f"  仓位不同的天数中，re-rank 更保守(更低)的比例: {(mm.loc[mm['a']!=mm['b'],'b'] < mm.loc[mm['a']!=mm['b'],'a']).mean():.0%}")
