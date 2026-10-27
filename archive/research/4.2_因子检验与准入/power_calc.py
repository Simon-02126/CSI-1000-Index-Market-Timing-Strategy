# -*- coding: utf-8 -*-
"""功效计算：加入第三个独立维度，需要多强的 IC 才能改善组合？

方法: 构造与 pv/slow 都正交的合成因子, 其 IC(对未来10日收益) 精确可控, 自相关匹配真实因子(AR1≈0.9)
      以 1/3 等权并入, 观察组合夏普如何随新因子 IC 变化
说明: 合成因子用到了未来收益(仅用于"功效计算/上界估计", 不是可实现的策略)
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


def load_dir(view):
    p = glob.glob(os.path.join(BASE, "output", "signals", f"*中证1000*_{view}_signals.csv"))[0]
    x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
    x = x.drop_duplicates("日期").set_index("日期").sort_index()
    return x[cols].mean(axis=1)


def roll_pct(s, win=750, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp: continue
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


def bt(pos, start="2019-01-01", cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[d.index >= pd.Timestamp(start)]
    if len(d) < 100: return np.nan
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return ann / vol if vol else np.nan


def corr(x, y):
    x = pd.Series(np.asarray(x, float)); y = pd.Series(np.asarray(y, float))
    m = x.notna() & y.notna()
    return float(x[m].corr(y[m]))


pv = load_dir("pv")
slow = load_dir("slow")
fwd10 = np.log(close.shift(-10) / close)
idx = pv.index.intersection(slow.index)
pct_pv = roll_pct(pv.reindex(idx))
pct_sl = roll_pct(slow.reindex(idx))

base2 = bt(pos_mode(tgt3(0.5 * pct_pv + 0.5 * pct_sl)))
slow_only = bt(pos_mode(tgt3(pct_sl)))
print(f"参照: slow 单因子 {slow_only:.3f} | 现行双因子(pv+slow 等权) {base2:.3f}")

# 合成因子: 与两因子正交 + IC 可控 + AR1 约 0.9
rng = np.random.default_rng(7)
f = fwd10.reindex(idx).values
ok = np.isfinite(f)
z_f = np.zeros_like(f); z_f[ok] = (f[ok] - f[ok].mean()) / f[ok].std()


def make_factor(target_ic, ar=0.90, seed=0):
    r = np.random.default_rng(seed)
    n = len(f)
    e = r.normal(size=n)
    arx = np.zeros(n)
    for i in range(1, n):
        arx[i] = ar * arx[i - 1] + e[i] * np.sqrt(1 - ar ** 2)
    arx = arx - arx.mean(); arx = arx / arx.std()
    s = target_ic * z_f + np.sqrt(max(1 - target_ic ** 2, 0)) * arx
    s[~ok] = np.nan
    return pd.Series(s, index=idx)


print("\n" + "=" * 100)
print("第三个独立维度达到不同 IC 时，三维组合（各 1/3）的夏普")
print("=" * 100)
print(f"  {'新因子IC':>9}{'三维组合夏普(10次随机)':>26}{'均值':>8}{'最小':>8}{'最大':>8}"
      f"{'较双因子':>9}{'胜率(优于双因子)':>14}")
for ic in (0.05, 0.10, 0.15, 0.20, 0.25):
    res = []
    cors = []
    for sd in range(10):
        nf = make_factor(ic, seed=sd)
        pn = roll_pct(nf)
        m = pd.DataFrame({"a": pct_pv, "b": pct_sl, "c": pn}).dropna()
        cors.append(corr(m["c"], 0.5 * m["a"] + 0.5 * m["b"]))
        pct3 = (m["a"] + m["b"] + m["c"]) / 3
        res.append(bt(pos_mode(tgt3(pct3))))
    res = np.array(res, dtype=float)
    print(f"  {ic:>9.2f}" + "".join(f"{x:>7.2f}" for x in res[:3]) + f"{'...':>5}"
          f"{np.nanmean(res):>8.2f}{np.nanmin(res):>8.2f}{np.nanmax(res):>8.2f}"
          f"{np.nanmean(res)-base2:>+9.2f}{(res > base2).mean():>14.0%}")

print("\n" + "=" * 100)
print("对照：把同样的合成分子的权重换成 0.2（pv 0.4 / slow 0.4 / 新 0.2）")
print("=" * 100)
print(f"  {'新因子IC':>9}{'三维组合夏普':>14}{'均值':>8}{'最小':>8}{'较双因子':>9}")
for ic in (0.10, 0.15, 0.20):
    res = []
    for sd in range(10):
        nf = make_factor(ic, seed=sd)
        m = pd.DataFrame({"a": pct_pv, "b": pct_sl, "c": roll_pct(nf)}).dropna()
        pct3 = 0.4 * m["a"] + 0.4 * m["b"] + 0.2 * m["c"]
        res.append(bt(pos_mode(tgt3(pct3))))
    res = np.array(res, dtype=float)
    print(f"  {ic:>9.2f}{'':>14}{np.nanmean(res):>8.2f}{np.nanmin(res):>8.2f}{np.nanmean(res)-base2:>+9.2f}")

print("\n" + "=" * 100)
print("对照：新维度替代 pv（slow 0.5 + 新 0.5），看情绪面能否顶掉量价面")
print("=" * 100)
print(f"  {'新因子IC':>9}{'组合夏普':>10}{'较双因子':>9}{'较slow单因子':>13}")
for ic in (0.10, 0.15, 0.20, 0.25):
    res = []
    for sd in range(10):
        nf = make_factor(ic, seed=sd)
        m = pd.DataFrame({"a": pct_sl, "c": roll_pct(nf)}).dropna()
        pct2 = 0.5 * m["a"] + 0.5 * m["c"]
        res.append(bt(pos_mode(tgt3(pct2))))
    res = np.array(res, dtype=float)
    print(f"  {ic:>9.2f}{np.nanmean(res):>10.2f}{np.nanmean(res)-base2:>+9.2f}{np.nanmean(res)-slow_only:>+13.2f}")
