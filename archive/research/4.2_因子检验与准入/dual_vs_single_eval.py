# -*- coding: utf-8 -*-
"""量价因子: 多期限标签 vs 单10日标签 同口径对比
所有信号均为 64 通道、2 seeds、覆盖 2019-2024（同一分位warm-up起点）
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
ANN, START, END = 252, "2019-01-01", "2024-12-31"


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
    if not parts:
        return None
    return pd.concat(parts).sort_index()


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


def spearman(x, y):
    a = pd.Series(np.asarray(x, float)); b = pd.Series(np.asarray(y, float))
    m = a.notna() & b.notna()
    return pearson(a[m].rank().values, b[m].rank().values) if m.sum() >= 10 else np.nan


VARIANTS = {"单10日(基线)": "h10", "双期5+10 MSE": "d5_10_mse", "双期5+10 IC": "d5_10_ic",
            "三期5+10+20 IC": "d5_10_20_ic", "三期5+10+20 MSE": "d5_10_20_mse"}
S = {}
for nm, tag in VARIANTS.items():
    s = load_tag(tag)
    if s is not None and len(s) > 500:
        S[nm] = s
print("可用信号:")
for nm, s in S.items():
    print(f"  {nm:<16} {s.index.min().date()}~{s.index.max().date()}  {len(s)} 日")

print("\n" + "=" * 100)
print("1) 因子层：逐年 IC 的均值 / ICIR / t（2019-2024，日频重叠）")
print("=" * 100)
print(f"  {'信号':<16}{'IC(H=10)':>10}{'ICIR':>7}{'t':>7}{'正年':>6}"
      f"{'IC(H=20)':>10}{'ICIR':>7}{'t':>7}{'正年':>6}{'RankIC20':>10}")
for nm, s in S.items():
    out = [f"  {nm:<16}"]
    for h in (10, 20):
        fwd = np.log(close.shift(-h) / close)
        d = pd.DataFrame({"s": s, "f": fwd.reindex(s.index)}).dropna()
        d = d[(d.index >= START) & (d.index <= END)]
        ics = [pearson(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)]
        ics = np.array([x for x in ics if np.isfinite(x)])
        mu, sd = ics.mean(), ics.std(ddof=1)
        out.append(f"{mu:>10.4f}{mu/sd:>7.2f}{mu/sd*np.sqrt(len(ics)):>7.2f}{(ics>0).mean():>6.0%}")
        if h == 20:
            rics = [spearman(g["s"].values, g["f"].values) for _, g in d.groupby(d.index.year)]
            out.append(f"{np.nanmean(rics):>10.4f}")
    print("".join(out))

print("\n" + "=" * 100)
print("2) 信号间相关性（分位化后，2019-2024）")
print("=" * 100)
names = list(S.keys())
if len(names) > 1:
    M = pd.DataFrame({nm: S[nm].rank(pct=True) for nm in names}).dropna()
    print("  " + "".join(f"{nm:>18}" for nm in names))
    for a in names:
        print(f"  {a:<16}" + "".join(f"{M[a].corr(M[b]):>18.3f}" for b in names))


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


def bt(pos, cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= START) & (d.index <= END)]
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


print("\n" + "=" * 100)
print("3) 策略层（滚动分位750/250 -> 三档0.30/0.80 -> 离散平滑）2019-2024")
print("=" * 100)
for k in (5, 10, 15):
    print(f"\n--- 众数{k} ---")
    print(f"  {'信号':<16}{'夏普':>7}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}{'换仓':>7}")
    for nm, s in S.items():
        r = bt(pos_mode(tgt3(roll_pct(s)), k))
        if r is None: continue
        print(f"  {nm:<16}{r['sharpe']:>7.2f}{r['ann']:>8.1%}{r['mdd']:>8.1%}"
              f"{r['calmar']:>8.2f}{r['avg']:>6.0%}{r['chg']:>7.1f}")

print("\n" + "=" * 100)
print("4) 平滑窗口网格（夏普）")
print("=" * 100)
ks = (3, 5, 7, 10, 12, 15, 20)
print(f"  {'信号':<16}" + "".join(f"{'k='+str(k):>7}" for k in ks))
for nm, s in S.items():
    t3 = tgt3(roll_pct(s))
    row = []
    for k in ks:
        r = bt(pos_mode(t3, k))
        row.append(f"{r['sharpe']:>7.2f}" if r else "      -")
    avg = np.mean([float(x) for x in row if x.strip() != "-"])
    print(f"  {nm:<16}" + "".join(row) + f"   均值{avg:>5.2f}")

print("\n" + "=" * 100)
print("5) 逐年策略收益（众数5）")
print("=" * 100)
yrs = None
res = {}
for nm, s in S.items():
    r = bt(pos_mode(tgt3(roll_pct(s)), 5))
    if r: res[nm] = r["byyear"]
    if yrs is None and r: yrs = sorted(r["byyear"].index)
print(f"  {'年份':<8}" + "".join(f"{nm:>16}" for nm in res))
for y in yrs:
    print(f"  {y:<8}" + "".join(f"{res[nm].get(y,0):>15.1%}" for nm in res))
print(f"  {'累计':<8}" + "".join(f"{res[nm].sum():>15.1%}" for nm in res))
