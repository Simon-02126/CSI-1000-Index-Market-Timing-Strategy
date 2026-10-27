# -*- coding: utf-8 -*-
"""受控实验结果汇总: label=10 vs 20 (固定64通道)
1) 逐年 IC (H=10 / H=20)
2) 策略层 (滚动分位 -> 三档 -> 离散平滑) 回测
3) 调仓频率实验: 因子在中长期更强, 是否该按 20 日节奏调仓?
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


def load_train(tag):
    """把 2019..2024 的逐年文件拼起来"""
    parts = []
    for fp in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        df = pd.read_csv(fp, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(df[cols].mean(axis=1).values, index=df["日期"], name=tag)
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


close = load_price()
sigs = {"label=10日": load_train("h10"), "label=20日": load_train("h20")}
sigs = {k: v for k, v in sigs.items() if v is not None}
print("受控实验信号 (64通道):")
for k, v in sigs.items():
    print(f"  {k}: {v.index.min().date()}~{v.index.max().date()}  {len(v)} 日")

print("\n" + "=" * 92)
print("1) 逐年 IC（日频重叠, 年内全样本）")
print("=" * 92)
summary = {}
for k, s in sigs.items():
    print(f"\n--- {k} ---")
    rows = []
    for h in (10, 20):
        fwd = np.log(close.shift(-h) / close)
        d = pd.DataFrame({"s": s, "f": fwd.reindex(s.index)}).dropna()
        print(f"  H={h:<3}", end="")
        ics = []
        for y, g in d.groupby(d.index.year):
            ic = pearson(g["s"].values, g["f"].values)
            ics.append((y, ic))
        print("  " + "  ".join(f"{y}:{ic:+.3f}" for y, ic in ics))
        arr = np.array([ic for _, ic in ics if np.isfinite(ic)])
        mu, sd = arr.mean(), arr.std(ddof=1)
        print(f"       均值{mu:+.4f}  ICIR{mu/sd:+.2f}  t={mu/sd*np.sqrt(len(arr)):+.2f}  "
              f"正年比例{(arr>0).mean():.0%}  ({len(arr)}年)")
    # RankIC
    fwd20 = np.log(close.shift(-20) / close)
    d = pd.DataFrame({"s": s, "f": fwd20.reindex(s.index)}).dropna()
    print(f"  H=20 RankIC 均值 {np.nanmean([spearman(g['s'].values, g['f'].values) for _, g in d.groupby(d.index.year)]):+.4f}")


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
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN))


print("\n" + "=" * 92)
print("2) 策略层：滚动分位(750/250) -> 三档(0.30/0.80) + 离散平滑")
print("=" * 92)
for k in (5, 10, 20):
    print(f"\n--- 众数{k} ---")
    print(f"  {'信号':<12}{'2019+夏普':>10}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}{'换仓':>7}"
          f"{'2023+夏普':>10}{'2023+年化':>10}")
    for nm, s in sigs.items():
        pos = pos_mode(tgt3(roll_pct(s)), k)
        a, b = bt(pos, START), bt(pos, "2023-01-01")
        if a is None or b is None: continue
        print(f"  {nm:<12}{a['sharpe']:>10.2f}{a['ann']:>8.1%}{a['mdd']:>8.1%}{a['calmar']:>8.2f}"
              f"{a['avg']:>6.0%}{a['chg']:>7.1f}{b['sharpe']:>10.2f}{b['ann']:>10.1%}")

print("\n" + "=" * 92)
print("3) 调仓频率实验（三档+众数5 的仓位, 改成每 N 日才调一次）")
print("=" * 92)
print(f"  {'信号':<12}" + "".join(f"{'N='+str(n):>22}" for n in (1, 5, 10, 20, 40)))
for nm, s in sigs.items():
    base_pos = pos_mode(tgt3(roll_pct(s)), 5)
    row = []
    for N in (1, 5, 10, 20, 40):
        if N == 1:
            p = base_pos
        else:
            v = base_pos.values.copy()
            cur = v[0] if len(v) else np.nan
            for i in range(len(v)):
                if i % N == 0:
                    cur = v[i]
                else:
                    v[i] = cur
            p = pd.Series(v, index=base_pos.index)
        a = bt(p, START)
        row.append(f"{a['sharpe']:>7.2f}/{a['ann']:>6.1%}")
    print(f"  {nm:<12}" + "".join(f"{x:>22}" for x in row))
print("  (格式: 夏普/年化)")
