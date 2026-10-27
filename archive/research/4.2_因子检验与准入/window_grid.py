# -*- coding: utf-8 -*-
"""滚动分位窗口长度对策略的影响（重点：能否救回"双期5+10 MSE"）

用户假设: 把滚动分位窗口从 750 缩短到 150，让过平滑的信号也能打到极端分位
检验:
  1) 各窗口下分位覆盖度 (q80/q90/满仓天数) —— 确认假设是否成立
  2) 各窗口 × 各信号的策略表现 (三档 0.30/0.80 + 众数5)
  3) 最优窗口附近做 k 网格，确认不是尖峰
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
    return pd.concat(parts).sort_index() if parts else None


def roll_pct(s, win, minp=None):
    if minp is None:
        minp = max(60, win // 3)
    v = s.values.astype(float)
    o = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp:
            continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


def tgt3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def pos_mode(t, k):
    v = t.values
    o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        if not len(w):
            continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def bt(pos, cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(START)) & (d.index <= pd.Timestamp(END))]
    if len(d) < 150:
        return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum()
    mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN)
    vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN))


S = {"单10日": load_tag("h10"), "双期MSE": load_tag("d5_10_mse"),
     "双期IC": load_tag("d5_10_ic"), "三期IC": load_tag("d5_10_20_ic")}
S = {k: v for k, v in S.items() if v is not None}

WINS = [120, 150, 200, 300, 500, 750]

print("=" * 104)
print("1) 分位覆盖度：不同窗口下，信号能否打到 80% 以上（满仓阈值）")
print("=" * 104)
print(f"  {'信号':<9}{'窗口':>6}{'q50':>7}{'q80':>7}{'q90':>7}{'满仓天数':>9}{'空仓%':>7}{'半仓%':>7}{'满仓%':>7}")
for nm, s in S.items():
    for W in WINS:
        p = roll_pct(s, W).dropna()
        p = p[(p.index >= START) & (p.index <= END)]
        print(f"  {nm:<9}{W:>6}{p.median():>7.2f}{p.quantile(.8):>7.2f}{p.quantile(.9):>7.2f}"
              f"{int((p >= 0.80).sum()):>9}{(p < .30).mean():>7.0%}"
              f"{((p >= .30) & (p < .80)).mean():>7.0%}{(p >= .80).mean():>7.0%}")
    print()

print("=" * 104)
print("2) 策略表现（三档 0.30/0.80 + 众数5）：窗口 × 信号")
print("=" * 104)
print(f"  {'信号':<9}" + "".join(f"{'W='+str(W):>16}" for W in WINS))
for nm, s in S.items():
    row = []
    for W in WINS:
        r = bt(pos_mode(tgt3(roll_pct(s, W)), 5))
        row.append(f"{r['sharpe']:>7.2f}/{r['ann']:>7.1%}" if r else "       -")
    print(f"  {nm:<9}" + "".join(f"{x:>16}" for x in row))
print("  (格式: 夏普/年化)")


def yearly(pos):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(START)) & (d.index <= pd.Timestamp(END))]
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * 0.0003).dropna()
    return st.groupby(st.index.year).sum()


print("\n" + "=" * 104)
print("3) 双向确认：W=150 详细 + 在 W=150/200 上做平滑窗口 k 网格")
print("=" * 104)
for W in (150, 200, 750):
    print(f"\n--- 分位窗口 W={W} ---")
    print(f"  {'信号':<9}{'夏普':>7}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}{'换仓':>7}{'逐年(20-24)':>34}")
    for nm, s in S.items():
        pos = pos_mode(tgt3(roll_pct(s, W)), 5)
        r = bt(pos)
        if r is None:
            continue
        y = yearly(pos)
        ys = "  ".join(f"{yy}:{v*100:+.0f}%" for yy, v in y.items())
        print(f"  {nm:<9}{r['sharpe']:>7.2f}{r['ann']:>8.1%}{r['mdd']:>8.1%}{r['calmar']:>8.2f}"
              f"{r['avg']:>6.0%}{r['chg']:>7.1f}   {ys}")

print("\n" + "=" * 104)
print("4) 平滑窗口 k 网格（分位窗口 W=150）")
print("=" * 104)
ks = (1, 3, 5, 7, 10, 15)
print(f"  {'信号':<9}" + "".join(f"{'k='+str(k):>8}" for k in ks) + f"{'均值':>8}")
for nm, s in S.items():
    p = tgt3(roll_pct(s, 150))
    row = []
    for k in ks:
        r = bt(pos_mode(p, k))
        row.append(r["sharpe"] if r else np.nan)
    print(f"  {nm:<9}" + "".join(f"{x:>8.2f}" for x in row) + f"{np.nanmean(row):>8.2f}")

print("\n" + "=" * 104)
print("5) 阈值网格（W=150, 众数5）：确认 0.30/0.80 是否合适")
print("=" * 104)
print(f"  {'信号':<9}" + "".join(f"{str(lo)+'/'+str(hi):>12}" for lo in (.20, .30, .40) for hi in (.70, .80, .90)))
for nm, s in S.items():
    p = roll_pct(s, 150)
    row = []
    for lo in (.20, .30, .40):
        for hi in (.70, .80, .90):
            r = bt(pos_mode(tgt3(p, lo, hi), 5))
            row.append(f"{r['sharpe']:>12.2f}" if r else "           -")
    print(f"  {nm:<9}" + "".join(row))
