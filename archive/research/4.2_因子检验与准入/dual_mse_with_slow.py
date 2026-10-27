# -*- coding: utf-8 -*-
"""双期5+10 MSE 与资金面(slow)结合：联合优化权重/窗口/平滑，并与现行方案对比

关键: 组合的最优 (w, W, k) 未必等于单因子的最优 → 必须联合搜
防过拟合: 同时报告"2019-2024 与 2025-2026 双段都达标"的配置（双重通过）
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
    if len(d) < 60: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                byyear=st.groupby(st.index.year).sum())


slow = load_dir("slow")
slow_pct = {W: roll_pct(slow, W) for W in (150, 200, 300, 500, 750)}
slow_only = {}
print("=" * 104)
print("参照：slow 单因子（各分位窗口，k=5）")
print("=" * 104)
for W in (150, 200, 300, 500, 750):
    r = bt(pos_mode(tgt3(slow_pct[W]), 5))
    r24 = bt(pos_mode(tgt3(slow_pct[W]), 5), "2019-01-01", "2024-12-31")
    r26 = bt(pos_mode(tgt3(slow_pct[W]), 5), "2026-01-01")
    slow_only[W] = r["sharpe"]
    print(f"  W={W:<4} 全期{r['sharpe']:>6.2f}  2019-24 {r24['sharpe']:>6.2f}  "
          f"2025 {bt(pos_mode(tgt3(slow_pct[W]),5),'2025-01-01','2025-12-31')['sharpe']:>6.2f}  "
          f"2026 {r26['sharpe']:>6.2f}  年化{r['ann']:>6.1%}  回撤{r['mdd']:>7.1%}")

PVSETS = {
    "双期5+10 MSE": [f"d5_10_mse_s{i}" for i in range(5)],
    "单10日(现行)": [f"s{i}" for i in range(6)],
    "三期5+10+20 IC": ["d5_10_20_ic"],
}
avg = {}
for nm, tags in PVSETS.items():
    ss = [load_tag(t) for t in tags]
    ss = [s for s in ss if s is not None]
    avg[nm] = pd.concat(ss, axis=1).mean(axis=1).dropna()

WEIGHTS = (0.25, 0.4, 0.5, 0.6, 0.75)
WINS = (150, 200, 300, 500, 750)
KS = (3, 5, 7, 10, 15)

print("\n" + "=" * 104)
print("A. 权重敏感性（各 W,k 网格的夏普均值 / 最小值，2019-2026 全期）")
print("=" * 104)
print(f"  {'pv版本':<16}{'w_pv':>6}{'网格均值':>10}{'网格最小':>10}{'最优配置':>16}{'最优夏普':>9}")
best_rec = {}
for nm, s in avg.items():
    pcts = {W: roll_pct(s, W) for W in WINS}
    for w in WEIGHTS:
        vals = []
        best = (-9e9, None, None)
        for W in WINS:
            for k in KS:
                m = pd.DataFrame({"a": pcts[W], "b": slow_pct[W]}).dropna()
                comp = w * m["a"] + (1 - w) * m["b"]
                r = bt(pos_mode(tgt3(comp), k))
                if r is None: continue
                vals.append(r["sharpe"])
                if r["sharpe"] > best[0]: best = (r["sharpe"], W, k)
        vals = np.array(vals)
        print(f"  {nm:<16}{w:>6.2f}{vals.mean():>10.3f}{vals.min():>10.3f}"
              f"{'W=%d,k=%d' % (best[1], best[2]):>16}{best[0]:>9.3f}")
        best_rec[(nm, w)] = best
    print()

print("=" * 104)
print("B. 双重通过筛选（2019-2024 夏普 ≥ 0.40 且 2025-2026 夏普 ≥ 0.60）")
print("=" * 104)
print(f"  {'pv版本':<16}{'w':>5}{'W':>5}{'k':>4}{'全期':>7}{'19-24':>7}{'2025':>7}{'2026':>7}"
      f"{'年化':>8}{'回撤':>8}{'均仓':>6}")
passed = []
for nm, s in avg.items():
    for w in WEIGHTS:
        for W in WINS:
            for k in KS:
                m = pd.DataFrame({"a": roll_pct(s, W), "b": slow_pct[W]}).dropna()
                comp = w * m["a"] + (1 - w) * m["b"]
                pos = pos_mode(tgt3(comp), k)
                f = bt(pos); r24 = bt(pos, "2019-01-01", "2024-12-31")
                y25 = bt(pos, "2025-01-01", "2025-12-31"); y26 = bt(pos, "2026-01-01")
                if not all([f, r24, y25, y26]): continue
                oos = bt(pos, "2025-01-01")
                if r24["sharpe"] >= 0.40 and oos["sharpe"] >= 0.60:
                    passed.append((nm, w, W, k, f, r24, y25, y26, oos))
for nm, w, W, k, f, r24, y25, y26, oos in passed:
    print(f"  {nm:<16}{w:>5.1f}{W:>5}{k:>4}{f['sharpe']:>7.2f}{r24['sharpe']:>7.2f}"
          f"{y25['sharpe']:>7.2f}{y26['sharpe']:>7.2f}{f['ann']:>8.1%}{f['mdd']:>8.1%}{f['avg']:>6.0%}")
if not passed:
    print("  （无配置同时达标）")

print("\n" + "=" * 104)
print("C. 推荐对比：现行(单10日 W=750 k=5) vs 双期MSE 最优 vs 双期MSE 双重通过首选")
print("=" * 104)
cur = pos_mode(tgt3(0.5 * roll_pct(avg["单10日(现行)"], 750) + 0.5 * slow_pct[750]), 5)
dse_best = best_rec[("双期5+10 MSE", 0.5)]
mse_best = pos_mode(tgt3(0.5 * roll_pct(avg["双期5+10 MSE"], dse_best[1]) + 0.5 * slow_pct[dse_best[1]]),
                    dse_best[2])
mse_pass = None
for nm, w, W, k, f, r24, y25, y26, oos in passed:
    if nm == "双期5+10 MSE":
        mse_pass = (w, W, k)
        break
if mse_pass:
    w, W, k = mse_pass
    pos_mp = pos_mode(tgt3(w * roll_pct(avg["双期5+10 MSE"], W) + (1 - w) * slow_pct[W]), k)
else:
    pos_mp = None
cands = [("现行(单10日 0.5/0.5 W=750 k=5)", cur),
         (f"双期MSE 最优(W={dse_best[1]},k={dse_best[2]},w=0.5)", mse_best)]
if pos_mp is not None:
    cands.append((f"双期MSE 双重通过(w={w},W={W},k={k})", pos_mp))
print(f"  {'方案':<34}{'全期':>7}{'19-24':>7}{'2025':>7}{'2026':>7}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}")
for nm, pos in cands:
    f = bt(pos); r24 = bt(pos, "2019-01-01", "2024-12-31")
    y25 = bt(pos, "2025-01-01", "2025-12-31"); y26 = bt(pos, "2026-01-01")
    print(f"  {nm:<34}{f['sharpe']:>7.2f}{r24['sharpe']:>7.2f}{y25['sharpe']:>7.2f}"
          f"{y26['sharpe']:>7.2f}{f['ann']:>8.1%}{f['mdd']:>8.1%}{f['calmar']:>8.2f}{f['avg']:>6.0%}")

print("\n" + "=" * 104)
print("D. 双期MSE 最优配置的 seed 分布（组合层）")
print("=" * 104)
Wb, kb = dse_best[1], dse_best[2]
vals = []
for i in range(5):
    s = load_tag(f"d5_10_mse_s{i}")
    m = pd.DataFrame({"a": roll_pct(s, Wb), "b": slow_pct[Wb]}).dropna()
    vals.append(bt(pos_mode(tgt3(0.5 * m["a"] + 0.5 * m["b"]), kb))["sharpe"])
vals = np.array(vals)
print(f"  W={Wb}, k={kb}: " + " ".join(f"{v:.2f}" for v in vals)
      + f"  | 均值{vals.mean():.2f} 标准差{vals.std(ddof=1):.2f} 最小{vals.min():.2f}")
cur_vals = []
for i in range(6):
    s = load_tag(f"s{i}")
    m = pd.DataFrame({"a": roll_pct(s, 750), "b": slow_pct[750]}).dropna()
    cur_vals.append(bt(pos_mode(tgt3(0.5 * m["a"] + 0.5 * m["b"]), 5))["sharpe"])
cur_vals = np.array(cur_vals)
print(f"  现行(单10日 W=750,k=5): " + " ".join(f"{v:.2f}" for v in cur_vals)
      + f"  | 均值{cur_vals.mean():.2f} 标准差{cur_vals.std(ddof=1):.2f} 最小{cur_vals.min():.2f}")
