# -*- coding: utf-8 -*-
"""四个量价版本，各自在最优 (W,k) 下公平对比：单因子 / 与slow组合 / 样本外 / seed稳定"""
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
fwd10 = np.log(close.shift(-10) / close)


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
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd, avg=float(d["p"].mean()),
                byyear=st.groupby(st.index.year).sum())


def pe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


slow = load_dir("slow")
VERS = {
    "单10日(MSE)": ([f"s{i}" for i in range(6)]),
    "双期5+10 MSE": ([f"d5_10_mse_s{i}" for i in range(5)]),
    "双期5+10 IC": (["d5_10_ic"]),
    "三期5+10+20 IC": (["d5_10_20_ic"]),
}

print("=" * 116)
print("各版本在其最优 (W,k) 下的公平对比")
print("=" * 116)
out = []
for nm, tags in VERS.items():
    ss = [load_tag(t) for t in tags]
    ss = [s for s in ss if s is not None]
    if not ss: continue
    avg = pd.concat(ss, axis=1).mean(axis=1).dropna()
    # 选最优 (W,k)：单因子 2019-2026 夏普最高
    best = (-9e9, None, None)
    for W in (150, 200, 300, 500, 750):
        for k in (3, 5, 7, 10, 15):
            r = bt(pos_mode(tgt3(roll_pct(avg, W)), k))
            if r and r["sharpe"] > best[0]:
                best = (r["sharpe"], W, k)
    _, W, k = best
    # 单因子
    a = bt(pos_mode(tgt3(roll_pct(avg, W)), k), "2019-01-01", "2024-12-31")
    f = bt(pos_mode(tgt3(roll_pct(avg, W)), k))
    y25 = bt(pos_mode(tgt3(roll_pct(avg, W)), k), "2025-01-01", "2025-12-31")
    y26 = bt(pos_mode(tgt3(roll_pct(avg, W)), k), "2026-01-01")
    # 与 slow 组合（同 W,k）
    pva = roll_pct(avg, W); slb = roll_pct(slow, W)
    m = pd.DataFrame({"a": pva, "b": slb}).dropna()
    posC = pos_mode(tgt3(0.5 * m["a"] + 0.5 * m["b"]), k)
    c = bt(posC)
    c25 = bt(posC, "2025-01-01", "2025-12-31")
    c26 = bt(posC, "2026-01-01")
    a24 = bt(posC, "2019-01-01", "2024-12-31")
    # 样本外 IC（pooled）
    d = pd.DataFrame({"s": avg, "f": fwd10.reindex(avg.index)}).dropna()
    oos = d[d.index >= "2025-01-01"]
    ic_oos = pe(oos["s"].values, oos["f"].values)
    # seed 分布（组合层）
    per_seed = []
    for s in ss:
        mm = pd.DataFrame({"a": roll_pct(s, W), "b": slb}).dropna()
        per_seed.append(bt(pos_mode(tgt3(0.5 * mm["a"] + 0.5 * mm["b"]), k))["sharpe"])
    per_seed = np.array(per_seed)
    out.append(dict(nm=nm, W=W, k=k, nseed=len(ss), ic_oos=ic_oos,
                    single_f=f["sharpe"], single_24=a["sharpe"],
                    comb=c["sharpe"], comb_24=a24["sharpe"],
                    c25=c25["sharpe"] if c25 else np.nan, c26=c26["sharpe"] if c26 else np.nan,
                    smean=per_seed.mean(), sstd=per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
                    smin=per_seed.min()))

print(f"  {'版本':<16}{'最优(W,k)':>11}{'seed':>5}{'单因子夏普':>10}{'组合夏普':>9}"
      f"{'组合19-24':>10}{'组合2025':>9}{'组合2026':>9}{'OOS pooledIC':>13}"
      f"{'seed均值':>9}{'std':>6}{'min':>6}")
for r in out:
    wk = "W=%d,k=%d" % (r["W"], r["k"])
    print(f"  {r['nm']:<16}{wk:>11}{r['nseed']:>5}"
          f"{r['single_f']:>10.2f}{r['comb']:>9.2f}{r['comb_24']:>10.2f}"
          f"{r['c25']:>9.2f}{r['c26']:>9.2f}{r['ic_oos']:>13.3f}"
          f"{r['smean']:>9.2f}{r['sstd']:>6.2f}{r['smin']:>6.2f}")

print("\n" + "=" * 116)
print("按不同标准排序")
print("=" * 116)
for key, lbl in [("single_f", "单因子全期夏普"), ("comb", "组合全期夏普"),
                 ("comb_24", "组合 2019-2024"), ("c26", "组合 2026(样本外)"),
                 ("ic_oos", "OOS pooled IC"), ("sstd", "seed 标准差(越小越好)")]:
    vals = [(r[key], r["nm"]) for r in out if np.isfinite(r[key])]
    vals.sort(reverse=(key != "sstd"))
    print(f"  {lbl:<20}" + "  >  ".join(f"{nm}({v:.2f})" for v, nm in vals))
