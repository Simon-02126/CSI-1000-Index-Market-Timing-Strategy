# -*- coding: utf-8 -*-
"""量价因子(pv)能否进入多因子组合 —— 三条准入检验

基准: 只用 slow 因子
处理: pv 以权重 w 加入（6 个 seed 各算一次，看下限）
安慰剂: 把 pv 信号随机平移（保留分布与自相关，只破坏时间对齐），重复 40 次
判据: 真实 pv 带来的组合夏普提升，必须显著超过安慰剂的提升分布
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
W, K = 750, 5


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


def roll_pct(s, win=W, minp=250):
    v = s.values.astype(float)
    o = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp: continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


def tgt3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def pos_mode(t, k=K):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        ww = v[max(0, i - k + 1):i + 1]; ww = ww[~np.isnan(ww)]
        if not len(ww): continue
        vals, cnt = np.unique(ww, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def bt(pos, start="2019-01-01", end="2099-12-31", cost=0.0003):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    if len(d) < 100: return np.nan, np.nan
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return (ann / vol if vol else np.nan), ann


slow = load_dir("slow")
pct_slow = roll_pct(slow)
pv = {sd: load_tag(f"s{sd}") for sd in SEEDS}
pct_pv = {sd: roll_pct(v) for sd, v in pv.items() if v is not None}

base_full = bt(pos_mode(tgt3(pct_slow)))
base_oos = bt(pos_mode(tgt3(pct_slow)), "2025-01-01")
print("=" * 100)
print("基准：只用 slow 因子（三档0.30/0.80 + 众数5 + 分位窗口750）")
print("=" * 100)
print(f"  2019-2026: 夏普 {base_full[0]:.3f} 年化 {base_full[1]:.2%}")
print(f"  2025-2026: 夏普 {base_oos[0]:.3f} 年化 {base_oos[1]:.2%}")

print("\n" + "=" * 100)
print("检验1+2：真实 pv 以权重 w 加入（6 个 seed，看下限与正交性）")
print("=" * 100)
print(f"  {'w_pv':<6}{'2019-26 夏普(6seed)':>34}{'2025-26 夏普':>30}")
rows = {}
for w in (0.25, 0.50):
    fulls, ooss = [], []
    for sd, pp in pct_pv.items():
        m = pd.DataFrame({"pv": pp, "sl": pct_slow})
        pct = w * m["pv"] + (1 - w) * m["sl"]
        f = bt(pos_mode(tgt3(pct))); o = bt(pos_mode(tgt3(pct)), "2025-01-01")
        fulls.append(f[0]); ooss.append(o[0])
    rows[w] = (np.array(fulls), np.array(ooss))
    print(f"  {w:<6}均值{np.mean(fulls):>8.3f} 标准差{np.std(fulls,ddof=1):>7.3f} "
          f"最小{np.min(fulls):>7.3f} 最大{np.max(fulls):>7.3f}"
          f"   均值{np.mean(ooss):>7.3f} 最小{np.min(ooss):>7.3f} 最大{np.max(ooss):>7.3f}")

print("\n" + "=" * 100)
print("检验3（关键）：安慰剂 —— 把 pv 信号随机平移，保留分布与自相关，只破坏时间对齐")
print("=" * 100)
rng = np.random.default_rng(20261025)
N_PLACEBO = 40
for w in (0.25, 0.50):
    sd0 = SEEDS[0]
    pp = pct_pv[sd0].dropna()
    vals = pp.values
    n = len(vals)
    pl_full, pl_oos = [], []
    for _ in range(N_PLACEBO):
        shift = int(rng.integers(120, n - 120))
        shifted = pd.Series(np.roll(vals, shift), index=pp.index)
        m = pd.DataFrame({"pv": shifted, "sl": pct_slow})
        pct = w * m["pv"] + (1 - w) * m["sl"]
        pl_full.append(bt(pos_mode(tgt3(pct)))[0])
        pl_oos.append(bt(pos_mode(tgt3(pct)), "2025-01-01")[0])
    pl_full, pl_oos = np.array(pl_full), np.array(pl_oos)
    real_f, real_o = rows[w]
    print(f"\n  --- w_pv = {w} ---")
    print(f"  安慰剂 2019-26 夏普: 均值{np.nanmean(pl_full):>7.3f} 标准差{np.nanstd(pl_full):>6.3f} "
          f"q25{np.nanquantile(pl_full,.25):>7.3f} q75{np.nanquantile(pl_full,.75):>7.3f} "
          f"max{np.nanmax(pl_full):>7.3f}")
    print(f"  真实 pv            : 均值{np.mean(real_f):>7.3f} 最小{np.min(real_f):>7.3f} 最大{np.max(real_f):>7.3f}")
    exceed = (real_f > np.nanquantile(pl_full, 0.9)).mean()
    print(f"  → 真实 seed 中超过安慰剂 90 分位的比例: {exceed:.0%}  "
          f"（基准夏普 {base_full[0]:.3f}，安慰剂均值 {np.nanmean(pl_full)-base_full[0]:+.3f}，"
          f"真实均值 {np.mean(real_f)-base_full[0]:+.3f}）")
    print(f"  安慰剂 2025-26 夏普: 均值{np.nanmean(pl_oos):>7.3f} 标准差{np.nanstd(pl_oos):>6.3f} "
          f"max{np.nanmax(pl_oos):>7.3f}")
    print(f"  真实 pv            : 均值{np.mean(real_o):>7.3f} 最小{np.min(real_o):>7.3f} 最大{np.max(real_o):>7.3f}")

print("\n" + "=" * 100)
print("附加：pv 与 slow 的相关性（正交性）")
print("=" * 100)
for sd, pp in pct_pv.items():
    m = pd.DataFrame({"pv": pp, "sl": pct_slow}).dropna()
    print(f"  seed{sd}: 分位后相关 {m['pv'].corr(m['sl']):+.3f}  (n={len(m)})")
