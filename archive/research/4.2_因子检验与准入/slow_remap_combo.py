# -*- coding: utf-8 -*-
"""slow 映射改造后的组合回测

M0 现行     : comp = 0.5*pv_pct + 0.5*slow_pct; 三档(0.30,0.80)
M1 slow三档hi=0.65 : 两条腿各自映射成仓位后加权（slow 用自己的 hi）
M2 slow线性(0.2/0.4): slow' = clip((slow_pct-0.2)/0.4,0,1)，再与 pv_pct 合成
M3 slow拉伸hi=0.65 : slow' = clip((slow_pct-0.3)/0.35,0,1)，再与 pv_pct 合成
另含 36 组 seed 配对分布（6 pv seed × 6 slow seed）
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


def stretch(p, lo, hi):
    """把分位 p 线性拉伸到 [0,1]，即 hi 以上视为满分"""
    return ((p - lo) / (hi - lo)).clip(0, 1)


def pos_mode(t, k=5):
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


pvS = {sd: load_tag(f"s{sd}") for sd in range(6)}
slS = {sd: load_tag(f"slow_s{sd}") for sd in range(6)}
pvS = {k: v for k, v in pvS.items() if v is not None}
slS = {k: v for k, v in slS.items() if v is not None}
pv_avg = pd.concat(pvS.values(), axis=1).mean(axis=1).dropna()
sl_avg = pd.concat(slS.values(), axis=1).mean(axis=1).dropna()


def build(pv_pct, sl_pct, mode, w=0.5, lo=0.30, hi=0.80, k=5):
    """返回仓位序列"""
    m = pd.DataFrame({"p": pv_pct, "s": sl_pct}).dropna()
    if mode == "M0":                       # 现行：分位平均后三档
        comp = w * m["p"] + (1 - w) * m["s"]
        return pos_mode(tgt3(comp, lo, hi), k)
    if mode == "M1":                       # 两条腿各自映射后加权（slow 用 hi=0.65）
        pos = w * tgt3(m["p"], lo, hi) + (1 - w) * tgt3(m["s"], lo, 0.65)
        return pos_mode(pos, k)
    if mode == "M2":                       # slow 线性 (0.2/0.4) 后与 pv 合成
        s2 = stretch(m["s"], 0.20, 0.60)
        comp = w * m["p"] + (1 - w) * s2
        return pos_mode(tgt3(comp, lo, hi), k)
    if mode == "M3":                       # slow 拉伸到 hi=0.65 后与 pv 合成
        s2 = stretch(m["s"], 0.30, 0.65)
        comp = w * m["p"] + (1 - w) * s2
        return pos_mode(tgt3(comp, lo, hi), k)
    if mode == "M4":                       # slow 拉伸到 hi=0.70
        s2 = stretch(m["s"], 0.30, 0.70)
        comp = w * m["p"] + (1 - w) * s2
        return pos_mode(tgt3(comp, lo, hi), k)


def ev(pos):
    f = bt(pos); r24 = bt(pos, "2019-01-01", "2024-12-31")
    y25 = bt(pos, "2025-01-01", "2025-12-31"); y26 = bt(pos, "2026-01-01")
    return f, r24, y25, y26


print("=" * 118)
print("A. 映射改造对比（w_pv=0.5, k=5；PCT 为两腿分位平均，均用 6seed 平均信号）")
print("=" * 118)
print(f"  {'方案':<44}{'W':>5}{'全期':>7}{'19-24':>7}{'2025':>7}{'2026':>7}"
      f"{'年化':>8}{'回撤':>8}{'Calmar':>8}{'均仓':>6}")
for mode, desc in [("M0", "M0 现行：分位平均→三档(0.30/0.80)"),
                   ("M1", "M1 两腿各映射→加权（slow hi=0.65）"),
                   ("M2", "M2 slow 线性 clip((p-0.2)/0.4) 后合成"),
                   ("M3", "M3 slow 拉伸 hi=0.65 后合成"),
                   ("M4", "M4 slow 拉伸 hi=0.70 后合成")]:
    for W in (300, 750):
        pos = build(roll_pct(pv_avg, W), roll_pct(sl_avg, W), mode)
        if pos is None: continue
        f, r24, y25, y26 = ev(pos)
        print(f"  {desc:<44}{W:>5}{f['sharpe']:>7.2f}{r24['sharpe']:>7.2f}"
              f"{y25['sharpe']:>7.2f}{y26['sharpe']:>7.2f}{f['ann']:>8.1%}"
              f"{f['mdd']:>8.1%}{f['calmar']:>8.2f}{f['avg']:>6.0%}")
    print()

print("=" * 118)
print("B. 权重扫描（W=300 / 750，k=5）")
print("=" * 118)
for W in (300, 750):
    print(f"\n  --- W={W} ---")
    print(f"  {'方案':<8}{'w=0.3':>16}{'w=0.4':>16}{'w=0.5':>16}{'w=0.6':>16}{'w=0.7':>16}")
    for mode in ("M0", "M1", "M2", "M3", "M4"):
        row = []
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            pos = build(roll_pct(pv_avg, W), roll_pct(sl_avg, W), mode, w=w)
            f, _, _, y26 = ev(pos)
            row.append(f"{f['sharpe']:>6.2f}/{y26['sharpe']:>+5.2f}")
        print(f"  {mode:<8}" + "".join(f"{x:>16}" for x in row))
    print("  (格式: 全期夏普/2026夏普)")

print("\n" + "=" * 118)
print("C. 36 组 seed 配对分布（6 pv seed × 6 slow seed，w=0.5, k=5）")
print("=" * 118)
for W in (300, 750):
    for mode in ("M0", "M2", "M3"):
        vals = []
        for ps in pvS.values():
            pvp = roll_pct(ps, W)
            for ss in slS.values():
                pos = build(pvp, roll_pct(ss, W), mode)
                r = bt(pos)
                if r: vals.append(r["sharpe"])
        v = np.array(vals)
        print(f"  W={W:<4}{mode}  均值{v.mean():>6.2f} 标准差{v.std(ddof=1):>5.2f} "
              f"最小{v.min():>5.2f} 25%{np.quantile(v,.25):>5.2f} 75%{np.quantile(v,.75):>5.2f}")

print("\n" + "=" * 118)
print("D. 逐年收益（W=300, w=0.5, k=5）")
print("=" * 118)
res = {}
for mode in ("M0", "M2", "M3"):
    f, _, _, _ = ev(build(roll_pct(pv_avg, 300), roll_pct(sl_avg, 300), mode))
    res[mode] = f
yrs = sorted(res["M0"]["byyear"].index)
print("  年份      M0现行     M2(slow线性)  M3(slow拉伸)")
for y in yrs:
    print(f"  {y}    " + "".join(f"{res[m]['byyear'].get(y,0):>12.1%}" for m in ("M0", "M2", "M3")))
print("  累计    " + "".join(f"{res[m]['byyear'].sum():>12.1%}" for m in ("M0", "M2", "M3")))
