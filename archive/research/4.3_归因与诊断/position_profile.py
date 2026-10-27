# -*- coding: utf-8 -*-
"""M3 仓位画像：时间占比、持仓/空仓段结构、换仓明细、逐年仓位、交易成本"""
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
ANN, COST = 252, 0.0003


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
    return ((p - lo) / (hi - lo)).clip(0, 1)


def pos_mode(t, k=5):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w): continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


pv_avg = pd.concat([load_tag(f"s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()
sl_avg = pd.concat([load_tag(f"slow_s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()


def m3(W=300, w=0.5, k=5):
    m = pd.DataFrame({"p": roll_pct(pv_avg, W), "s": roll_pct(sl_avg, W)}).dropna()
    comp = w * m["p"] + (1 - w) * stretch(m["s"], 0.30, 0.65)
    return pos_mode(tgt3(comp, 0.30, 0.80), k)


def m0(W=750, w=0.5, k=5):
    m = pd.DataFrame({"p": roll_pct(pv_avg, W), "s": roll_pct(sl_avg, W)}).dropna()
    return pos_mode(tgt3(w * m["p"] + (1 - w) * m["s"], 0.30, 0.80), k)


def profile(pos, lbl, start="2019-01-01", end="2099-12-31"):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    p = d["p"].values
    n = len(p)
    yrs = n / ANN
    turn = np.abs(np.diff(p, prepend=p[0]))
    chg = turn > 1e-9
    print(f"\n--- {lbl} ---")
    print(f"  样本 {d.index.min().date()}~{d.index.max().date()}，{n} 个交易日（{yrs:.1f} 年）")
    print(f"  【时间占比】满仓 {(p>=.99).mean():.1%} | 半仓 {((p>.01)&(p<.99)).mean():.1%} | "
          f"空仓 {(p<=.01).mean():.1%} → 在仓时间占比 {(p>0.01).mean():.1%}")
    print(f"  【仓位水平】日均 {p.mean():.1%} | 中位 {np.median(p):.1%} | "
          f"仓位标准差 {p.std():.1%}")
    # 段结构
    hold = p > 0.01
    segs = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and hold[j + 1] == hold[i]: j += 1
        segs.append((hold[i], j - i + 1))
        i = j + 1
    H = [L for h, L in segs if h]
    E = [L for h, L in segs if not h]
    print(f"  【持仓段】{len(H)} 段 | 平均 {np.mean(H):.0f} 天 | 中位 {np.median(H):.0f} 天 | "
          f"最长 {max(H)} 天 | 最短 {min(H)} 天")
    print(f"  【空仓段】{len(E)} 段 | 平均 {np.mean(E):.0f} 天 | 中位 {np.median(E):.0f} 天 | "
          f"最长 {max(E)} 天 | 最短 {min(E)} 天")
    # 换仓
    print(f"  【换仓】总 {chg.sum()} 次 | 年化 {chg.sum()/yrs:.1f} 次 | "
          f"平均单次变动 {turn[chg].mean():.2f} 仓位 | 年化换手 {turn.sum()/yrs:.2f} 倍")
    print(f"  【交易成本】累计 {turn.sum()*COST:.2%}（年化 {turn.sum()/yrs*COST:.2%}）")
    # 转移矩阵
    tr = {}
    for a, b in zip(p[:-1], p[1:]):
        if abs(b - a) > 1e-9:
            tr[(round(a, 1), round(b, 1))] = tr.get((round(a, 1), round(b, 1)), 0) + 1
    top = sorted(tr.items(), key=lambda x: -x[1])[:6]
    print("  【主要换仓方向】" + "  ".join(f"{a:.1f}→{b:.1f}:{c}次" for (a, b), c in top))
    # 逐年
    print(f"  【逐年仓位】")
    print(f"    {'年份':<6}{'日均仓位':>9}{'满仓%':>8}{'半仓%':>8}{'空仓%':>8}{'换仓次数':>9}")
    dd = pd.Series(p, index=d.index)
    for y, g in dd.groupby(dd.index.year):
        v = g.values
        tc = int((np.abs(np.diff(v, prepend=v[0])) > 1e-9).sum())
        print(f"    {y:<6}{v.mean():>9.1%}{(v>=.99).mean():>8.0%}"
              f"{((v>.01)&(v<.99)).mean():>8.0%}{(v<=.01).mean():>8.0%}{tc:>9}")
    return d, p


profile(m3(), "M3（最终方案）")
profile(m0(), "M0（原方案）")
profile(m3(), "M3 —— 2025-2026 样本外", start="2025-01-01")
