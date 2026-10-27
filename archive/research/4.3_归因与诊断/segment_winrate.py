# -*- coding: utf-8 -*-
"""分段胜率：把时间轴按「持仓段 / 空仓段」全部切开，每段比较「策略收益 vs 指数收益」
胜 = 该段策略收益 > 指数收益（跑赢基准）
同时给出：分类型胜率、日频口径胜率、逐段超额
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


pvS = {sd: load_tag(f"s{sd}") for sd in range(6)}
slS = {sd: load_tag(f"slow_s{sd}") for sd in range(6)}
pvS = {k: v for k, v in pvS.items() if v is not None}
slS = {k: v for k, v in slS.items() if v is not None}
pv_avg = pd.concat(pvS.values(), axis=1).mean(axis=1).dropna()
sl_avg = pd.concat(slS.values(), axis=1).mean(axis=1).dropna()


def m3_pos(W=300):
    m = pd.DataFrame({"p": roll_pct(pv_avg, W), "s": roll_pct(sl_avg, W)}).dropna()
    comp = 0.5 * m["p"] + 0.5 * stretch(m["s"], 0.30, 0.65)
    return pos_mode(tgt3(comp, 0.30, 0.80), 5)


def m0_pos():
    m = pd.DataFrame({"p": roll_pct(pv_avg, 750), "s": roll_pct(sl_avg, 750)}).dropna()
    return pos_mode(tgt3(0.5 * m["p"] + 0.5 * m["s"], 0.30, 0.80), 5)


def segment_stats(pos, start="2019-01-01", end="2099-12-31", thr=0.01):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    r = np.log(d["c"]).diff().fillna(0)
    turn = d["p"].diff().abs().fillna(0)
    strat = d["p"].shift(1).fillna(0) * r - turn * COST
    idxr = r
    p = d["p"].values
    holding = p > thr
    # 切段
    segs = []
    i = 0
    n = len(p)
    while i < n:
        j = i
        while j + 1 < n and holding[j + 1] == holding[i]:
            j += 1
        segs.append((i, j, holding[i]))
        i = j + 1
    rows = []
    for a, b, hold in segs:
        # 段内收益：用 a..b 的日收益（a 日收益归属前一日持仓，简化为段内直接加总）
        rs = strat.iloc[a:b + 1].sum()
        ri = idxr.iloc[a:b + 1].sum()
        rows.append(dict(type="持仓" if hold else "空仓", days=b - a + 1,
                         strat=rs, index=ri, excess=rs - ri, win=(rs > ri)))
    S = pd.DataFrame(rows)
    return S, strat, idxr


print("=" * 108)
print("分段胜率定义：把时间轴按「持仓段 / 空仓段」全部切开，每段比较策略收益 vs 指数收益")
print("=" * 108)
for nm, pos in [("M3（最终方案）", m3_pos()), ("M0（原方案）", m0_pos())]:
    S, strat, idxr = segment_stats(pos)
    tot = len(S)
    print(f"\n--- {nm} ---")
    print(f"  总段数 {tot}（持仓 {sum(S['type']=='持仓')} 段 / 空仓 {sum(S['type']=='空仓')} 段）")
    print(f"  {'类型':<6}{'段数':>6}{'平均天数':>9}{'胜(跑赢指数)':>13}{'胜率':>8}"
          f"{'平均策略':>10}{'平均指数':>10}{'平均超额':>10}")
    for tp in ("持仓", "空仓", "全部"):
        g = S if tp == "全部" else S[S["type"] == tp]
        if len(g) == 0: continue
        print(f"  {tp:<6}{len(g):>6}{g['days'].mean():>9.0f}{int(g['win'].sum()):>13}"
              f"{g['win'].mean():>8.1%}{g['strat'].mean():>10.2%}{g['index'].mean():>10.2%}"
              f"{g['excess'].mean():>10.2%}")
    # 日频口径
    day_win = (strat > idxr).mean()
    print(f"  日频口径（每日比较）：胜率 {day_win:.1%}（{int((strat>idxr).sum())}/{len(strat)} 天）")
    # 长度加权（按天数）
    w = S["days"] / S["days"].sum()
    print(f"  按天数加权的分段胜率：{(S['win']*w).sum():.1%}")
    # 持仓段内再细分（满仓/半仓）
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[d.index >= pd.Timestamp("2019-01-01")]
    r = np.log(d["c"]).diff().fillna(0)
    turn = d["p"].diff().abs().fillna(0)
    strat2 = d["p"].shift(1).fillna(0) * r - turn * COST
    for lo, hi, lbl in [(0.49, 0.51, "半仓"), (0.99, 1.01, "满仓")]:
        mask = (d["p"] >= lo) & (d["p"] <= hi)
        if mask.sum() > 50:
            sub_s, sub_i = strat2[mask], r[mask]
            print(f"  其中{lbl}期间：{mask.sum()} 天，日胜率 {(sub_s>sub_i).mean():.1%}，"
                  f"日均策略 {sub_s.mean():.3%} vs 指数 {sub_i.mean():.3%}")

print("\n" + "=" * 108)
print("M3 全部段明细（前 12 段 + 后 12 段，单位 %）")
print("=" * 108)
S, _, _ = segment_stats(m3_pos())
S["days"] = S["days"].astype(int)
show = pd.concat([S.head(12), S.tail(12)])
print(f"  {'#':>3}{'类型':<6}{'天数':>5}{'策略':>9}{'指数':>9}{'超额':>9}{'跑赢':>6}")
for i, row in show.iterrows():
    print(f"  {i:>3}{row['type']:<6}{row['days']:>5}{row['strat']*100:>8.2f}%{row['index']*100:>8.2f}%"
          f"{row['excess']*100:>8.2f}%{'✓' if row['win'] else '✗':>6}")
