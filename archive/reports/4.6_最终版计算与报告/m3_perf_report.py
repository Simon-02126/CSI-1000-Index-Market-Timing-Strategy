# -*- coding: utf-8 -*-
"""M3 最终方案完整绩效报告（含交易级指标、捕获率、对照基准、36seed 分布）"""
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


def m3_pos(pv, sl, W=300, w=0.5, k=5):
    m = pd.DataFrame({"p": roll_pct(pv, W), "s": roll_pct(sl, W)}).dropna()
    comp = w * m["p"] + (1 - w) * stretch(m["s"], 0.30, 0.65)
    return pos_mode(tgt3(comp, 0.30, 0.80), k)


def m0_pos(pv, sl, W=750, w=0.5, k=5):
    m = pd.DataFrame({"p": roll_pct(pv, W), "s": roll_pct(sl, W)}).dropna()
    return pos_mode(tgt3(w * m["p"] + (1 - w) * m["s"], 0.30, 0.80), k)


def perf(pos, start="2019-01-01", end="2099-12-31"):
    d = pd.DataFrame({"p": pos, "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
    if len(d) < 60: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * COST).dropna()
    gross = (d["p"].shift(1) * r).dropna()
    eq = st.cumsum(); peak = eq.cummax(); dd = eq - peak
    mdd = float(np.expm1(dd.min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    # 最长回撤持续
    under = dd < -1e-9
    longest, cur = 0, 0
    for u in under.values:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    # 交易级（按连续持仓段）
    p = d["p"].values
    segs, i = [], 0
    while i < len(p):
        if p[i] > 1e-9:
            j = i
            while j + 1 < len(p) and p[j + 1] > 1e-9: j += 1
            segs.append(st.iloc[max(i - 1, 0):j].sum())
            i = j + 1
        else:
            i += 1
    segs = np.array(segs)
    win = (segs > 0).mean() if len(segs) else np.nan
    gain = segs[segs > 0].mean() if (segs > 0).any() else np.nan
    loss = -segs[segs < 0].mean() if (segs < 0).any() else np.nan
    odds = gain / loss if loss and loss > 0 else np.nan
    nchg = int((turn > 1e-9).sum())
    return dict(cum=float(np.expm1(eq.iloc[-1])), ann=ann, vol=vol, sharpe=ann / vol if vol else np.nan,
                mdd=mdd, calmar=ann / abs(mdd) if mdd else np.nan,
                dd_days=longest, win=win, odds=odds, nseg=len(segs),
                avg_hold=len(d) / max(len(segs), 1), chg_yr=nchg / (len(d) / ANN),
                avg_pos=float(d["p"].mean()), full=(d["p"] >= 0.99).mean(),
                empty=(d["p"] <= 0.01).mean(), byyear=st.groupby(st.index.year).sum(),
                strat=st, bh=r.dropna())


def row(lbl, p, pb=None):
    if p is None:
        print(f"  {lbl:<22} 样本不足"); return
    print(f"  {lbl:<22}{p['cum']:>9.1%}{p['ann']:>9.1%}{p['vol']:>8.1%}{p['sharpe']:>7.2f}"
          f"{p['mdd']:>9.1%}{p['calmar']:>7.2f}{p['dd_days']:>9}{p['win']:>7.1%}"
          f"{p['odds']:>7.2f}{p['avg_hold']:>8.0f}{p['chg_yr']:>7.1f}{p['avg_pos']:>7.0%}")


pos3 = m3_pos(pv_avg, sl_avg)
pos0 = m0_pos(pv_avg, sl_avg)
pos_sl = pos_mode(tgt3(roll_pct(sl_avg, 750)), 5)
pos_pv = pos_mode(tgt3(roll_pct(pv_avg, 200)), 5)
pos_bh = pd.Series(1.0, index=close.index)

for start, end, lbl in [("2019-01-01", "2099-12-31", "全样本 2019-2026"),
                        ("2019-01-01", "2024-12-31", "2019-2024"),
                        ("2025-01-01", "2099-12-31", "2025-2026 样本外")]:
    print("\n" + "=" * 132)
    print(f"绩效指标 —— {lbl}")
    print("=" * 132)
    print(f"  {'方案':<22}{'累计':>9}{'年化':>9}{'波动':>8}{'夏普':>7}{'回撤':>9}{'Calmar':>7}"
          f"{'回撤天数':>9}{'胜率':>7}{'赔率':>7}{'持仓天数':>8}{'换仓/年':>7}{'均仓':>7}")
    row("M3 (最终方案)", perf(pos3, start, end))
    row("M0 (原方案)", perf(pos0, start, end))
    row("仅 slow", perf(pos_sl, start, end))
    row("仅 pv", perf(pos_pv, start, end))
    row("买入持有", perf(pos_bh, start, end))

print("\n" + "=" * 132)
print("逐年收益（M3 vs 买入持有）")
print("=" * 132)
a, b = perf(pos3), perf(pos_bh)
print(f"  {'年份':<8}{'M3':>10}{'买入持有':>10}{'超额':>10}")
for y in sorted(a["byyear"].index):
    x, yb = a["byyear"].get(y, 0), b["byyear"].get(y, 0)
    print(f"  {y:<8}{x:>9.1%}{yb:>10.1%}{(x-yb):>10.1%}")

print("\n" + "=" * 132)
print("月度统计（M3，2019-2026）")
print("=" * 132)
s3 = perf(pos3)["strat"]
m = s3.groupby(s3.index.to_period("M")).sum()
print(f"  月度胜率 {(m>0).mean():.0%}（{len(m)} 个月，正 {(m>0).sum()} / 负 {(m<0).sum()}）")
print(f"  最好月 {m.max():.1%}  最差月 {m.min():.1%}  月均 {m.mean():.2%}")
bh_m = perf(pos_bh)["strat"].groupby(lambda d: d.to_period("M")).sum()
up = bh_m > 0
print(f"  上涨月({up.sum()}个) 平均 {m[up].mean():.2%} vs 买持 {bh_m[up].mean():.2%} → 上行捕获 {m[up].mean()/bh_m[up].mean():.0%}")
dn = bh_m <= 0
print(f"  下跌月({dn.sum()}个) 平均 {m[dn].mean():.2%} vs 买持 {bh_m[dn].mean():.2%} → 下行捕获 {m[dn].mean()/bh_m[dn].mean():.0%}")

print("\n" + "=" * 132)
print("36 组 seed 配对分布（M3 vs M0，全样本夏普）")
print("=" * 132)
for nm, f in [("M3", m3_pos), ("M0", m0_pos)]:
    vals = []
    for ps in pvS.values():
        for ss in slS.values():
            r = perf(f(ps, ss))
            if r: vals.append(r["sharpe"])
    v = np.array(vals)
    print(f"  {nm}: 均值{v.mean():>6.2f} 中位{np.median(v):>6.2f} std{v.std(ddof=1):>5.2f} "
          f"最小{v.min():>5.2f} 25%{np.quantile(v,.25):>5.2f} 75%{np.quantile(v,.75):>5.2f} 最大{v.max():>5.2f}")
