# -*- coding: utf-8 -*-
"""逐年归因：超额 = 躲跌收益 − 踏空损失 − 成本，并按 空仓/半仓 × 涨日/跌日 交叉拆解

另给：
  - 逐日贡献 Top 贡献/拖累日（帮定位到具体行情事件）
  - 择时效率 = 实际超额 / 完美择时超额（理论上限）
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


pv_avg = pd.concat([load_tag(f"s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()
sl_avg = pd.concat([load_tag(f"slow_s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()


def m3(W=300):
    m = pd.DataFrame({"p": roll_pct(pv_avg, W), "s": roll_pct(sl_avg, W)}).dropna()
    comp = 0.5 * m["p"] + 0.5 * stretch(m["s"], 0.30, 0.65)
    return pos_mode(tgt3(comp, 0.30, 0.80), 5)


pos = m3()
d = pd.DataFrame({"p": pos, "c": close}).dropna()
d = d[d.index >= "2019-06-01"]
r = np.log(d["c"]).diff().fillna(0)
turn = d["p"].diff().abs().fillna(0)
w = d["p"].shift(1).fillna(0)          # 当日生效仓位
delta = 1 - w                           # 少拿的仓位
cost = turn * COST
excess_daily = -delta * r - cost        # 逐日超额：= (w-1)r - cost
idx_daily = r
contrib = pd.DataFrame({"w": w, "r": r, "delta": delta, "cost": cost, "excess": excess_daily})
contrib["year"] = contrib.index.year

print("=" * 126)
print("逐年归因：超额 = 躲跌收益 − 踏空损失 − 成本   （单位：%，对数收益；向上为正贡献）")
print("=" * 126)
print(f"  {'年份':<6}{'指数':>8}{'策略':>8}{'超额':>8}│"
      f"{'躲跌收益':>9}{'踏空损失':>9}{'成本':>7}│"
      f"{'空仓·跌日':>10}{'半仓·跌日':>10}{'空仓·涨日':>10}{'半仓·涨日':>10}│"
      f"{'完美超额':>9}{'择时效率':>8}")
rows = []
for y, g in contrib.groupby("year"):
    idx = g["r"].sum()
    strat = (g["w"] * g["r"]).sum() - g["cost"].sum()
    exc = strat - idx
    up, dn = g["r"] > 0, g["r"] < 0
    # 躲跌：跌日少拿的仓位错过的跌幅（正贡献）
    hide = {}
    for lvl, msk in (("empty", g["w"] <= 0.01), ("half", (g["w"] > 0.01) & (g["w"] < 0.99))):
        hide[(lvl, "dn")] = float(-(g.loc[msk & dn, "delta"] * g.loc[msk & dn, "r"]).sum())
        hide[(lvl, "up")] = float((g.loc[msk & up, "delta"] * g.loc[msk & up, "r"]).sum())
    dodge = hide[("empty", "dn")] + hide[("half", "dn")]
    miss = hide[("empty", "up")] + hide[("half", "up")]
    cst = float(g["cost"].sum())
    perfect = float(g["r"].abs().sum())   # 完美择时：涨日满仓、跌日空仓
    eff = exc / perfect if perfect else np.nan
    rows.append(dict(year=y, idx=idx, strat=strat, exc=exc, dodge=dodge, miss=miss, cost=cst,
                     e_dn=hide[("empty", "dn")], h_dn=hide[("half", "dn")],
                     e_up=hide[("empty", "up")], h_up=hide[("half", "up")],
                     perfect=perfect, eff=eff))
    print(f"  {y:<6}{idx*100:>8.1f}{strat*100:>8.1f}{exc*100:>+8.1f}│"
          f"{dodge*100:>9.1f}{miss*100:>9.1f}{cst*100:>7.1f}│"
          f"{hide[('empty','dn')]*100:>10.1f}{hide[('half','dn')]*100:>10.1f}"
          f"{hide[('empty','up')]*100:>10.1f}{hide[('half','up')]*100:>10.1f}│"
          f"{perfect*100:>9.1f}{eff:>8.1%}")
A = pd.DataFrame(rows)
print(f"\n  合计  指数{A['idx'].sum()*100:>6.1f} 策略{A['strat'].sum()*100:>6.1f} 超额{A['exc'].sum()*100:>+6.1f} │"
      f" 躲跌{A['dodge'].sum()*100:>6.1f} 踏空{A['miss'].sum()*100:>6.1f} 成本{A['cost'].sum()*100:>5.1f}")

print("\n" + "=" * 126)
print("逐日贡献最大的日子（帮定位到具体行情事件）")
print("=" * 126)
for y, g in contrib.groupby("year"):
    g = g[g["w"] < 0.99]        # 只看减仓日（满仓日无超额）
    if len(g) == 0: continue
    top = g.nlargest(3, "excess")
    bot = g.nsmallest(3, "excess")
    f = lambda t: "  ".join(f"{d.strftime('%m-%d')}({v*100:+.2f}%)" for d, v in zip(t.index, t["excess"]))
    print(f"  {y}  最赚: {f(top)}")
    print(f"        最亏: {f(bot)}")

print("\n" + "=" * 126)
print("按年判断「为什么赢/为什么输」的自动结论")
print("=" * 126)
for _, x in A.iterrows():
    parts = []
    if x["exc"] > 0:
        parts.append(f"赢：躲跌 {x['dodge']*100:+.1f} > 踏空 {x['miss']*100:.1f}")
        if x["e_dn"] > x["h_dn"]:
            parts.append("主要靠空仓躲跌")
        else:
            parts.append("主要靠半仓减敞口")
    else:
        parts.append(f"输：踏空 {x['miss']*100:.1f} > 躲跌 {x['dodge']*100:.1f}")
        if x["h_up"] > x["e_up"]:
            parts.append("主因半仓踏空（该满仓只给半仓）")
        else:
            parts.append("主因空仓踏空（该在场却离场）")
    print(f"  {int(x['year'])}  " + "；".join(parts) + f"；择时效率 {x['eff']:.1%}")
