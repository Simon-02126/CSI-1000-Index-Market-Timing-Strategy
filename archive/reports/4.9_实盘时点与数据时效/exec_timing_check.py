# -*- coding: utf-8 -*-
"""执行时点检验：T收盘成交（回测口径） vs T+1开盘成交（实盘唯一可执行）"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob
import numpy as np
import pandas as pd

BASE = _P.ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = _P.TRAIN
INPUT = os.path.join(BASE, "input")
ANN, BP = 252, 0.0003
W, W_SENT, K = 300, 250, 5
GRID, DEAD = 0.05, 0.10


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


fp = glob.glob(os.path.join(INPUT, "*中证1000*.csv"))[0]
_df = read_any(fp)
dc = [c for c in _df.columns if "日期" in c][0]
cc = [c for c in _df.columns if "收盘" in c][0]
oc = [c for c in _df.columns if "开盘" in c][0]
print("列名：", dc, cc, oc)
idx = pd.to_datetime(_df[dc])
def _ser(col):
    s = pd.Series(pd.to_numeric(_df[col], errors="coerce").values, index=idx)
    return s[~s.index.duplicated(keep="last")].sort_index().dropna()
close, open_ = _ser(cc), _ser(oc)

panel = pd.read_csv(os.path.join(INPUT, "情绪_面板_日频.csv"), encoding="utf-8-sig", parse_dates=["日期"])
panel = panel.drop_duplicates("日期").set_index("日期").sort_index()
raw_if = panel["IF_会员多空比"].dropna()


def load_seed(tag, sd):
    parts = []
    for p in sorted(glob.glob(os.path.join(TRAIN, f"{tag}{sd}_*.csv"))):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def load_tag(tag):
    return pd.concat([v for v in (load_seed(tag, sd) for sd in range(6)) if v is not None],
                     axis=1).mean(axis=1).dropna()


def roll_pct(s, win, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    mp = min(minp, max(60, win // 3))
    for i in range(len(v)):
        lo = max(0, i - win)
        if i < mp or i - lo < mp:
            continue
        w = v[lo:i]; w = w[~np.isnan(w)]
        if len(w) < mp or np.isnan(v[i]):
            continue
        o[i] = (w < v[i]).sum() / len(w)
    return pd.Series(o, index=s.index)


def pos_mode(t, k=K):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w):
            continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def tgt3(comp, lo=0.30, hi=0.80):
    return pd.Series(np.where(comp < lo, 0.0, np.where(comp < hi, 0.5, 1.0)),
                     index=comp.index).where(comp.notna())


def grid(p, q):
    return p if q <= 0 else (p / q).round().clip(0, 1 / q) * q


def band(p, bd):
    v = p.values.astype(float); out = np.full(len(v), np.nan); cur = v[0]
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = cur; continue
        if abs(v[i] - cur) >= bd - 1e-12:
            cur = v[i]
        out[i] = cur
    return pd.Series(out, index=p.index)


pv_avg, sl_avg = load_tag("s"), load_tag("slow_s")
mm = pd.DataFrame({"p": roll_pct(pv_avg, W), "s": roll_pct(sl_avg, W)}).dropna()
mm["s_sat"] = ((mm["s"] - 0.30) / 0.35).clip(0, 1)
mm["comp"] = 0.5 * mm["p"] + 0.5 * mm["s_sat"]
base = pos_mode(tgt3(mm["comp"], 0.30, 0.80), K)

d = pd.DataFrame({"base": base, "IF": roll_pct(raw_if, W_SENT, minp=60).reindex(base.index).ffill(),
                  "c": close, "o": open_}).dropna()
d = d[(d.index >= pd.Timestamp("2019-06-04")) & (d.index <= pd.Timestamp("2026-09-30"))]
half = (d["base"] - 0.5).abs() < 1e-9
d["A0"] = d["base"].where(~half, d["IF"])
d["P"] = band(grid(d["A0"], GRID), DEAD)
N_ = len(d); YR = N_ / ANN

# 收益拆成「隔夜跳空 gap」+「日内 intra」
d["g"] = np.log(d["o"] / d["c"].shift(1))     # close_{t-1} -> open_t
d["it"] = np.log(d["c"] / d["o"])             # open_t -> close_t
d["r"] = np.log(d["c"] / d["c"].shift(1))     # = g + it
d.loc[d.index[0], ["g", "it", "r"]] = [0.0, 0.0, 0.0]

P = d["P"]
dP = P.diff().abs().fillna(0.0)
cost = dP * BP

# A：T 日收盘成交（回测口径）—— T 收盘定的仓位吃 T+1 全天收益
rA = P.shift(1) * d["r"]
# B：T+1 开盘成交（实盘）—— 隔夜跳空段仍持旧仓，开盘后才换成新仓
rB = P.shift(2) * d["g"] + P.shift(1) * d["it"]
# C：T+1 收盘成交（最保守，之前测过的 lag=1）
rC = P.shift(2) * d["r"]

# 基准：买入持有
rBH = d["r"]


def stat(r, nm, cst=None):
    r = r.fillna(0.0)
    net = r - (cost if cst is None else cst)
    cum = float(np.expm1(net.sum()))
    ann = float(net.mean() * ANN)
    vol = float(net.std() * np.sqrt(ANN))
    sh = ann / vol if vol > 0 else np.nan
    eq = np.exp(net.cumsum())
    mdd = float((eq / eq.cummax()).min() - 1)
    return dict(名=nm, 累计=cum * 100, 年化=ann * 100, 波动=vol * 100, 夏普=sh,
                回撤=mdd * 100, Calmar=(ann / -mdd) if mdd < 0 else np.nan)


print("=" * 108)
print(f"回测区间 {d.index[0].date()} ~ {d.index[-1].date()}，{N_} 个交易日 / {YR:.2f} 年，单边 {BP*1e4:.0f}bp")
print("=" * 108)
rows = [stat(rBH, "买入持有（基准）", pd.Series(0.0, index=d.index)),
        stat(rA, "A  T日收盘成交（=回测口径）"),
        stat(rB, "B  T+1开盘成交（实盘唯一可执行）"),
        stat(rC, "C  T+1收盘成交（最保守）")]
print(f"{'方案':<34}{'累计%':>10}{'年化%':>9}{'波动%':>8}{'夏普':>8}{'回撤%':>9}{'Calmar':>8}")
for r in rows:
    print(f"{r['名']:<34}{r['累计']:>10.1f}{r['年化']:>9.2f}{r['波动']:>8.2f}{r['夏普']:>8.3f}{r['回撤']:>9.1f}{r['Calmar']:>8.2f}")
print()
print(f"A→B（改到 T+1 开盘成交）：Δ夏普 {rows[2]['夏普']-rows[1]['夏普']:+.3f}，"
      f"Δ年化 {rows[2]['年化']-rows[1]['年化']:+.2f}pct，Δ回撤 {rows[2]['回撤']-rows[1]['回撤']:+.2f}pct")
print(f"A→C（再晚到 T+1 收盘）  ：Δ夏普 {rows[3]['夏普']-rows[1]['夏普']:+.3f}，"
      f"Δ年化 {rows[3]['年化']-rows[1]['年化']:+.2f}pct，Δ回撤 {rows[3]['回撤']-rows[1]['回撤']:+.2f}pct")

# 跳空效应的直接度量
gapdiff = (P.shift(2) - P.shift(1)) * d["g"]
chg = dP > 1e-12
print()
print("=" * 108)
print("隔夜跳空到底伤不伤：只对「调仓日」看 (旧仓-新仓) × 当日跳空")
sub = pd.DataFrame({"dP": dP, "g": d["g"], "gapdiff": gapdiff,
                    "nw": P.shift(1), "od": P.shift(2)})
sub = sub[(dP > 1e-12)]
add = sub[sub["nw"] > sub["od"]]      # 加仓：新仓 > 旧仓
red = sub[sub["nw"] < sub["od"]]      # 减仓：新仓 < 旧仓
print(f"  调仓日 {len(sub)} 天，跳空项合计 {sub['gapdiff'].sum()*100:+.2f} log%")
print(f"  加仓日 {len(add)} 天（旧仓<新仓，跳空涨则占便宜）：跳空项 {add['gapdiff'].sum()*100:+.2f} log%，"
      f"平均跳空 {add['g'].mean()*100:+.3f}%")
print(f"  减仓日 {len(red)} 天（旧仓>新仓，跳空跌则占便宜）：跳空项 {red['gapdiff'].sum()*100:+.2f} log%，"
      f"平均跳空 {red['g'].mean()*100:+.3f}%")
print(f"  全样本日均跳空 {d['g'].mean()*100:+.4f}%（年化 {d['g'].mean()*ANN*100:+.2f}%），"
      f"日内日均 {d['it'].mean()*100:+.4f}%（年化 {d['it'].mean()*ANN*100:+.2f}%）")
print("=" * 108)
