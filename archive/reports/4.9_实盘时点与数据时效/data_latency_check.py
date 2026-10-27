# -*- coding: utf-8 -*-
"""数据到手时点敏感性：各块数据晚一天，策略还成不成立"""
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
IFp_raw = roll_pct(raw_if, W_SENT, minp=60)


def build(pv_sig, sl_sig, if_sig):
    mm = pd.DataFrame({"p": roll_pct(pv_sig, W), "s": roll_pct(sl_sig, W)}).dropna()
    mm["s_sat"] = ((mm["s"] - 0.30) / 0.35).clip(0, 1)
    mm["comp"] = 0.5 * mm["p"] + 0.5 * mm["s_sat"]
    base = pos_mode(tgt3(mm["comp"], 0.30, 0.80), K)
    d = pd.DataFrame({"base": base, "IF": if_sig.reindex(base.index).ffill(),
                      "c": close, "o": open_}).dropna()
    d = d[(d.index >= pd.Timestamp("2019-06-04")) & (d.index <= pd.Timestamp("2026-09-30"))].copy()
    half = (d["base"] - 0.5).abs() < 1e-9
    d["A0"] = d["base"].where(~half, d["IF"])
    d["P"] = band(grid(d["A0"], GRID), DEAD)
    d["g"] = np.log(d["o"] / d["c"].shift(1))
    d["it"] = np.log(d["c"] / d["o"])
    d["r"] = np.log(d["c"] / d["c"].shift(1))
    d.loc[d.index[0], ["g", "it", "r"]] = [0.0, 0.0, 0.0]
    P = d["P"]
    cost = P.diff().abs().fillna(0.0) * BP
    rB = P.shift(2) * d["g"] + P.shift(1) * d["it"]          # T+1 开盘成交
    return d, rB.fillna(0.0) - cost


def stat(net, nm):
    cum = float(np.expm1(net.sum())) * 100
    ann = float(net.mean() * ANN) * 100
    vol = float(net.std() * np.sqrt(ANN)) * 100
    eq = np.exp(net.cumsum())
    mdd = float((eq / eq.cummax()).min() - 1) * 100
    return dict(名=nm, 累计=cum, 年化=ann, 波动=vol, 夏普=ann / vol, 回撤=mdd, Calmar=ann / -mdd)


cases = [
    ("基准：所有数据 T 日盘后到手", pv_avg, sl_avg, IFp_raw),
    ("慢一天：两融 T+1 才到手（slow 滞后 1 日）", pv_avg, sl_avg.shift(1), IFp_raw),
    ("慢一天：会员持仓 T+1 才到手（IF 分位滞后 1 日）", pv_avg, sl_avg, IFp_raw.shift(1)),
    ("慢一天：两块都滞后 1 日", pv_avg, sl_avg.shift(1), IFp_raw.shift(1)),
    ("慢两天：两融 T+2 才到手（slow 滞后 2 日）", pv_avg, sl_avg.shift(2), IFp_raw),
    ("更保守：pv 也用 T-1 的（量价滞后 1 日）", pv_avg.shift(1), sl_avg, IFp_raw),
]
print("=" * 112)
print("数据到手时点敏感性（统一用 T+1 开盘成交口径）")
print("=" * 112)
print(f"{'情形':<44}{'累计%':>10}{'年化%':>9}{'波动%':>8}{'夏普':>8}{'回撤%':>9}{'Calmar':>8}")
base_sh = None
for nm, a, b, c in cases:
    _, net = build(a, b, c)
    s = stat(net, nm)
    if base_sh is None:
        base_sh = s["夏普"]
        print(f"{nm:<44}{s['累计']:>10.1f}{s['年化']:>9.2f}{s['波动']:>8.2f}{s['夏普']:>8.3f}{s['回撤']:>9.1f}{s['Calmar']:>8.2f}")
    else:
        print(f"{nm:<44}{s['累计']:>10.1f}{s['年化']:>9.2f}{s['波动']:>8.2f}{s['夏普']:>8.3f}{s['回撤']:>9.1f}{s['Calmar']:>8.2f}"
              f"   Δ夏普 {s['夏普']-base_sh:+.3f}")
print("=" * 112)

# 两融披露滞后对 slow 信号本身的影响
print()
print("slow 信号自身对 1 日滞后的敏感度：")
a_, b_ = sl_avg.dropna(), sl_avg.shift(1).dropna()
j = pd.concat([a_, b_], axis=1).dropna()
print(f"  corr(slow_t, slow_t-1) = {j.corr().iloc[0,1]:.4f}")
pa = roll_pct(a_, W).dropna(); pb = roll_pct(b_, W).dropna()
jj = pd.concat([pa, pb], axis=1).dropna()
print(f"  corr(分位_t, 分位_t-1)  = {jj.corr().iloc[0,1]:.4f}，平均绝对差 {jj.diff(axis=1).iloc[:,1].abs().mean():.4f}")
