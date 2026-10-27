# -*- coding: utf-8 -*-
"""最终版（A0 + 5%网格 + 10%死区）的稳健性：三段增量 / 安慰剂 / 36seed"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = _P.ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = _P.TRAIN
INPUT = os.path.join(BASE, "input")
ANN, BP = 252, 0.0003
W, W_SENT, K, GRID, DEAD = 300, 250, 5, 0.05, 0.10


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
close = pd.Series(pd.to_numeric(_df[cc], errors="coerce").values, index=pd.to_datetime(_df[dc]))
close = close[~close.index.duplicated(keep="last")].sort_index().dropna()
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


def tgt3(c, lo=0.30, hi=0.80):
    return pd.Series(np.where(c < lo, 0.0, np.where(c < hi, 0.5, 1.0)), index=c.index).where(c.notna())


def grid(p, q=GRID):
    return (p / q).round().clip(0, 1 / q) * q


def band(p, bd=DEAD):
    v = p.values.astype(float); out = np.full(len(v), np.nan); cur = v[0]
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = cur; continue
        if abs(v[i] - cur) >= bd - 1e-12:
            cur = v[i]
        out[i] = cur
    return pd.Series(out, index=p.index)


def make(IF_, pv, sl):
    mm = pd.DataFrame({"p": roll_pct(pv, W), "s": roll_pct(sl, W)}).dropna()
    comp = 0.5 * mm["p"] + 0.5 * ((mm["s"] - 0.30) / 0.35).clip(0, 1)
    b = pos_mode(tgt3(comp, 0.30, 0.80), K)
    d = pd.DataFrame({"base": b, "IF": IF_.reindex(b.index).ffill(), "c": close}).dropna()
    d = d[(d.index >= pd.Timestamp("2019-06-04")) & (d.index <= pd.Timestamp("2026-09-30"))]
    d["r"] = np.log(d["c"]).diff().fillna(0.0)
    half = (d["base"] - 0.5).abs() < 1e-9
    d["A0"] = d["base"].where(~half, d["IF"])
    d["PRAC"] = band(grid(d["A0"]))
    return d


def sh(p, r, cost=BP):
    u = pd.Series(p.shift(1) * r - p.diff().abs().fillna(0) * cost).dropna()
    return u.mean() / u.std() * np.sqrt(ANN) if u.std() else np.nan


IFp = roll_pct(raw_if, W_SENT, minp=60)
PV_M = load_tag("s")
SL_M = load_tag("slow_s")
d = make(IFp, PV_M, SL_M)
print("=" * 110)
print("一、三段增量（夏普）")
print("=" * 110)
print(f"  {'区间':<26}{'基准':>9}{'A0连续':>10}{'Δ':>8}{'最终版':>10}{'Δ':>8}{'正/负':>7}")
for lbl, s0, s1 in (("IS 2019-06~2022-12", "2019-06-04", "2022-12-31"),
                    ("OOS 2023-01~2026-09", "2023-01-01", "2026-09-30"),
                    ("OOS 2025-01~2026-09", "2025-01-01", "2026-09-30")):
    m = (d.index >= s0) & (d.index <= s1)
    sub = d[m]
    a, b_, c_ = sh(d["base"][m], sub["r"]), sh(d["A0"][m], sub["r"]), sh(d["PRAC"][m], sub["r"])
    print(f"  {lbl:<26}{a:>9.3f}{b_:>10.3f}{b_-a:>+8.3f}{c_:>10.3f}{c_-a:>+8.3f}"
          f"{('+' if c_-a>0 else '-'):>7}")

print()
print("=" * 110)
print("二、安慰剂检验（情绪分位循环平移，最终版）")
print("=" * 110)
v = IFp.values
real = sh(d["PRAC"], d["r"])
plc = []
for step in range(20, len(v) - 20, 40):
    d2 = make(pd.Series(np.roll(v, step), index=IFp.index), PV_M, SL_M)
    plc.append(sh(d2["PRAC"], d2["r"]))
plc = np.array([x for x in plc if x == x])
print(f"  安慰剂样本 {len(plc)} 个：均值 {plc.mean():.3f}，std {plc.std():.3f}，"
      f"95% 分位 {np.percentile(plc,95):.3f}，最大 {plc.max():.3f}")
print(f"  实际 {real:.3f}　→　分位 {float((plc<real).mean())*100:.0f}%（超过 {int((plc<real).sum())}/{len(plc)} 个安慰剂样本）")

print()
print("=" * 110)
print("三、36 组 seed（pv seed × slow seed）")
print("=" * 110)
res = []
for i in range(6):
    pv_i = load_seed("s", i)
    for j in range(6):
        sl_j = load_seed("slow_s", j)
        dj = make(IFp, pv_i, sl_j)
        res.append((sh(dj["base"], dj["r"]), sh(dj["PRAC"], dj["r"])))
R = np.array(res)
dd = R[:, 1] - R[:, 0]
print(f"  样本 {len(R)} 组")
print(f"  {'版本':<20}{'均值':>9}{'中位':>9}{'std':>9}{'最小':>9}{'最大':>9}")
print(f"  {'基准 M3':<20}{R[:,0].mean():>9.3f}{np.median(R[:,0]):>9.3f}{R[:,0].std():>9.3f}"
      f"{R[:,0].min():>9.3f}{R[:,0].max():>9.3f}")
print(f"  {'最终版':<20}{R[:,1].mean():>9.3f}{np.median(R[:,1]):>9.3f}{R[:,1].std():>9.3f}"
      f"{R[:,1].min():>9.3f}{R[:,1].max():>9.3f}")
print(f"  Δ夏普：均值 {dd.mean():+.3f}，std {dd.std():.3f}，最小 {dd.min():+.3f}，"
      f"最大 {dd.max():+.3f}，正占比 {(dd>0).mean():.0%}")
print("\n完成")
