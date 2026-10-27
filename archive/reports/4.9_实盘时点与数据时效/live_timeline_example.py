# -*- coding: utf-8 -*-
"""实盘时间线示例：逐日拆出每一块数据到达后信号怎么变"""
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
W, W_SENT, K = 300, 250, 5
GRID, DEAD = 0.05, 0.10
LO, HI = "2024-09-20", "2024-10-11"


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
len_m = panel["全市场融资余额(元)"] if "全市场融资余额(元)" in panel.columns else None


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
mm = pd.DataFrame({"pv_pct": roll_pct(pv_avg, W), "sl_pct": roll_pct(sl_avg, W)}).dropna()
mm["sl_sat"] = ((mm["sl_pct"] - 0.30) / 0.35).clip(0, 1)
mm["comp"] = 0.5 * mm["pv_pct"] + 0.5 * mm["sl_sat"]
mm["base"] = pos_mode(tgt3(mm["comp"], 0.30, 0.80), K)
mm["IF_pct"] = roll_pct(raw_if, W_SENT, minp=60)
mm["IF_raw"] = raw_if

d = mm.join(pd.DataFrame({"c": close, "o": open_}), how="inner").dropna(subset=["c", "o"])
full = d.copy()
d = d.loc[LO:HI].copy()
half = (d["base"] - 0.5).abs() < 1e-9
d["A0"] = d["base"].where(~half, d["IF_pct"])
# 死区要按全序列算，这里用全序列结果切片
full_half = (full["base"] - 0.5).abs() < 1e-9
full["A0"] = full["base"].where(~full_half, full["IF_pct"])
full["P"] = band(grid(full["A0"], GRID), DEAD)
full["g"] = np.log(full["o"] / full["c"].shift(1))

print("逐日明细（2024-09-20 ~ 2024-10-11）")
print("=" * 150)
print(f"{'日期':<12}{'收盘':>9}{'涨跌%':>8}{'pv分位':>8}{'slow分位':>9}{'合成':>7}{'基础仓位':>9}"
      f"{'IF原值':>8}{'IF分位':>8}{'目标(A0)':>9}{'网格+死区后':>12}{'当日动作':>10}")
prev = None
for dt, r in full.loc[LO:HI].iterrows():
    chg = (r["c"] / full["c"].shift(1).loc[dt] - 1) * 100 if dt != full.index[0] else np.nan
    act = "—"
    if prev is not None:
        dd = r["P"] - prev
        if abs(dd) > 1e-9:
            act = f"{'加' if dd>0 else '减'}{abs(dd)*100:.0f}%"
    print(f"{dt.strftime('%Y-%m-%d'):<12}{r['c']:>9.1f}{chg:>8.2f}{r['pv_pct']:>8.3f}{r['sl_pct']:>9.3f}"
          f"{r['comp']:>7.3f}{r['base']:>9.2f}{r['IF_raw']:>8.3f}{r['IF_pct']:>8.3f}"
          f"{r['A0']:>9.3f}{r['P']:>12.3f}{act:>10}")
    prev = r["P"]

print("=" * 150)
print("\n对照：同一时间窗口，基础仓位(M3) vs 最终仓位")
sub = full.loc[LO:HI]
print(f"  日均基础仓位   {sub['base'].mean()*100:.1f}%")
print(f"  日均最终仓位   {sub['P'].mean()*100:.1f}%")
print(f"  窗口内指数      {(np.exp(np.log(sub['c']).iloc[-1]-np.log(sub['c']).iloc[0])-1)*100:+.1f}%")
