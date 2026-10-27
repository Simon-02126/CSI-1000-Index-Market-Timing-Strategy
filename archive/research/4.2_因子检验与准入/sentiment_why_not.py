# -*- coding: utf-8 -*-
"""为什么大多数情绪因子不能用：时间稳定性 / 与已有信号共线 / 极值段翻转 / 多重检验"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, numpy as np, pandas as pd, warnings, glob
from scipy import stats as st
warnings.filterwarnings("ignore")
BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
TRAIN = _P.TRAIN
for enc in ("gbk", "utf-8-sig"):
    try:
        pdf = pd.read_csv(os.path.join(INPUT, "中证1000_000852.SH.csv"), encoding=enc); break
    except Exception:
        continue
dc = [c for c in pdf.columns if "日期" in c][0]
cc = [c for c in pdf.columns if "收盘" in c][0]
px = pd.Series(pd.to_numeric(pdf[cc], errors="coerce").values, index=pd.to_datetime(pdf[dc])).sort_index()
px = px[~px.index.duplicated(keep="last")]
lpx = np.log(px)

panel = pd.read_csv(os.path.join(INPUT, "情绪_面板_日频.csv"), encoding="utf-8-sig", parse_dates=["日期"])
panel = panel.drop_duplicates("日期").set_index("日期").sort_index()
FWD = 10
fwd = (lpx.shift(-FWD) - lpx).reindex(panel.index)

CAND = ["50ETF_PCR成交", "50ETF_PCR持仓", "50ETF_IV平值", "50ETF_IV虚沽", "50ETF_IV沽购比", "50ETF_IV偏斜",
        "300ETF_IV平值", "300ETF_IV偏斜", "500ETF_IV平值", "500ETF_PCR成交",
        "IF_会员多空比", "IC_会员多空比", "IH_会员多空比", "IM_会员多空比"]
CAND = [c for c in CAND if c in panel.columns]

print("=" * 120)
print("【1】IC 的时间稳定性：分 4 段各自算 Spearman IC（方向翻转 = 阈值规则必然失效）")
print("=" * 120)
ERAS = [("2015-2018", "2015-01-01", "2018-12-31"), ("2019-2021", "2019-01-01", "2021-12-31"),
        ("2022-2023", "2022-01-01", "2023-12-31"), ("2024-2026", "2024-01-01", "2026-12-31")]
print(f"  {'字段':<18}" + "".join([f"{e[0]:>12}" for e in ERAS]) + f"{'同号段数':>10}{'极差':>8}")
for c in CAND:
    s = pd.to_numeric(panel[c], errors="coerce")
    d = pd.DataFrame({"x": s, "f": fwd}).dropna()
    ics, vals = [], []
    for nm, a, b in ERAS:
        g = d[(d.index >= a) & (d.index <= b)]
        v = st.spearmanr(g["x"], g["f"]).statistic if len(g) > 100 else np.nan
        vals.append(v)
    vals_ = [v for v in vals if not np.isnan(v)]
    same = max(sum(v > 0 for v in vals_), sum(v < 0 for v in vals_)) if vals_ else 0
    rng = (max(vals_) - min(vals_)) if vals_ else np.nan
    print(f"  {c:<18}" + "".join([(f"{v:>+12.3f}" if not np.isnan(v) else f"{'—':>12}") for v in vals])
          + f"{same:>10}{rng:>8.2f}")

print("\n" + "=" * 120)
print("【2】否决式用法（分位<0.20 后未来10日收益）分年代检验：这是 IF 被选中的真正理由")
print("=" * 120)
print(f"  {'字段':<18}{'全样本':>10}{'2015-18':>10}{'2019-21':>10}{'2022-23':>10}{'2024-26':>10}{'负号段数':>10}")
for c in CAND:
    s = pd.to_numeric(panel[c], errors="coerce").dropna()
    q = s.rank(pct=True)
    vals = []
    for nm, a, b in [("all", "2000-01-01", "2030-01-01")] + ERAS:
        g = q[(q.index >= a) & (q.index <= b) & (q <= 0.20)]
        f = fwd.reindex(g.index).dropna()
        vals.append(np.expm1(f.mean()) * 100 if len(f) > 30 else np.nan)
    nneg = sum(1 for v in vals[1:] if not np.isnan(v) and v < 0)
    print(f"  {c:<18}" + "".join([(f"{v:>+9.2f}%" if not np.isnan(v) else f"{'—':>10}") for v in vals])
          + f"{nneg:>10}/4")

print("\n" + "=" * 120)
print("【3】与已有信号（pv / slow 6-seed 均值）的共线：高 IC 是否已被 pv 含量吃掉了")
print("=" * 120)


def load_tag(tag):
    parts = []
    for p in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


pv = pd.concat([load_tag(f"s{d}") for d in range(6)], axis=1).mean(axis=1).dropna()
sl = pd.concat([load_tag(f"slow_s{d}") for d in range(6)], axis=1).mean(axis=1).dropna()
print(f"  {'字段':<18}{'corr(pv)':>10}{'corr(slow)':>12}{'corr(IF)':>10}{'正交pv后IC':>12}{'正交IF后IC':>12}")
IFs = pd.to_numeric(panel["IF_会员多空比"], errors="coerce")
for c in CAND:
    s = pd.to_numeric(panel[c], errors="coerce")
    d = pd.DataFrame({"x": s, "pv": pv, "sl": sl, "IF": IFs, "f": fwd}).dropna()
    if len(d) < 300:
        print(f"  {c:<18}{'样本不足':>10}"); continue
    r_pv = d["x"].corr(d["pv"]); r_sl = d["x"].corr(d["sl"]); r_if = d["x"].corr(d["IF"])
    # 对 pv 正交
    b = np.polyfit(d["pv"], d["x"], 1); resid = d["x"] - np.polyval(b, d["pv"])
    ic_o = st.spearmanr(resid, d["f"]).statistic
    b2 = np.polyfit(d["IF"], d["x"], 1); resid2 = d["x"] - np.polyval(b2, d["IF"])
    ic_if = st.spearmanr(resid2, d["f"]).statistic
    print(f"  {c:<18}{r_pv:>+10.3f}{r_sl:>+12.3f}{r_if:>+10.3f}{ic_o:>+12.3f}{ic_if:>+12.3f}")

print("\n" + "=" * 120)
print("【4】IF 与其近亲的相关（证明「多品种」不是独立信息）")
print("=" * 120)
sub = panel[["IF_会员多空比", "IC_会员多空比", "IH_会员多空比", "IM_会员多空比"]].apply(pd.to_numeric, errors="coerce")
print(sub.corr().round(3).to_string())
print("\n  成交量/IV 跨标的同族相关（同一指标不同标的）:")
pairs = [("50ETF_IV平值", "300ETF_IV平值"), ("300ETF_IV平值", "500ETF_IV平值"), ("50ETF_IV平值", "500ETF_IV平值"),
         ("50ETF_PCR成交", "300ETF_PCR成交"), ("300ETF_PCR成交", "500ETF_PCR成交")]
for a, b in pairs:
    if a in panel.columns and b in panel.columns:
        d = panel[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        print(f"    corr({a}, {b}) = {d[a].corr(d[b]):+.3f}  (n={len(d)})")
