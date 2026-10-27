# -*- coding: utf-8 -*-
"""情绪类因子全面 IC 体检：不止看 IC，还要看方向/单调性/滞后性/独立性/多重检验"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob, numpy as np, pandas as pd, warnings
from scipy import stats as st
warnings.filterwarnings("ignore")
BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")

px_df = None
for enc in ("gbk", "utf-8-sig"):
    try:
        px_df = pd.read_csv(os.path.join(INPUT, "中证1000_000852.SH.csv"), encoding=enc)
        break
    except Exception:
        continue
dc = [c for c in px_df.columns if "日期" in c][0]
cc = [c for c in px_df.columns if "收盘" in c][0]
px = pd.Series(pd.to_numeric(px_df[cc], errors="coerce").values,
               index=pd.to_datetime(px_df[dc])).sort_index()
px = px[~px.index.duplicated(keep="last")]
lpx = np.log(px)

panel = pd.read_csv(os.path.join(INPUT, "情绪_面板_日频.csv"), encoding="utf-8-sig", parse_dates=["日期"])
panel = panel.drop_duplicates("日期").set_index("日期").sort_index()
cols = [c for c in panel.columns if c != "日期"]
FWD = 10
fwd = (lpx.shift(-FWD) - lpx).reindex(panel.index)
trail = (lpx - lpx.shift(FWD)).reindex(panel.index)
ret = (lpx - lpx.shift(1)).reindex(panel.index)

print("=" * 132)
print("【表 A】全部情绪字段的横向体检（区间：字段自身可用起点 ~ 2026-09-30）")
print("=" * 132)
print(f"  {'字段':<18}{'起点':<11}{'n':>5}{'IC(fwd10)':>10}{'IC(2019+)':>10}{'IC(trail10)':>12}"
      f"{'IC(|fwd10|)':>12}{'低位后10日':>11}{'高位后10日':>11}{'t(低-高)':>9}{'单调性':>8}")
rows = []
for c in cols:
    s = pd.to_numeric(panel[c], errors="coerce").dropna()
    if len(s) < 200:
        continue
    d = pd.DataFrame({"x": s, "f": fwd.reindex(s.index), "t": trail.reindex(s.index), "r": ret.reindex(s.index)}).dropna()
    if len(d) < 200:
        continue
    ic = st.spearmanr(d["x"], d["f"]).statistic
    d2 = d[d.index >= "2019-06-04"]
    ic2 = st.spearmanr(d2["x"], d2["f"]).statistic if len(d2) > 100 else np.nan
    ict = st.spearmanr(d["x"], d["t"]).statistic
    ica = st.spearmanr(d["x"], d["f"].abs()).statistic
    q = d["x"].rank(pct=True)
    lo = d.loc[q <= 0.10, "f"]; hi = d.loc[q >= 0.90, "f"]
    tval = (lo.mean() - hi.mean()) / np.sqrt(lo.var() / len(lo) + hi.var() / len(hi))
    # 单调性：10 分箱均值序列与箱序号的 Spearman
    d = d.assign(b=pd.qcut(d["x"].rank(method="first"), 10, labels=False))
    bm = d.groupby("b")["f"].mean()
    mono = st.spearmanr(np.arange(len(bm)), bm.values).statistic
    rows.append(dict(field=c, start=s.index[0].date(), n=len(d), ic=ic, ic2=ic2, ict=ict, ica=ica,
                     lo=np.expm1(lo.mean()), hi=np.expm1(hi.mean()), t=tval, mono=mono))
    print(f"  {c:<18}{str(s.index[0].date()):<11}{len(d):>5}{ic:>+10.3f}{ic2:>+10.3f}{ict:>+12.3f}"
          f"{ica:>+12.3f}{np.expm1(lo.mean())*100:>+10.2f}%{np.expm1(hi.mean())*100:>+10.2f}%"
          f"{tval:>+9.2f}{mono:>+8.2f}")

R = pd.DataFrame(rows).set_index("field")
print("\n  说明：IC = 与未来10日收益的 Spearman；IC(trail10) = 与「过去10日收益」的相关（衡量是否只是滞后/同期变量）；")
print("        IC(|fwd10|) = 与未来10日绝对收益的相关（衡量是否只是波动率代理）；单调性 = 10分箱均值随分位上升的秩相关（±1 为严格单调）。")

print("\n" + "=" * 132)
print("【表 B】10 分箱明细：未来 10 日平均收益（%）—— 找 U 型 / 非单调")
print("=" * 132)
focus = [c for c in ["50ETF_PCR成交", "50ETF_PCR持仓", "50ETF_IV平值", "50ETF_IV沽购比", "50ETF_IV偏斜",
                     "IF_会员多空比", "IC_会员多空比", "IM_会员多空比", "IH_会员多空比"] if c in R.index]
print(f"  {'字段':<18}" + "".join([f"{'D'+str(i+1):>8}" for i in range(10)]))
for c in focus:
    s = pd.to_numeric(panel[c], errors="coerce").dropna()
    d = pd.DataFrame({"x": s, "f": fwd.reindex(s.index)}).dropna()
    d = d.assign(b=pd.qcut(d["x"].rank(method="first"), 10, labels=False))
    bm = d.groupby("b")["f"].mean() * 100
    print(f"  {c:<18}" + "".join([f"{v:>+8.2f}" for v in bm.values]))

print("\n" + "=" * 132)
print("【表 C】独立性：28 个字段其实只有几个独立方向（标准化后 PCA）")
print("=" * 132)
X = panel[[c for c in cols]].apply(pd.to_numeric, errors="coerce")
X = X[X.notna().sum(axis=1) >= 6]
X = X.fillna(X.mean())
Z = (X - X.mean()) / X.std()
Z = Z.dropna(axis=1)
C = np.corrcoef(Z.values.T)
ev = np.sort(np.linalg.eigvalsh(C))[::-1]
ev = np.clip(ev, 0, None)
pr = (ev.sum() ** 2) / (ev ** 2).sum()      # 参与比 = 有效独立方向数
cum = np.cumsum(ev) / ev.sum()
print(f"  字段数 {len(ev)}，有效独立方向数（参与比）≈ {pr:.1f}")
print(f"  前 3 个主成分累积解释率 {cum[2]:.1%}，前 5 个 {cum[4]:.1%}")
print("\n  高相关簇（|ρ|>0.7）：")
seen = set()
for i, a in enumerate(Z.columns):
    grp = [b for j, b in enumerate(Z.columns) if j != i and abs(C[i, j]) > 0.7]
    if len(grp) >= 2 and a not in seen:
        seen.update([a] + grp)
        print(f"    {a}: 与 {', '.join(grp)} 高度相关（同源信息）")

print("\n" + "=" * 132)
print("【表 D】多重检验：28 个字段里挑最好的，需要多大的 IC 才算真信号")
print("=" * 132)
n_avg = int(R["n"].median())
print(f"  样本量中位数 {n_avg}；|IC| 的 t 值 ≈ IC*sqrt(n-2)/sqrt(1-IC^2)")
for k in [1, 5, 10, 22]:
    # Bonferroni 临界 z (双侧 5%)
    zc = st.norm.ppf(1 - 0.05 / (2 * k))
    ic_crit = zc / np.sqrt(n_avg)
    print(f"    检验 {k:>2} 个字段 → 临界 |IC| = {ic_crit:.3f}（t≈{zc:.2f}）")
print("\n  各字段 |IC| 排名（前 8）：")
for c, v in R["ic"].abs().sort_values(ascending=False).head(8).items():
    print(f"    {c:<18} IC={R.loc[c,'ic']:+.3f}  t≈{R.loc[c,'ic']*np.sqrt(R.loc[c,'n']-2):.2f}"
          f"  (2019+ IC={R.loc[c,'ic2']:+.3f}, 单调={R.loc[c,'mono']:+.2f})")
