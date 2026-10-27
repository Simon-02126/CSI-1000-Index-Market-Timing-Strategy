# -*- coding: utf-8 -*-
"""机制归因：为什么这两个因子是"风险管理型"而不是"收益增强型"

判别检验（对每个因子 + 合成信号）：
  A. 收益预测 vs 波动预测：corr(sig, 未来收益) 与 corr(sig, 未来绝对收益/波动) 谁更强
  B. 上行/下行分解：把未来收益拆成 max(r,0) 与 max(-r,0)，看信号对哪一侧解释力更强
  C. 分层"坏状态识别" vs "好状态识别"：Q1 与其余四组的差距 vs Q5 与其余四组的差距
  D. 仓位与波动的关系：策略仓位 与 同期已实现波动的相关性
  E. 大涨月/大跌月的平均仓位对比
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
ANN = 252


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
r1 = np.log(close).diff()


def load_tag(tag):
    parts = []
    for p in sorted(glob.glob(os.path.join(TRAIN, f"{tag}_*.csv"))):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def roll_pct(s, win=300, minp=250):
    v = s.values.astype(float); o = np.full(len(v), np.nan)
    mp = min(minp, max(60, win // 3))
    for i in range(len(v)):
        if i < mp: continue
        lo = max(0, i - win)
        o[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(o, index=s.index)


pv_avg = pd.concat([load_tag(f"s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()
sl_avg = pd.concat([load_tag(f"slow_s{sd}") for sd in range(6)], axis=1).mean(axis=1).dropna()
pv_pct = roll_pct(pv_avg)
sl_pct = roll_pct(sl_avg)
sl_adj = ((sl_pct - 0.30) / 0.35).clip(0, 1)
comp = (0.5 * pv_pct + 0.5 * sl_adj).dropna()

H = 10
fwd = np.log(close.shift(-H) / close)


def pe(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 30: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


SIG = {"量价因子 pv": pv_pct, "资金因子 slow(饱和)": sl_adj, "合成信号": comp}

print("=" * 118)
print("检验 A/D：预测的是「收益」还是「波动」？（2019+，H=10 日）")
print("=" * 118)
print(f"  {'信号':<18}{'corr(sig,未来收益)':>18}{'corr(sig,|未来收益|)':>20}"
      f"{'corr(sig,未来20日波动)':>22}{'判定':>12}")
for nm, s in SIG.items():
    d = pd.DataFrame({"s": s, "f": fwd, "af": fwd.abs(),
                      "vol": r1.rolling(20).std().shift(-20) * np.sqrt(ANN)}).dropna()
    d = d[d.index >= "2019-01-01"]
    c_r = pe(d["s"].values, d["f"].values)
    c_abs = pe(d["s"].values, d["af"].values)
    c_v = pe(d["s"].values, d["vol"].values)
    verdict = "波动信号" if abs(c_v) > abs(c_r) else "收益信号"
    print(f"  {nm:<18}{c_r:>18.4f}{c_abs:>20.4f}{c_v:>22.4f}{verdict:>12}")

print("\n" + "=" * 118)
print("检验 B：上行侧 vs 下行侧解释力（把未来收益拆成 max(r,0) 与 max(-r,0)）")
print("=" * 118)
print(f"  {'信号':<18}{'corr(sig,涨幅)':>15}{'corr(sig,跌幅)':>15}"
      f"{'下行/上行 比':>13}{'判定':>14}")
for nm, s in SIG.items():
    d = pd.DataFrame({"s": s, "f": fwd}).dropna()
    d = d[d.index >= "2019-01-01"]
    up = np.maximum(d["f"].values, 0)
    dn = np.maximum(-d["f"].values, 0)
    c_up = pe(d["s"].values, up)
    c_dn = pe(d["s"].values, dn)
    ratio = c_dn / c_up if c_up else np.nan
    verdict = "以识别下行风险为主" if ratio < -0.5 or abs(c_dn) > abs(c_up) * 1.3 else "上下行相对均衡"
    print(f"  {nm:<18}{c_up:>15.4f}{c_dn:>15.4f}{ratio:>13.2f}{verdict:>14}")
print("  （corr 为负 = 信号越高、该侧幅度越小；下行侧绝对值大 → 主要在预测「跌得深不深」）")

print("\n" + "=" * 118)
print("检验 C：识别「坏状态」vs 识别「好状态」（5 分箱前瞻 10 日收益，单位 %）")
print("=" * 118)
print(f"  {'信号':<18}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5':>8}"
      f"{'Q1 vs其余(坏状态)':>18}{'Q5 vs其余(好状态)':>18}")
for nm, s in SIG.items():
    d = pd.DataFrame({"s": s, "f": fwd}).dropna()
    d = d[d.index >= "2019-01-01"].iloc[::H]
    q = pd.qcut(d["s"], 5, labels=False, duplicates="drop")
    m = [d["f"][q == k].mean() * 100 for k in range(5)]
    bad = m[0] - np.mean(m[1:])
    good = m[4] - np.mean(m[:4])
    print(f"  {nm:<18}" + "".join(f"{v:>8.2f}" for v in m)
          + f"{bad:>18.2f}{good:>18.2f}")
print("  （坏状态 = Q1 相对其余四组低多少；好状态 = Q5 相对其余四组高多少）")

print("\n" + "=" * 118)
print("检验 D：策略仓位与波动的关系（是否在做「波动率调节」）")
print("=" * 118)
print(f"  {'仓位序列':<18}{'corr(仓位,当期20日波动)':>24}{'corr(仓位,未来20日收益)':>24}"
      f"{'高波动期均仓':>13}{'低波动期均仓':>13}")
pos_sets = {"M3 策略仓位": None}
# 重建 M3 仓位
from numpy import where
m = pd.DataFrame({"p": pv_pct, "s": sl_pct}).dropna()
raw = 0.5 * m["p"] + 0.5 * ((m["s"] - 0.30) / 0.35).clip(0, 1)
t3 = pd.Series(np.where(raw < 0.30, 0.0, np.where(raw < 0.80, 0.5, 1.0)), index=raw.index)
v = t3.values; o = np.full(len(v), np.nan)
for i in range(len(v)):
    w = v[max(0, i - 4):i + 1]; w = w[~np.isnan(w)]
    if len(w):
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
pos = pd.Series(o, index=t3.index)
rv = r1.rolling(20).std() * np.sqrt(ANN)
fwd20 = np.log(close.shift(-20) / close)
d = pd.DataFrame({"pos": pos, "rv": rv, "f20": fwd20}).dropna()
d = d[d.index >= "2019-06-01"]
med = d["rv"].median()
print(f"  {'M3 策略仓位':<18}{pe(d['pos'], d['rv']):>24.3f}{pe(d['pos'], d['f20']):>24.3f}"
      f"{d.loc[d['rv']>med,'pos'].mean():>13.0%}{d.loc[d['rv']<=med,'pos'].mean():>13.0%}")

print("\n" + "=" * 118)
print("检验 E：大涨月 / 大跌月的平均仓位（仓位是否只在跌市下降）")
print("=" * 118)
d2 = pd.DataFrame({"pos": pos, "c": close}).dropna()
d2 = d2[d2.index >= "2019-06-01"]
mo = d2.groupby(d2.index.to_period("M")).agg(pos=("pos", "mean"),
                                            ret=("c", lambda x: np.log(x.iloc[-1] / x.iloc[0])))
for lbl, mask in [("大涨月 (指数>+5%)", mo["ret"] > 0.05), ("小涨月 (0~+5%)", (mo["ret"] > 0) & (mo["ret"] <= 0.05)),
                  ("小跌月 (-5%~0)", (mo["ret"] <= 0) & (mo["ret"] > -0.05)), ("大跌月 (指数<-5%)", mo["ret"] <= -0.05)]:
    g = mo[mask]
    if len(g) == 0: continue
    print(f"  {lbl:<20} 月数 {len(g):>3}  平均仓位 {g['pos'].mean():>6.0%}  指数平均 {g['ret'].mean():>7.2%}")
