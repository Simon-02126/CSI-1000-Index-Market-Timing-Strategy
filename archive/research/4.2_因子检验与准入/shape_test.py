# -*- coding: utf-8 -*-
"""因子A 非单调形状的稳健性检验 + 三种处理方式的实证

Part 1 形状是否真实: 换采样起点 / 分时段 / 换持有期 / 换分箱数
Part 2 波动率归一后形状是否改善
Part 3 用"条件期望映射"(learn the shape) 替代单调映射, 与三档对比
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
INPUT, SIGDIR = os.path.join(BASE, "input"), os.path.join(BASE, "output", "signals")
ANN, START = 252, "2019-01-01"


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


def load_price():
    fp = glob.glob(os.path.join(INPUT, "*中证1000*.csv"))[0]
    df = read_any(fp)
    d = [c for c in df.columns if "日期" in c][0]
    c = [c for c in df.columns if "收盘" in c][0]
    s = pd.Series(pd.to_numeric(df[c], errors="coerce").values, index=pd.to_datetime(df[d]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_mat(view):
    fp = glob.glob(os.path.join(SIGDIR, f"*中证1000*_{view}_signals.csv"))[0]
    df = pd.read_csv(fp, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    df = df.drop_duplicates("日期").set_index("日期").sort_index()
    return df[cols].mean(axis=1)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


def monotonicity(vals):
    """层均值与层序号的秩相关, 1=完全单调递增"""
    return pearson(np.arange(len(vals)), vals)


close = load_price()
pv = load_mat("pv")
sl = load_mat("slow")
ret = np.log(close).diff()

print("=" * 104)
print("Part 1  形状是否真实？")
print("=" * 104)

# ---- 1a. 换采样起点 offset (0..H-1) ----
H = 10
fwd = np.log(close.shift(-H) / close)
df = pd.DataFrame({"s": pv, "f": fwd}).dropna()
df = df[df.index >= START]
print(f"\n[1a] 换采样起点（H={H}, 5 分箱, 每行一个起点; 单位 %）")
print("  offset  " + "".join(f"{'Q'+str(i+1):>9}" for i in range(5)) + f"{'单调性':>8}{'样本':>7}")
allmeans = []
for off in range(H):
    sub = df.iloc[off::H]
    q = pd.qcut(sub["s"], 5, labels=False, duplicates="drop")
    m = [sub["f"][q == k].mean() * 100 for k in range(5)]
    allmeans.append(m)
    print(f"  {off:>6}  " + "".join(f"{v:>9.2f}" for v in m) + f"{monotonicity(m):>8.2f}{len(sub):>7}")
A = np.array(allmeans)
print(f"  均值    " + "".join(f"{v:>9.2f}" for v in A.mean(0)) + f"{monotonicity(A.mean(0)):>8.2f}")
print(f"  标准差  " + "".join(f"{v:>9.2f}" for v in A.std(0, ddof=1)))
print(f"  → Q5-Q4 差值在 {H} 个起点中为负的比例: {np.mean(A[:,4] < A[:,3]):.0%}")

# ---- 1b. 分时段 ----
print(f"\n[1b] 分时段（offset=0, H={H}）")
for a, b, lbl in [("2019-01-01", "2022-12-31", "2019-2022"), ("2023-01-01", "2099-12-31", "2023-2026")]:
    sub = df[(df.index >= a) & (df.index <= b)].iloc[::H]
    q = pd.qcut(sub["s"], 5, labels=False, duplicates="drop")
    m = [sub["f"][q == k].mean() * 100 for k in range(min(5, q.nunique()))]
    print(f"  {lbl}  " + "".join(f"{v:>8.2f}" for v in m) + f"   单调性{monotonicity(m):>6.2f}  n={len(sub)}")

# ---- 1c. 换持有期 ----
print(f"\n[1c] 换持有期（offset=0, 5 分箱）")
for h in (5, 10, 20):
    f2 = np.log(close.shift(-h) / close)
    d2 = pd.DataFrame({"s": pv, "f": f2}).dropna()
    d2 = d2[d2.index >= START].iloc[::h]
    q = pd.qcut(d2["s"], 5, labels=False, duplicates="drop")
    m = [d2["f"][q == k].mean() * 100 for k in range(5)]
    print(f"  H={h:<3} " + "".join(f"{v:>8.2f}" for v in m) + f"   单调性{monotonicity(m):>6.2f}  n={len(d2)}")

# ---- 1d. 换分箱数 ----
print(f"\n[1d] 换分箱数（H={H}, offset=0）")
for nb in (3, 4, 5, 8):
    sub = df.iloc[::H]
    q = pd.qcut(sub["s"], nb, labels=False, duplicates="drop")
    m = [sub["f"][q == k].mean() * 100 for k in range(nb)]
    print(f"  {nb} 箱  " + "".join(f"{v:>8.2f}" for v in m) + f"   单调性{monotonicity(m):>6.2f}")

# ---- 1e. 各层均值的标准误 ----
print(f"\n[1e] 各层均值是否可分（H={H}, offset=0, 用各层标准误判断）")
sub = df.iloc[::H]
q = pd.qcut(sub["s"], 5, labels=False, duplicates="drop")
for k in range(5):
    g = sub["f"][q == k]
    se = g.std(ddof=1) / np.sqrt(len(g))
    print(f"  Q{k+1}: 均值 {g.mean()*100:>+6.2f}%  ±SE {se*100:>5.2f}%  n={len(g)}  "
          f"95%区间 [{(g.mean()-1.96*se)*100:>+6.2f}, {(g.mean()+1.96*se)*100:>+6.2f}]")

print("\n" + "=" * 104)
print("Part 2  波动率归一后形状是否改善")
print("=" * 104)
for k in (20, 60):
    z = pv / pv.rolling(k, min_periods=k // 2).std()
    d3 = pd.DataFrame({"s": z, "f": fwd}).dropna()
    d3 = d3[d3.index >= START].iloc[::H]
    q = pd.qcut(d3["s"], 5, labels=False, duplicates="drop")
    m = [d3["f"][q == k2].mean() * 100 for k2 in range(5)]
    print(f"  信号/滚动{k}日std  " + "".join(f"{v:>8.2f}" for v in m) + f"   单调性{monotonicity(m):>6.2f}")
# 对照: 用 20日涨幅做"过热"过滤
r20 = close.pct_change(20)
d4 = pd.DataFrame({"s": pv, "f": fwd, "r20": r20}).dropna()
d4 = d4[d4.index >= START].iloc[::H]
q = pd.qcut(d4["s"], 5, labels=False, duplicates="drop")
hi = d4["r20"] > d4["r20"].median()
print("  按前20日涨幅切分（同分箱内）:")
for lbl, mask in [("前20日涨幅 高", q.notna() & hi), ("前20日涨幅 低", q.notna() & ~hi)]:
    m2 = [d4["f"][(q == k2) & mask].mean() * 100 for k2 in range(5)]
    n2 = [int(((q == k2) & mask).sum()) for k2 in range(5)]
    print(f"    {lbl}  " + "".join(f"{v:>8.2f}" for v in m2) + f"   n={n2}")

print("\n" + "=" * 104)
print("Part 3  用「条件期望映射」(learn the shape) 替代单调映射")
print("=" * 104)
# 构造滚动分位
def roll_pct(s, win=750, minp=250):
    v = s.values.astype(float); out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp: continue
        lo = max(0, i - win)
        out[i] = (v[lo:i] < v[i]).sum() / (i - lo)
    return pd.Series(out, index=s.index)

pct_pv = roll_pct(pv)
pct_sl = roll_pct(sl)
pct = 0.5 * pct_pv + 0.5 * pct_sl

# 条件期望映射: 每 t 日只用 s+H <= t 的历史, 按 5 箱估计各层平均前瞻收益
fwdH = np.log(close.shift(-H) / close)
hist = pd.DataFrame({"p": pct, "f": fwdH}).dropna()
hist = hist[hist.index >= "2017-01-01"]

NB, MINOBS = 5, 120
pos_ce = pd.Series(np.nan, index=pct.index)
dates = list(pct.index)
for t in dates:
    hh = hist[hist.index <= t - pd.Timedelta(days=H * 2)]   # 保守: 留出 H*2 天缓冲
    if len(hh) < MINOBS:
        continue
    pt = pct.loc[t]
    if not np.isfinite(pt):
        continue
    # 用历史分位切箱(等频), 得到 bin 边界
    edges = np.quantile(hh["p"], np.linspace(0, 1, NB + 1))
    edges[-1] += 1e-9
    hh_bin = np.clip(np.searchsorted(edges, hh["p"].values, side="right") - 1, 0, NB - 1)
    means = np.array([hh["f"].values[hh_bin == b].mean() if (hh_bin == b).sum() >= 20 else np.nan
                      for b in range(NB)])
    if np.isnan(means).any():
        continue
    b_t = int(np.clip(np.searchsorted(edges, pt, side="right") - 1, 0, NB - 1))
    e = means[b_t]
    lo, hi = np.quantile(means, 0.05), np.quantile(means, 0.95)
    pos_ce.loc[t] = float(np.clip((e - lo) / (hi - lo + 1e-12), 0, 1))


def bt(pos, start=START, cost=0.0003, smooth=1):
    p = pos.rolling(smooth, min_periods=1).mean() if smooth > 1 else pos
    d = pd.DataFrame({"p": p, "c": close}).dropna()
    d = d[d.index >= start]
    if len(d) < 150: return None
    r = np.log(d["c"]).diff()
    turn = d["p"].diff().abs().fillna(0)
    st = (d["p"].shift(1) * r - turn * cost).dropna()
    eq = st.cumsum(); mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(st.mean() * ANN); vol = float(st.std() * np.sqrt(ANN))
    return dict(sharpe=ann / vol if vol else np.nan, ann=ann, vol=vol, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan, avg=float(d["p"].mean()),
                chg=float((turn > 1e-9).sum()) / (len(d) / ANN), byyear=st.groupby(st.index.year).sum())


def tgt_band3(p, lo=0.30, hi=0.80):
    return pd.Series(np.where(p < lo, 0.0, np.where(p < hi, 0.5, 1.0)), index=p.index).where(p.notna())


def pos_mode(t, k=5):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if len(w) == 0: continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def pos_linear(p, lo=0.30, hi=0.80):
    return ((p - lo) / (hi - lo)).clip(0, 1)


t3 = tgt_band3(pct)
cands = [("三档+众数5（现方案）", pos_mode(t3, 5)),
         ("v3连续线性", pos_linear(pct)),
         ("条件期望映射（连续）", pos_ce),
         ("条件期望映射（三档：按预期收益分位切）", None)]
ce = pos_ce.dropna()
q3 = pd.qcut(ce, 3, labels=False, duplicates="drop")
band_ce = pd.Series(np.where(q3 == 0, 0.0, np.where(q3 == 1, 0.5, 1.0)), index=ce.index).reindex(pct.index)
cands[-1] = ("条件期望映射（三档）", band_ce)

for start, lbl in [(START, "2019+"), ("2023-01-01", "2023+ 留出")]:
    print(f"\n--- {lbl} ---")
    for nm, p in cands:
        s = bt(p, start, smooth=5 if "条件" in nm or "线性" in nm else 1)
        if s is None:
            print(f"  {nm:<28} 样本不足"); continue
        print(f"  {nm:<28} 夏普{s['sharpe']:>5.2f} 年化{s['ann']:>7.1%} 波动{s['vol']:>6.1%} "
              f"回撤{s['mdd']:>7.1%} Calmar{s['calmar']:>5.2f} 均仓{s['avg']:>4.0%} 换仓{s['chg']:>4.1f}/年")

# 学习到的映射形状（最后一次估计）
print("\n  条件期望映射的形状（各箱分位 → 历史平均前瞻10日收益, 末次估计）:")
print("  " + "".join(f"箱{b+1}:{means[b]*100:>+6.2f}%  " for b in range(NB)))
