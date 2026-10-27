# -*- coding: utf-8 -*-
"""最终方案（A0 + 5%网格 + 10%死区）完整报告数据 + 净值图"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = _P.ROOT
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = _P.TRAIN
INPUT = os.path.join(BASE, "input")
ANN, BP = 252, 0.0003
W, W_SENT, K, H = 300, 250, 5, 10
GRID, DEAD = 0.05, 0.10
OUT = {}


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


def tgt3(comp, lo=0.30, hi=0.80):
    return pd.Series(np.where(comp < lo, 0.0, np.where(comp < hi, 0.5, 1.0)),
                     index=comp.index).where(comp.notna())


def grid(p, q):
    return p if q <= 0 else (p / q).round().clip(0, 1 / q) * q


def band(p, bd):
    if bd <= 0:
        return p
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

d = pd.DataFrame({"base": base, "IFraw": raw_if,
                  "IF": roll_pct(raw_if, W_SENT, minp=60).reindex(base.index).ffill(),
                  "c": close}).dropna()
d = d[(d.index >= pd.Timestamp("2019-06-04")) & (d.index <= pd.Timestamp("2026-09-30"))]
d["r"] = np.log(d["c"]).diff().fillna(0.0)
half = (d["base"] - 0.5).abs() < 1e-9
d["A0"] = d["base"].where(~half, d["IF"])
d["PRAC"] = band(grid(d["A0"], GRID), DEAD)
N_ = len(d)
YR = N_ / ANN
print(f"回测区间 {d.index[0].date()} ~ {d.index[-1].date()}，{N_} 个交易日，{YR:.2f} 年\n")

# ---------------------------------------------------------------- 因子 IC
fwd = np.log(close.shift(-H) / close)


def ic_stats(sig, label=fwd, sub=None):
    x = pd.DataFrame({"s": sig, "y": label}).dropna()
    if sub is not None:
        x = x[(x.index >= sub[0]) & (x.index <= sub[1])]
    if len(x) < 60:
        return dict(ic=np.nan, t=np.nan, n=len(x), icno=np.nan, tno=np.nan)
    ic = spearmanr(x["s"], x["y"]).statistic
    xo = x.iloc[::H]
    ico = spearmanr(xo["s"], xo["y"]).statistic
    n = len(xo)
    tno = ico * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - ico ** 2))
    return dict(ic=ic, n=len(x), icno=ico, nno=n, tno=tno)


FACT = [("量价 pv（原始信号 64维均值）", pv_avg),
        ("资金 slow（原始信号 64维均值）", sl_avg),
        ("情绪 IF（250日滚动分位）", roll_pct(raw_if, W_SENT, minp=60)),
        ("合成 composite（pv分位 0.5 + slow饱和分位 0.5）", mm["comp"])]
print("=" * 130)
print("一、因子 IC（label = ln(收盘_{t+10} / 收盘_t)，与两个神经网络同一定义；IC 为 Spearman）")
print("=" * 130)
print(f"  {'因子':<46}{'区间':<24}{'IC':>8}{'非重叠IC':>10}{'t':>7}{'样本':>7}")
OUT["ic"] = []
for nm, sig in FACT:
    for lbl, sub in (("全可用样本", None), ("回测区间 2019-06~2026-09", ("2019-06-04", "2026-09-30"))):
        st = ic_stats(sig, sub=sub)
        print(f"  {nm:<46}{lbl:<24}{st['ic']:>8.3f}{st['icno']:>10.3f}{st['tno']:>7.2f}{st.get('n',''):>7}")
        OUT["ic"].append((nm, lbl, st))
print()

print("  分年 IC（非重叠，每 10 个交易取 1 个观测）")
yrs = list(range(2019, 2027))
print(f"  {'因子':<46}" + "".join(f"{y:>8}" for y in yrs))
for nm, sig in FACT:
    row = []
    for y in yrs:
        st = ic_stats(sig, sub=(f"{y}-01-01", f"{y}-12-31"))
        row.append(st["icno"])
    print(f"  {nm:<46}" + "".join(f"{v:>8.3f}" if v == v else f"{'—':>8}" for v in row))
print()

print("  分箱单调性（按因子自身滚动分位分成 5 组，组内未来 10 日平均收益 %，回测区间）")
print(f"  {'因子':<20}{'Q1(最低)':>10}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'Q5(最高)':>10}{'Q5−Q1':>9}")
for nm, col in (("量价 pv 分位", "p"), ("资金 slow 分位", "s"), ("情绪 IF 分位", "IF"),
                ("合成 composite", "comp")):
    s = mm[col] if col in mm.columns else d["IF"]
    x = pd.DataFrame({"s": s, "y": fwd}).dropna()
    x = x[(x.index >= "2019-06-04") & (x.index <= "2026-09-30")]
    q = pd.qcut(x["s"].rank(method="first"), 5, labels=False)
    g = x.groupby(q)["y"].mean() * 100
    print(f"  {nm:<20}" + "".join(f"{g.get(i, np.nan):>9.2f}%" for i in range(5))
          + f"{g.get(4, np.nan)-g.get(0, np.nan):>9.2f}p")
print()

# ---------------------------------------------------------------- 绩效
def perf(p, label=""):
    pr = (p.shift(1) * d["r"]).fillna(0.0)
    to = p.diff().abs().fillna(0.0)
    net = pr - to * BP
    cq = net.cumsum()
    ann = net.mean() * ANN
    vol = net.std() * np.sqrt(ANN)
    mdd = float((np.exp(cq - cq.cummax()) - 1).min())
    eq = np.exp(cq)
    peak = eq.cummax()
    dd_days = int((eq < peak).astype(int).groupby((eq >= peak).cumsum()).cumsum().max())
    return dict(label=label, sharpe=ann / vol, ann=ann, vol=vol, cum=np.expm1(cq.iloc[-1]),
                mdd=mdd, calmar=ann / abs(mdd), pos=float(p.mean()), pos_med=float(p.median()),
                pos_std=float(p.std()), turn=float(to.sum() / YR), cost=float(to.sum() / YR * BP),
                nchg=int((to > 1e-9).sum()), nlev=int(len(np.unique(np.round(p.dropna().values, 6)))),
                win=float((net > 0).mean()), dd_days=dd_days, unit=ann / float(p.mean()),
                net=net, eq=eq)


S = perf(d["PRAC"], "策略")
B = perf(d["base"], "M3基准")
bh_ret = d["r"].copy()
cqb = bh_ret.cumsum()
BM = dict(label="买入持有", sharpe=bh_ret.mean() / bh_ret.std() * np.sqrt(ANN),
          ann=bh_ret.mean() * ANN, vol=bh_ret.std() * np.sqrt(ANN), cum=np.expm1(cqb.iloc[-1]),
          mdd=float((np.exp(cqb - cqb.cummax()) - 1).min()), pos=1.0,
          dd_days=int((np.exp(cqb) < np.exp(cqb).cummax()).astype(int)
                      .groupby((np.exp(cqb) >= np.exp(cqb).cummax()).cumsum()).cumsum().max()),
          win=float((bh_ret > 0).mean()))
BM["calmar"] = BM["ann"] / abs(BM["mdd"])

print("=" * 130)
print("二、核心绩效（单边 3bp；年化 = 日对数收益均值 × 252）")
print("=" * 130)
print(f"  {'指标':<24}{'策略（最终版）':>17}{'M3 基准':>15}{'买入持有':>15}")
fmt = [("累计收益", "cum", "{:.1%}"), ("年化收益", "ann", "{:.2%}"), ("年化波动", "vol", "{:.2%}"),
       ("夏普比率", "sharpe", "{:.2f}"), ("最大回撤", "mdd", "{:.1%}"), ("Calmar", "calmar", "{:.2f}"),
       ("最长回撤持续（交易日）", "dd_days", "{}"), ("日胜率", "win", "{:.1%}"),
       ("日均仓位", "pos", "{:.1%}")]
for lbl, k, f in fmt:
    a = f.format(S[k]); b = f.format(B[k]); c = f.format(BM[k]) if k in BM else "—"
    print(f"  {lbl:<24}{a:>17}{b:>15}{c:>15}")
print(f"  {'单位仓位年化':<24}{S['ann']/S['pos']:>17.1%}{B['ann']/B['pos']:>15.1%}{'—':>15}")

# 超额
ex = S["net"] - B["net"]
ex_m = S["net"] - bh_ret
ir_ex = ex_m.mean() / ex_m.std() * np.sqrt(ANN) if ex_m.std() else np.nan
t_ex = ex_m.mean() / (ex_m.std() / np.sqrt(len(ex_m))) if ex_m.std() else np.nan
print(f"\n  相对买入持有超额：年化 {(S['ann']-BM['ann'])*100:+.2f}pct，"
      f"累计复利差 {(S['cum']-BM['cum'])*100:+.0f}pct，日均 {ex_m.mean()*1e4:+.2f}bp，"
      f"IR {ir_ex:.2f}，t {t_ex:.2f}")
exb_m = B["net"] - bh_ret
print(f"  （M3 基准相对买入持有：年化 {(B['ann']-BM['ann'])*100:+.2f}pct，"
      f"IR {exb_m.mean()/exb_m.std()*np.sqrt(ANN):.2f}，t {exb_m.mean()/(exb_m.std()/np.sqrt(len(exb_m))):.2f}）")

# 月度
mo = d.copy()
mo["strat"] = S["net"]; mo["bm"] = bh_ret
gm = mo.groupby([mo.index.year, mo.index.month]).sum()
mw = float((gm["strat"] > 0).mean())
print(f"  月度胜率 {mw:.0%}（{len(gm)} 个月：{int((gm['strat']>0).sum())} 正 / {int((gm['strat']<=0).sum())} 负）")
up = gm["bm"] > 0
capt_up = gm.loc[up, "strat"].mean() / gm.loc[up, "bm"].mean() * 100
capt_dn = gm.loc[~up, "strat"].mean() / gm.loc[~up, "bm"].mean() * 100
print(f"  上行捕获 {capt_up:.0f}%（上涨 {int(up.sum())} 个月：策略 {gm.loc[up,'strat'].mean()*100:+.2f}% "
      f"vs 指数 {gm.loc[up,'bm'].mean()*100:+.2f}%）")
print(f"  下行捕获 {capt_dn:.0f}%（下跌 {int((~up).sum())} 个月：策略 {gm.loc[~up,'strat'].mean()*100:+.2f}% "
      f"vs 指数 {gm.loc[~up,'bm'].mean()*100:+.2f}%）")
print()

# ---------------------------------------------------------------- 交易级
print("=" * 130)
print("三、交易级与仓位结构")
print("=" * 130)
print(f"  单边换手 {S['turn']:.2f} 倍/年（累计 {S['turn']*YR:.1f} 倍），年化成本 {S['cost']*100:.2f}%"
      f"（累计 {S['cost']*YR*100:.2f}%）")
print(f"  调整次数 {S['nchg']} 次 = {S['nchg']/YR:.1f} 次/年，平均单次 |Δ仓位| "
      f"{float(d['PRAC'].diff().abs().sum()/max(1,S['nchg'])):.2f}；仓位取值 {S['nlev']} 档")
occ = d["PRAC"].value_counts(normalize=True).sort_index()
z = float(occ.get(0.0, 0.0)); o = float(occ.get(1.0, 0.0))
print(f"  仓位结构：空仓(0%) {z:.1%}，满仓(100%) {o:.1%}，中间档 {1-z-o:.1%}；"
      f"最高频档位：" + "，".join(f"{k:.0%}({v:.1%})" for k, v in occ.nlargest(5).items()))
print(f"  日均仓位 {S['pos']:.1%}（中位 {S['pos_med']:.0%}，std {S['pos_std']:.1%}）")
inpos = d["PRAC"].shift(1) > 1e-9
print(f"  在仓日胜率 {(S['net'][inpos] > 0).mean():.1%}（在仓 {int(inpos.sum())} 天），"
      f"空仓日 {(~inpos).sum()} 天收益为 0（仅成本）")
# 段
p = d["PRAC"]
seg_in = (p > 1e-9).astype(int)
grp = (seg_in != seg_in.shift(1)).cumsum()
segs = []
for g, idx in d.groupby(grp):
    hold = float(p.loc[idx.index[0]])
    if hold > 1e-9:
        segs.append(("持仓", idx.index[0], idx.index[-1], len(idx), hold))
    else:
        segs.append(("空仓", idx.index[0], idx.index[-1], len(idx), 0.0))
hold_segs = [s for s in segs if s[0] == "持仓"]
got = []
for _, a, b2, n, h in hold_segs:
    got.append(np.expm1(d.loc[a:b2, "r"].sum()))
got = np.array(got)
print(f"  持仓段 {len(hold_segs)} 段，段胜率 {(got>0).mean():.0%}，平均 {got.mean()*100:+.2f}%，"
      f"赔率 {got[got>0].mean()/abs(got[got<0].mean()):.2f}，平均持仓 {np.mean([s[3] for s in hold_segs]):.0f} 天"
      f"（中位 {np.median([s[3] for s in hold_segs]):.0f}，最长 {max(s[3] for s in hold_segs)}）")
print(f"  空仓段 {len(segs)-len(hold_segs)} 段")
chg = d["PRAC"].diff().dropna()
chg = chg[chg.abs() > 1e-9]
path = {}
for a, b2 in zip(d["PRAC"].shift(1)[chg.index], d["PRAC"][chg.index]):
    key = f"{a:.0%}→{b2:.0%}"
    path[key] = path.get(key, 0) + 1
print("  换仓路径（前 8 高频）：" + "，".join(f"{k} {v}次" for k, v in sorted(path.items(), key=lambda z: -z[1])[:8]))
print()

# ---------------------------------------------------------------- 分段超额
print("=" * 130)
print("四、分段超额（段内策略 vs 段内指数）")
print("=" * 130)
for nm, dd_ in (("持仓段", [s for s in segs if s[0] == "持仓"]), ("空仓段", [s for s in segs if s[0] == "空仓"])):
    vs, vbm = [], []
    for _, a, b2, n, h in dd_:
        vs.append(np.expm1(S["net"].loc[a:b2].sum()))
        vbm.append(np.expm1(d.loc[a:b2, "r"].sum()))
    vs, vbm = np.array(vs), np.array(vbm)
    print(f"  {nm} {len(dd_)} 段，平均 {np.mean([x[3] for x in dd_]):.0f} 天："
          f"胜率 {(vs>vbm).mean():.0%}，策略 {vs.mean()*100:+.2f}% vs 指数 {vbm.mean()*100:+.2f}%，"
          f"超额 {(vs-vbm).mean()*100:+.2f}pct")
print()

# ---------------------------------------------------------------- 分年
print("=" * 130)
print("五、分年统计")
print("=" * 130)
print(f"  {'年份':<12}{'指数':>8}{'策略':>8}{'超额':>8}{'策略夏普':>9}{'策略回撤':>9}"
      f"{'指数回撤':>9}{'日均仓位':>9}{'空/中/满':>12}{'调仓':>6}")
OUT["year"] = []
for y in yrs:
    m = d.index.year == y
    if m.sum() == 0:
        continue
    sub_d = d[m]
    ix = np.expm1(sub_d["r"].sum())
    sn = S["net"][m]
    st = np.expm1(sn.sum())
    cq = sn.cumsum()
    mdd = float((np.exp(cq - cq.cummax()) - 1).min())
    cqi = sub_d["r"].cumsum()
    mdd_i = float((np.exp(cqi - cqi.cummax()) - 1).min())
    shp = sn.mean() / sn.std() * np.sqrt(ANN) if sn.std() > 0 else np.nan
    pp = sub_d["PRAC"]
    occ_y = pp.value_counts(normalize=True)
    z_y = float(occ_y.get(0.0, 0.0)); o_y = float(occ_y.get(1.0, 0.0))
    nch = int((pp.diff().abs() > 1e-9).sum())
    tag = " (6月起)" if y == 2019 else (" (至9月)" if y == 2026 else "")
    print(f"  {str(y)+tag:<12}{ix*100:>7.1f}%{st*100:>7.1f}%{(st-ix)*100:>7.1f}p{shp:>9.2f}"
          f"{mdd*100:>8.1f}%{mdd_i*100:>8.1f}%{pp.mean()*100:>8.1f}%"
          f"{z_y:>5.0%}/{1-z_y-o_y:>4.0%}/{o_y:>3.0%}{nch:>6d}")
    OUT["year"].append(dict(year=int(y), idx=float(ix), strat=float(st), mdd=float(mdd),
                            mdd_i=float(mdd_i), sharpe=float(shp), pos=float(pp.mean()), nchg=nch))
print()

# ---------------------------------------------------------------- 净值图
eq_s = np.exp(S["net"].cumsum())
eq_b = np.exp(bh_ret.cumsum())
eq_rel = eq_s / eq_b
fig, ax = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True,
                       gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
ax[0].plot(eq_s.index, eq_s.values, lw=1.9, color="#c0392b", label=f"策略（最终版）累计 {S['cum']:.0%}")
ax[0].plot(eq_b.index, eq_b.values, lw=1.5, color="#34495e", label=f"基准（中证1000）累计 {BM['cum']:.0%}")
ax[0].plot(eq_rel.index, eq_rel.values, lw=1.4, color="#2980b9", ls="--",
           label=f"策略 / 基准 相对净值 {eq_rel.iloc[-1]:.2f}")
ax[0].axhline(1.0, color="#999", lw=0.9)
ax[0].set_yscale("log")
ax[0].set_ylabel("净值（起点 = 1，对数刻度）")
ax[0].set_title(f"中证1000 择时策略净值走势　{d.index[0].date()} ~ {d.index[-1].date()}"
                f"（{N_} 个交易日）", fontsize=13, pad=10)
ax[0].legend(loc="upper left", frameon=False, fontsize=10)
ax[0].grid(alpha=0.25, which="both")

ax[1].fill_between(d.index, 0, d["PRAC"].values, step="post", color="#c0392b", alpha=0.45, lw=0)
ax[1].set_ylim(-0.02, 1.05)
ax[1].set_ylabel("仓位")
ax[1].grid(alpha=0.25)
ax[1].set_yticks([0, 0.5, 1.0])
ax[1].set_yticklabels(["0%", "50%", "100%"])
png = os.path.join(_P.REPORTS, "净值走势图_策略vs基准.png")
png = os.path.abspath(png)
fig.savefig(png, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"净值图已保存：{png}")
print(f"  期末净值：策略 {eq_s.iloc[-1]:.3f}，基准 {eq_b.iloc[-1]:.3f}，相对 {eq_rel.iloc[-1]:.3f}")
print(f"  策略/基准 相对净值 最高 {eq_rel.max():.3f}（{eq_rel.idxmax().date()}），"
      f"最低 {eq_rel.min():.3f}（{eq_rel.idxmin().date()}）")

json.dump({k: v for k, v in OUT.items()},
          open(os.path.join(_P.REPORTS, "_report_out.json"), "w"),
          ensure_ascii=False, indent=1, default=float)
print("\n完成")
