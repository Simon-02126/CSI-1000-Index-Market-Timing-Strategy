# -*- coding: utf-8 -*-
"""M3 + IF overlay（方案D最优版）完整绩效：策略指标 / 指数指标 / 分年统计

策略：M3 基础仓位 → IF_会员多空比 250日分位<0.20 且连续3日 → 仓位清零
"""
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
TRAIN = _P.TRAIN
INPUT = os.path.join(BASE, "input")
ANN, COST = 252, 0.0003
START, END = "2019-06-04", "2026-09-30"
OOS = "2025-01-02"


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try: return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError): continue
    raise RuntimeError(fp)


fp = glob.glob(os.path.join(INPUT, "*中证1000*.csv"))[0]
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


def pos_mode(t, k=5):
    v = t.values; o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        if not len(w): continue
        vals, cnt = np.unique(w, return_counts=True); o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


pvS = {sd: load_tag(f"s{sd}") for sd in range(6)}
slS = {sd: load_tag(f"slow_s{sd}") for sd in range(6)}
pvS = {k: v for k, v in pvS.items() if v is not None}
slS = {k: v for k, v in slS.items() if v is not None}
pv_avg = pd.concat(pvS.values(), axis=1).mean(axis=1).dropna()
sl_avg = pd.concat(slS.values(), axis=1).mean(axis=1).dropna()

pv_pct = roll_pct(pv_avg, 300)
sl_pct = ((roll_pct(sl_avg, 300) - 0.30) / 0.35).clip(0, 1)
comp = (0.5 * pv_pct + 0.5 * sl_pct).dropna()
base = pos_mode(tgt3(comp, 0.30, 0.80), 5)

panel = pd.read_csv(os.path.join(INPUT, "情绪_面板_日频.csv"), encoding="utf-8-sig", parse_dates=["日期"])
panel = panel.drop_duplicates("日期").set_index("日期").sort_index()
IFp = roll_pct(pd.to_numeric(panel["IF_会员多空比"], errors="coerce").dropna(), 250, minp=60)
cond = (IFp < 0.20).rolling(3, min_periods=3).sum().fillna(0) >= 3
TRIG = cond.reindex(base.index).fillna(False).astype(bool)
pos = base.where(~TRIG, 0.0)

d = pd.DataFrame({"pos": pos, "base": base, "c": close, "trig": TRIG}).dropna()
d = d[(d.index >= START) & (d.index <= END)]
r = np.log(d["c"]).diff()
turn = d["pos"].diff().abs().fillna(0)
turn_b = d["base"].diff().abs().fillna(0)
strat = (d["pos"].shift(1) * r - turn * COST).fillna(0.0)
bh = r.fillna(0.0)
base_r = (d["base"].shift(1) * r - turn_b * COST).fillna(0.0)
gap = ((d["base"].shift(1) - d["pos"].shift(1)) * r - (turn - turn_b) * COST).fillna(0.0)
ovl = -gap      # overlay 相对 M3 的逐日贡献（正=改善）


def stats(x, pos_=None):
    eq = x.cumsum()
    cum = float(np.expm1(eq.iloc[-1]))
    ann = float(x.mean() * ANN)
    vol = float(x.std() * np.sqrt(ANN))
    mdd = float(np.expm1((eq - eq.cummax()).min()))
    # 回撤持续
    dd = eq - eq.cummax()
    in_dd = dd < -1e-12
    longest, cur = 0, 0
    for f in in_dd.values:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    nz = x[np.abs(x) > 1e-12]
    return dict(cum=cum, ann=ann, vol=vol, sharpe=ann / vol if vol else np.nan,
                mdd=mdd, calmar=ann / abs(mdd) if mdd else np.nan, dd_days=longest,
                win=(nz > 0).mean(), n=len(x))


S, B, BD = stats(strat, d["pos"]), stats(bh), stats(base_r)
ex = strat - bh
ir = float(ex.mean() * ANN / (ex.std() * np.sqrt(ANN)))

print("=" * 110)
print(f"【一、核心绩效】{START} ~ {END}（{len(strat)} 交易日 / {len(strat)/ANN:.2f} 年）")
print("=" * 110)
rows = [("累计收益", "cum", 1), ("年化收益", "ann", 1), ("年化波动", "vol", 1),
        ("夏普比率", "sharpe", 2), ("最大回撤", "mdd", 1), ("Calmar", "calmar", 2),
        ("最大回撤持续(交易日)", "dd_days", 0), ("日胜率", "win", 1)]
print(f"  {'指标':<24}{'策略(M3+overlay)':>20}{'M3基准':>14}{'买入持有':>14}")
for nm, k, dec in rows:
    if k in ("sharpe", "calmar", "dd_days"):
        f_ = lambda z: f"{z:.{dec}f}"
    else:
        f_ = lambda z: f"{z*100:.{dec}f}%"
    print(f"  {nm:<24}{f_(S[k]):>20}{f_(BD[k]):>14}{f_(B[k]):>14}")

# 超额
ex = strat - bh
ts = float(ex.mean() / (ex.std() / np.sqrt(len(ex))))
print(f"\n  超额收益: 年化 {S['ann']-B['ann']:>+6.2%}  累计(复利差) {S['cum']-B['cum']:>+8.1%}  "
      f"日均超额 {ex.mean()*1e4:>+6.2f}bp  超额IR {ir:>+5.2f}  超额t {ts:>+5.2f}")

# 月度
ms = strat.resample("ME").sum(); mb = bh.resample("ME").sum()
mb3 = base_r.resample("ME").sum()
print(f"\n  月度胜率（策略）{(ms>0).mean():.0%}（{len(ms)} 个月：{(ms>0).sum()} 正 / {(ms<=0).sum()} 负；M3基准 {(mb3>0).mean():.0%}）")
up = mb > 0; dn = mb < 0
for nm, x in [("M3+overlay", ms), ("M3 基准  ", mb3)]:
    uc = np.expm1(x[up]).mean() / np.expm1(mb[up]).mean()
    dc = np.expm1(x[dn]).mean() / np.expm1(mb[dn]).mean()
    print(f"  {nm}: 上行捕获 {uc:>5.0%}（涨月 策略 {np.expm1(x[up]).mean()*100:+.2f}% vs 指数 {np.expm1(mb[up]).mean()*100:+.2f}%，{up.sum()}月）"
          f"  下行捕获 {dc:>5.0%}（跌月 策略 {np.expm1(x[dn]).mean()*100:+.2f}% vs 指数 {np.expm1(mb[dn]).mean()*100:+.2f}%，{dn.sum()}月）")

print("\n" + "=" * 110)
print("【二、交易级 / 仓位结构】")
print("=" * 110)
cost = float(turn.sum() * COST)
print(f"  调仓次数 {int((turn>1e-9).sum())} 次 → {int((turn>1e-9).sum())/(len(strat)/ANN):.1f} 次/年；"
      f"平均单次幅度 {float(turn[turn>1e-9].mean()):.2f}")
print(f"  年化换手(单边) {float(turn.sum())/(len(strat)/ANN):.1f}x；年化成本 {cost/(len(strat)/ANN):.2%}（累计 {cost:.2%}）")
print(f"  时间占比 满仓 {(d['pos']>=0.99).mean():.1%} / 半仓 {((d['pos']>0.01)&(d['pos']<0.99)).mean():.1%} / 空仓 {(d['pos']<=0.01).mean():.1%}")
print(f"  日均仓位 {d['pos'].mean():.1%}（中位 {d['pos'].median():.0%}，std {d['pos'].std():.1%}）")
print(f"  基础M3日均仓位 {d['base'].mean():.1%} → overlay 后 {d['pos'].mean():.1%}；净压降 {(d['base']-d['pos']).sum()/len(d):.1%}")

# 持仓段
def segments(p):
    pv = p.values; segs = []; cur = None
    for i, v in enumerate(pv):
        if v > 0.01 and cur is None: cur = [i, i]
        elif v > 0.01: cur[1] = i
        elif cur is not None: segs.append(tuple(cur)); cur = None
    if cur: segs.append(tuple(cur))
    return segs
segs = segments(d["pos"])
lr = np.log(d["c"]).values
hold_w, hold_n, hold_d = [], 0, []
for a, b in segs:
    ret = lr[b] - lr[a]
    hold_w.append(ret); hold_d.append(b - a + 1)
hw = np.array(hold_w)
print(f"  持仓段 {len(segs)} 段；胜率 {(hw>0).mean():.1%}；平均 {np.expm1(hw.mean())*100:+.2f}%；"
      f"赔率 {hw[hw>0].mean()/abs(hw[hw<0].mean()):.2f}" if (hw>0).any() and (hw<0).any() else "")
print(f"  平均持仓 {np.mean(hold_d):.0f} 天（中位 {np.median(hold_d):.0f}，最长 {max(hold_d)}，最短 {min(hold_d)}）")

# 换仓路径
chg = d["pos"].diff().fillna(0)
print(f"  换仓路径: 半→空 {int(((chg<0)&(d['base'].shift(1)>=0.4)&(d['base'].shift(1)<=0.6)).sum())} 次、"
      f"空→半 {int(((chg>0)&(d['base']>=0.4)&(d['base']<=0.6)).sum())} 次、"
      f"半→满 {int(((chg>0)&(d['base']>=0.99)).sum())} 次、满→半 {int(((chg<0)&(d['base'].shift(1)>=0.99)).sum())} 次")

print("\n" + "=" * 110)
print("【三、分年统计】")
print("=" * 110)
print(f"  {'年份':<10}{'策略':>9}{'指数':>9}{'超额':>9}{'策略夏普':>10}{'策略回撤':>10}{'指数回撤':>10}"
      f"{'日均仓位':>10}{'满/半/空':>14}{'调仓':>6}{'触发天':>8}{'overlay贡献':>12}")
for y in sorted(set(d.index.year)):
    m = d.index.year == y
    if m.sum() < 20: continue
    sy, iy = strat[d.index.year == y], bh[d.index.year == y]
    s_y = stats(sy); i_y = stats(iy)
    dd = d[m]
    tg = int(dd["trig"].sum())
    lab = f"{y}" if not (y == 2019 or y == 2026) else (f"{y}(6月起)" if y == 2019 else f"{y}(至9月)")
    posmix = f"{(dd['pos']>=0.99).mean():.0%}/{(dd['pos'].between(0.01,0.99)).mean():.0%}/{(dd['pos']<=0.01).mean():.0%}"
    print(f"  {lab:<10}{s_y['cum']:>8.1%}{i_y['cum']:>9.1%}{s_y['cum']-i_y['cum']:>+9.1%}"
          f"{s_y['sharpe']:>10.2f}{s_y['mdd']:>10.1%}{i_y['mdd']:>10.1%}"
          f"{dd['pos'].mean():>10.1%}{posmix:>14}{int((dd['pos'].diff().abs()>1e-9).sum()):>6}{tg:>8}"
          f"{ovl[d.index.year == y].sum()*100:>+11.1f}p")

print("\n" + "=" * 110)
print(f"【四、真样本外 {OOS} ~ {END}】")
print("=" * 110)
mo = d.index >= OOS
s_o, b_o = stats(strat[mo]), stats(bh[mo])
print(f"  策略: 累计{s_o['cum']:>7.1%} 年化{s_o['ann']:>7.1%} 波动{s_o['vol']:>6.1%} 夏普{s_o['sharpe']:>5.2f} "
      f"回撤{s_o['mdd']:>7.1%} Calmar{s_o['calmar']:>5.2f} 日均仓位{d['pos'][mo].mean():>5.1%} 触发{int(d['trig'][mo].sum())}天")
print(f"  指数: 累计{b_o['cum']:>7.1%} 年化{b_o['ann']:>7.1%} 波动{b_o['vol']:>6.1%} 夏普{b_o['sharpe']:>5.2f} 回撤{b_o['mdd']:>7.1%}")

print("\n" + "=" * 110)
print("【五、分段超额（持仓段 / 空仓段）】")
print("=" * 110)
def seg_stats(p):
    sg = segments(p); rows = []
    for a, b in sg:
        se = (d["pos"].shift(1) * r).iloc[a:b+1].sum() - (turn * COST).iloc[a:b+1].sum()
        ie = r.iloc[a:b+1].sum()
        rows.append((b - a + 1, se, ie))
    return pd.DataFrame(rows, columns=["n", "s", "i"])
hs = seg_stats(d["pos"])
zero_segs = []
pv_ = d["pos"].values; cur = None
for i, v in enumerate(pv_):
    if v <= 0.01 and cur is None: cur = [i, i]
    elif v <= 0.01: cur[1] = i
    elif cur is not None: zero_segs.append(tuple(cur)); cur = None
if cur: zero_segs.append(tuple(cur))
zs = pd.DataFrame([(b - a + 1, r.iloc[a:b+1].sum()) for a, b in zero_segs], columns=["n", "i"])
allsegs = [(b - a + 1, (d["pos"].shift(1) * r).iloc[a:b+1].sum() - (turn * COST).iloc[a:b+1].sum(), r.iloc[a:b+1].sum())
           for a, b in segments(d["pos"])] + [(b - a + 1, 0.0 - (turn * COST).iloc[a:b+1].sum(), r.iloc[a:b+1].sum())
                                              for a, b in zero_segs]
al = pd.DataFrame(allsegs, columns=["n", "s", "i"])
print(f"  {'类型':<10}{'段数':>6}{'平均天数':>10}{'跑赢段数':>10}{'胜率':>8}{'平均策略':>10}{'平均指数':>10}{'平均超额':>10}")
for nm, x, has_s in [("持仓段", hs, True), ("空仓段", zs, False), ("全部", al, True)]:
    if not len(x): continue
    if has_s:
        win = (x["s"] - x["i"] > 0).sum()
        print(f"  {nm:<10}{len(x):>6}{x['n'].mean():>10.0f}{win:>10}{win/len(x):>8.1%}"
              f"{np.expm1(x['s'].mean())*100:>9.2f}%{np.expm1(x['i'].mean())*100:>9.2f}%"
              f"{(np.expm1(x['s'].mean())-np.expm1(x['i'].mean()))*100:>+9.2f}%")
    else:
        win = (x["i"] < 0).sum()
        print(f"  {nm:<10}{len(x):>6}{x['n'].mean():>10.0f}{win:>10}{win/len(x):>8.1%}"
              f"{0.0:>9.2f}%{np.expm1(x['i'].mean())*100:>9.2f}%{-np.expm1(x['i'].mean())*100:>+9.2f}%")

print("\n" + "=" * 110)
print("【六、归因：超额 = 躲跌 − 踏空 − 成本】")
print("=" * 110)
rp = d["pos"].shift(1)
inpos = pd.Series(np.isclose(d["pos"].shift(1).fillna(0), 0.0), index=d.index)
half = d["pos"].shift(1).fillna(0)
miss = ((1 - half) * r).where(r > 0, 0).sum()
avoid = ((1 - half) * (-r)).where(r < 0, 0).sum()
print(f"  躲跌收益 {avoid*100:>+7.1f}（log%）  − 踏空损失 {miss*100:>7.1f}  − 成本 {cost*100:>5.1f}  "
      f"= 超额 {(strat.sum()-bh.sum())*100:>+6.1f}")
print(f"  恒等式核对: {avoid*100 - miss*100 - cost*100:+.1f} vs 实际 {(strat.sum()-bh.sum())*100:+.1f}")

print("\n" + "=" * 110)
print("【七、触发事件清单】（IF分位<0.20 连续3日 → 清仓）")
print("=" * 110)
tgv = d["trig"].values; segs_t = []; cur = None
for i, v in enumerate(tgv):
    if v and cur is None: cur = [i, i]
    elif v: cur[1] = i
    elif cur is not None: segs_t.append(tuple(cur)); cur = None
if cur: segs_t.append(tuple(cur))
lr_full = np.log(d["c"]).values
N_ = len(d)
rows = []
for a, b in segs_t:
    b2 = min(b + 1, N_ - 1)          # 空仓生效的最后一个收益日 = 恢复仓位前一日
    ie = lr_full[b2] - lr_full[a]    # 实际平掉的行情：r[a+1..b2]
    bp_mean = d["base"].iloc[a:b2 + 1].mean()
    contrib = -float(gap.iloc[a:b2 + 1].sum())
    rows.append((d.index[a], d.index[b], b2 - a, np.expm1(ie), bp_mean, contrib, "有效" if contrib > 0 else "无效"))
ev = pd.DataFrame(rows, columns=["开始", "结束", "空仓收益日", "指数", "基础仓位", "净贡献", "判定"])
print(f"  共 {len(ev)} 段，累计 {int(ev['空仓收益日'].sum())} 天（占 {ev['空仓收益日'].sum()/len(d):.1%}）")
ne, ni = int((ev['判定'] == '有效').sum()), int((ev['判定'] == '无效').sum())
print(f"  有效 {ne} 段 / 无效 {ni} 段 = 段胜率 {ne/len(ev):.0%}")
eff = ev[ev['基础仓位'] > 0.01]
print(f"  其中「有实际影响」的触发 {len(eff)} 段（基础仓位>1%），"
      f"改善 {(eff['净贡献']>0).sum()} 段 / 恶化 {(eff['净贡献']<0).sum()} 段 = 命中率 {(eff['净贡献']>0).mean():.0%}；"
      f"无影响（当时已空仓）{len(ev)-len(eff)} 段")
print(f"  有效段合计贡献 {ev.loc[ev['判定']=='有效','净贡献'].sum()*100:+.1f}pct，无效段合计 {ev.loc[ev['判定']=='无效','净贡献'].sum()*100:+.1f}pct，"
      f"净 {ev['净贡献'].sum()*100:+.1f}pct log")
_mask = pd.Series(False, index=d.index)
for a, b in segs_t: _mask.iloc[a:min(b + 2, N_)] = True
print(f"  （核对：overlay−M3 全日差之和 {-gap.sum()*100:+.1f}pct log；触发相关日 {-gap[_mask].sum()*100:+.2f}pct；其余日 {-gap[~_mask].sum()*100:+.3f}pct）")
print(f"\n  {'开始':<12}{'结束':<12}{'空仓日':>6}{'区间指数':>9}{'基础仓位':>9}{'净贡献pct':>10}{'判定':>6}")
for _, x in ev.iterrows():
    print(f"  {x['开始'].date()!s:<12}{x['结束'].date()!s:<12}{int(x['空仓收益日']):>6}{x['指数']*100:>8.1f}%"
          f"{x['基础仓位']:>9.0%}{x['净贡献']*100:>+10.2f}{x['判定']:>6}")

print("\n" + "=" * 110)
print("【八、对照基准】")
print("=" * 110)
print(f"  {'方案':<26}{'累计':>9}{'年化':>8}{'波动':>8}{'夏普':>7}{'回撤':>9}{'Calmar':>8}{'日均仓位':>10}")
for nm, x, ps in [("M3+overlay（最终）", strat, d["pos"]), ("M3 基准", base_r, d["base"]),
                  ("买入持有", bh, pd.Series(1.0, index=bh.index))]:
    st = stats(x)
    print(f"  {nm:<26}{st['cum']:>8.1%}{st['ann']:>8.1%}{st['vol']:>8.1%}{st['sharpe']:>7.2f}"
          f"{st['mdd']:>9.1%}{st['calmar']:>8.2f}{ps.mean():>10.0%}")
