# -*- coding: utf-8 -*-
"""
情绪因子「信号输入」口径体检（区别于否决口径）
label = ln(close_{t+10}/close_t)，与 pv / slow 完全一致。

用法改变 → 筛选标准必须改变：
  否决口径关心「尾部单调 + 阈值符号」，信号输入口径关心
  「全样本秩IC + 符号可预知 + 与 pv/slow 正交 + 形状单调」。
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _P.ROOT
PANEL = os.path.join(ROOT, 'input', '情绪_面板_日频.csv')
IDX = os.path.join(ROOT, 'input', '中证1000_000852.SH.csv')
OUT = _P.TRAIN
H = 10
ANN = 252


# ---------------- 基础工具 ----------------
def roll_pct(s, win, minp=250):
    v = s.values.astype(float)
    n = len(v)
    o = np.full(n, np.nan)
    mp = min(minp, max(60, win // 3))
    for i in range(n):
        lo = max(0, i - win)
        if i - lo < mp:
            continue
        w = v[lo:i]
        w = w[~np.isnan(w)]
        if len(w) < mp or np.isnan(v[i]):
            continue
        o[i] = (w < v[i]).sum() / len(w)
    return pd.Series(o, index=s.index)


def sp(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 30:
        return np.nan
    if np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return spearmanr(a[m], b[m]).statistic


def roll_ic(xv, yv, win=500, step=20):
    n = len(xv)
    out = []
    for end in range(win, n + 1, step):
        out.append(sp(xv[end - win:end], yv[end - win:end]))
    return np.array(out, dtype=float)


def load_idx():
    for enc in ('utf-8', 'gbk', 'gb18030'):
        try:
            d = pd.read_csv(IDX, encoding=enc)
            break
        except Exception:
            continue
    dc = [c for c in d.columns if '日期' in c][0]
    cc = [c for c in d.columns if '收盘' in c][0]
    s = pd.to_datetime(d[dc])
    v = pd.to_numeric(d[cc], errors='coerce')
    ser = pd.Series(v.values, index=s).dropna().sort_index()
    ser = ser[~ser.index.duplicated(keep='last')]
    return ser


def load_seed_mean(prefix):
    files = sorted([f for f in os.listdir(OUT) if f.startswith(prefix) and f.endswith('.csv')])
    acc = {}
    for f in files:
        d = pd.read_csv(os.path.join(OUT, f), index_col=0)
        col = [c for c in d.columns if c.startswith('信号') and '均值' not in c]
        s = d[col[0]]
        s.index = pd.to_datetime(s.index)
        for dt, val in s.items():
            acc.setdefault(dt, []).append(val)
    return pd.Series({k: np.nanmean(v) for k, v in acc.items()}).sort_index()


# ---------------- 数据 ----------------
close = load_idx()
panel = pd.read_csv(PANEL, encoding='utf-8')
panel['日期'] = pd.to_datetime(panel['日期'])
panel = panel.set_index('日期').sort_index()
panel = panel.reindex(close.index)

fwd = np.log(close.shift(-H) / close)
ret1 = np.log(close).diff()
vol20 = ret1.rolling(20).std()

pv = load_seed_mean('s').reindex(close.index)
slow = load_seed_mean('slow_s').reindex(close.index)

FIELDS = [c for c in panel.columns]
print('=' * 118)
print('情绪因子「信号输入」口径体检   label = ln(close_{t+10}/close_t)   pv/slow 亦为同 label')
print('=' * 118)
print(f'指数区间 {close.index[0].date()} ~ {close.index[-1].date()}   共 {len(close)} 日')
print(f'pv 覆盖 {pv.notna().sum()} 日   slow 覆盖 {slow.notna().sum()} 日')

# ---------------- 表 1：逐字段体检 ----------------
rows = []
detail = {}
for f in FIELDS:
    x = panel[f]
    y = fwd
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 200:
        continue
    xv = x[m].values.astype(float)
    yv = y[m].values.astype(float)
    ic = sp(xv, yv)
    # 非重叠 IC（每 H 个取 1，消除重叠自相关导致的 t 虚高）
    xo, yo = xv[::H], yv[::H]
    ico = sp(xo, yo)
    no = len(xo)
    t_o = ico * np.sqrt(no - 2) / np.sqrt(max(1e-12, 1 - ico ** 2)) if np.isfinite(ico) else np.nan
    # IC 对 |label|（波动率代理检测）
    icabs = sp(xv, np.abs(yv))
    # 8 段稳定性
    segs = np.array_split(np.arange(n), 8)
    seg_ic = np.array([sp(xv[s_], yv[s_]) for s_ in segs], dtype=float)
    seg_ic = seg_ic[np.isfinite(seg_ic)]
    major = np.sign(ic)
    same = int((np.sign(seg_ic) == major).sum())
    rng = float(np.nanmax(seg_ic) - np.nanmin(seg_ic)) if len(seg_ic) else np.nan
    # 滚动 500 日 IC 最差值
    ric = roll_ic(xv, yv, win=500, step=20)
    ric = ric[np.isfinite(ric)]
    worst = float(ric.min()) if len(ric) else np.nan
    rollflip = float((np.sign(ric) != major).mean()) if len(ric) else np.nan
    # 缩尾稳健性
    lo, hi = np.nanpercentile(xv, [2.5, 97.5])
    icw = sp(np.clip(xv, lo, hi), yv)
    # 10 箱单调性
    order = np.argsort(xv)
    bins = np.array_split(order, 10)
    bm = np.array([yv[b].mean() for b in bins])
    mono = spearmanr(np.arange(10, dtype=float), bm).statistic
    ac1 = float(pd.Series(xv).autocorr(1))
    # 与 pv / slow 共线 + 双正交残差 IC
    mm = m & pv.notna() & slow.notna()
    cpv = csl = np.nan
    resid_ic = np.nan
    if mm.sum() > 200:
        a = x[mm].values.astype(float)
        p = pv[mm].values.astype(float)
        s2 = slow[mm].values.astype(float)
        cpv = sp(a, p)
        csl = sp(a, s2)
        Az = np.column_stack([np.ones(len(a)), (p - p.mean()) / p.std(), (s2 - s2.mean()) / s2.std()])
        beta, *_ = np.linalg.lstsq(Az, a, rcond=None)
        r = a - Az @ beta
        resid_ic = sp(r, fwd[mm].values.astype(float))
    # expanding-sign 可预知性：每年只用该年之前的数据定方向
    yrs, hit = [], []
    for Y in range(2019, int(close.index[-1].year) + 1):
        mask_hist = (x.index.year < Y) & m
        mask_now = (x.index.year == Y) & m
        if mask_hist.sum() < 400 or mask_now.sum() < 100:
            continue
        sgn = np.sign(sp(x[mask_hist].values.astype(float), fwd[mask_hist].values.astype(float)))
        ic_y = sp(x[mask_now].values.astype(float), fwd[mask_now].values.astype(float))
        if np.isfinite(sgn) and np.isfinite(ic_y) and sgn != 0:
            yrs.append(Y)
            hit.append(np.sign(ic_y) == sgn)
    sign_hit = float(np.mean(hit)) if hit else np.nan
    rows.append(dict(field=f, n=n, start=str(x.dropna().index[0].date()),
                     ic=ic, ico=ico, t_o=t_o, icabs=icabs, same=same, nseg=len(seg_ic),
                     rng=rng, worst=worst, rollflip=rollflip, icw=icw, mono=mono, ac1=ac1,
                     cpv=cpv, csl=csl, resid=resid_ic, sign_hit=sign_hit, ny=len(hit)))
    detail[f] = dict(seg_ic=seg_ic, ric=ric, bm=bm)

T = pd.DataFrame(rows).set_index('field')
T['volproxy'] = T['icabs'] - T['ic'].abs()

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

print('\n' + '=' * 118)
print('【表1】逐字段体检（label 同 pv/slow）')
print('=' * 118)
show = T.copy()
print(f"{'字段':<18}{'起点':<12}{'n':>5}{'IC':>8}{'非重叠IC':>9}{'t':>7}{'IC|y|':>8}"
      f"{'同号':>6}{'极差':>6}{'滚最差':>8}{'翻号率':>7}{'缩尾IC':>8}{'单调':>6}{'AC1':>6}"
      f"{'corr_pv':>8}{'corr_sl':>8}{'残差IC':>8}{'符号命中':>9}")
print('-' * 118)
for f, r in show.sort_values('ic', key=lambda s: s.abs(), ascending=False).iterrows():
    if np.isfinite(r['sign_hit']):
        sh = f"{r['sign_hit']:.0%}({int(r['ny'])})"
    else:
        sh = 'n/a'
    print(f"{f:<18}{r['start']:<12}{int(r['n']):>5}{r['ic']:>+8.3f}{r['ico']:>+9.3f}{r['t_o']:>7.2f}"
          f"{r['icabs']:>+8.3f}{int(r['same']):>3}/{int(r['nseg']):<2}{r['rng']:>6.2f}{r['worst']:>+8.3f}"
          f"{r['rollflip']:>7.0%}{r['icw']:>+8.3f}{r['mono']:>+6.2f}{r['ac1']:>6.2f}"
          f"{r['cpv']:>+8.3f}{r['csl']:>+8.3f}{r['resid']:>+8.3f}{sh:>9}")

# ---------------- 表 2：七项准入 ----------------
print('\n' + '=' * 118)
print('【表2】信号输入口径七项准入（全部满足才可进合成）')
print('    C1 n>=1600(6.3年)  C2 8段同号>=7/8  C3 |IC|>=0.075(Bonf 22测)  C4 非波动代理(volproxy<=0)')
print('    C5 |corr(pv)|,|corr(slow)|<=0.35  C6 单调性|rho|>=0.70  C7 残差IC(正交pv+slow)>=0.5*|IC|')
print('=' * 118)
c1 = T['n'] >= 1600
c2 = T['same'] >= 7
c3 = T['ic'].abs() >= 0.075
c4 = T['volproxy'] <= 0
c5 = (T['cpv'].abs() <= 0.35) & (T['csl'].abs() <= 0.35)
c6 = T['mono'].abs() >= 0.70
c7 = T['resid'].abs() >= 0.5 * T['ic'].abs()
SC = pd.DataFrame({'C1覆盖': c1, 'C2符号': c2, 'C3强度': c3, 'C4非波动': c4,
                   'C5正交': c5, 'C6单调': c6, 'C7残差': c7})
SC['通过数'] = SC.sum(axis=1)
print(f"{'字段':<18}{'C1':>4}{'C2':>4}{'C3':>4}{'C4':>4}{'C5':>4}{'C6':>4}{'C7':>4}{'通过':>6}")
print('-' * 118)
for f in SC.sort_values('通过数', ascending=False).index:
    r = SC.loc[f]
    mark = lambda b: ' O ' if b else ' . '
    print(f"{f:<18}{mark(r['C1覆盖'])}{mark(r['C2符号'])}{mark(r['C3强度'])}{mark(r['C4非波动'])}"
          f"{mark(r['C5正交'])}{mark(r['C6单调'])}{mark(r['C7残差'])}{int(r['通过数']):>4}/7")

full = SC.index[SC['通过数'] == 7].tolist()
print(f'\n>>> 全过 7 项：{full if full else "【无】"}')

# ---------------- 表 3：合成测试 ----------------
print('\n' + '=' * 118)
print('【表3】合成测试：把候选按 rolling 250 日分位等权平均，看 IC 是否大于单因子最优')
print('=' * 118)

# 候选池：按通过数降序取前若干组
def build_comp(fields, use_roll=True, win=250):
    cols = []
    for f in fields:
        x = panel[f]
        sgn = np.sign(T.loc[f, 'ic'])
        if use_roll:
            p = roll_pct(x, win)
            c = p if sgn > 0 else (1 - p)
        else:
            r = x.rank(pct=True)
            c = r if sgn > 0 else (1 - r)
        cols.append(c)
    return pd.concat(cols, axis=1).mean(axis=1)


pools = {}
pools['最优单因子(IC最高)'] = [T['ic'].idxmax()]
if len(full) >= 1:
    pools['全过7项'] = full
pools['IH+IF(同类)'] = ['IF_会员多空比', 'IH_会员多空比']
pools['IV偏斜族(50+300)'] = ['50ETF_IV偏斜', '300ETF_IV偏斜']
pools['IV沽购比族(50+300)'] = ['50ETF_IV沽购比', '300ETF_IV沽购比']
pools['通过数>=5'] = SC.index[SC['通过数'] >= 5].tolist()
pools['全部C2+C3通过(符号稳且够强)'] = T.index[(c2 & c3) & (T['n'] >= 1600)].tolist()
pools['三个独立族各取最优'] = ['50ETF_IV偏斜', '50ETF_PCR成交', 'IC_会员多空比']
pools['IF单独(现否决层字段)'] = ['IF_会员多空比']

print(f"{'组合':<34}{'字段数':>6}{'覆盖起':>12}{'n':>6}{'IC':>8}{'非重叠IC':>9}{'残差IC':>8}")
print('-' * 118)
comp_store = {}
for nm, fl in pools.items():
    fl = [f for f in fl if f in T.index]
    if not fl:
        continue
    c = build_comp(fl)
    m = c.notna() & fwd.notna()
    if m.sum() < 300:
        continue
    ic = sp(c[m].values.astype(float), fwd[m].values.astype(float))
    ico = sp(c[m].values.astype(float)[::H], fwd[m].values.astype(float)[::H])
    mm = m & pv.notna() & slow.notna()
    ri = np.nan
    if mm.sum() > 300:
        a = c[mm].values.astype(float)
        p = pv[mm].values.astype(float)
        s2 = slow[mm].values.astype(float)
        Az = np.column_stack([np.ones(len(a)), (p - p.mean()) / p.std(), (s2 - s2.mean()) / s2.std()])
        beta, *_ = np.linalg.lstsq(Az, a, rcond=None)
        ri = sp(a - Az @ beta, fwd[mm].values.astype(float))
    comp_store[nm] = c
    print(f"{nm:<34}{len(fl):>6}{str(c.dropna().index[0].date()):>12}{int(m.sum()):>6}"
          f"{ic:>+8.3f}{ico:>+9.3f}{ri:>+8.3f}")

# ---------------- 表 4：三因子合成（pv + slow + 情绪）的秩IC ----------------
print('\n' + '=' * 118)
print('【表4】三因子合成的前置诊断：pv/slow 各自 + 加入情绪后 的秩IC（不做回测，只看IC与共线）')
print('=' * 118)
pv_p = roll_pct(pv, 300)
slow_v = ((roll_pct(slow, 300) - 0.30) / 0.35).clip(0, 1)
base2 = 0.5 * pv_p + 0.5 * slow_v

best_pool = max(comp_store.items(), key=lambda kv: (np.isfinite(T.loc[[f for f in pools[kv[0]] if f in T.index], 'ic'].abs().mean()), kv[0])) if comp_store else None

print(f"{'信号':<40}{'n':>6}{'IC':>8}{'非重叠IC':>9}{'残差IC(正交另两者)':>20}")
print('-' * 118)
rows4 = [('pv 分位(250→300日)', pv_p), ('slow 饱和分位', slow_v), ('2因子合成 0.5/0.5', base2)]
for nm, c in comp_store.items():
    rows4.append((f'情绪:{nm}', c))
for nm, c in rows4:
    m = c.notna() & fwd.notna() & pv_p.notna() & slow_v.notna()
    if m.sum() < 300:
        continue
    a = c[m].values.astype(float)
    y = fwd[m].values.astype(float)
    ic = sp(a, y)
    ico = sp(a[::H], y[::H])
    others = [x for x in (pv_p, slow_v, c) if x is not c]
    Az = np.column_stack([np.ones(m.sum())] + [((o[m].values - o[m].mean()) / o[m].std()) for o in others])
    beta, *_ = np.linalg.lstsq(Az, a, rcond=None)
    ri = sp(a - Az @ beta, y)
    print(f"{nm:<40}{int(m.sum()):>6}{ic:>+8.3f}{ico:>+9.3f}{ri:>+20.3f}")

# 三因子 vs 二因子直接比
if comp_store:
    nm0 = list(comp_store.keys())[0]
    for nm, c in comp_store.items():
        m = c.notna() & base2.notna() & fwd.notna()
        if m.sum() < 400:
            continue
        c2 = c[m].values.astype(float)
        b2 = base2[m].values.astype(float)
        y = fwd[m].values.astype(float)
        cp = np.corrcoef(c2, b2)[0, 1]
        tri = (b2 - b2.mean()) / b2.std() + (c2 - c2.mean()) / c2.std()
        print(f"\n  情绪[{nm}] vs 2因子合成: corr={cp:+.3f}  "
              f"2因子IC={sp(b2,y):+.3f}  2因子+情绪等权IC={sp(tri,y):+.3f}  "
              f"(增益 {sp(tri,y)-sp(b2,y):+.3f})")

# ---------------- 表 5：条件 IC（情绪在什么状态下补位） ----------------
print('\n' + '=' * 118)
print('【表5】条件 IC：按 20 日波动率三分位 / 指数趋势，看情绪是否在 pv 失效区补位')
print('=' * 118)
if comp_store:
    for nm, c in list(comp_store.items())[:4]:
        m = c.notna() & fwd.notna() & vol20.notna() & pv_p.notna()
        if m.sum() < 400:
            continue
        vv = vol20[m]
        q = pd.qcut(vv.rank(method='first'), 3, labels=['低波', '中波', '高波'])
        print(f"\n  情绪[{nm}]  n={int(m.sum())}")
        for lab in ['低波', '中波', '高波']:
            sel = m.copy()
            sel[:] = False
            sel.loc[q[q == lab].index] = True
            mm = sel & c.notna() & fwd.notna() & pv_p.notna()
            if mm.sum() < 80:
                continue
            print(f"    {lab}  n={int(mm.sum()):>4}  情绪IC={sp(c[mm].values.astype(float), fwd[mm].values.astype(float)):+.3f}"
                  f"   pvIC={sp(pv_p[mm].values.astype(float), fwd[mm].values.astype(float)):+.3f}"
                  f"   slowIC={sp(slow_v[mm].values.astype(float), fwd[mm].values.astype(float)):+.3f}")

print('\n' + '=' * 118)
print('完成')
