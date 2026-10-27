# -*- coding: utf-8 -*-
"""
情绪多字段「能不能合成为一个因子」的决定性测试
1) 等权 / IC加权 / 单字段最优  三种合成方式的 IC 对比
2) 贪婪前向选字段：全样本选 vs 只用历史选（隔离「选择偏差」）
3) 每年 OOS：只用 Y 年之前的数据决定「选哪些字段 + 方向」，看 Y 年 IC
label = ln(close_{t+10}/close_t)，信号一律用 250 日滚动分位（因果，无前视）
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _P.ROOT
PANEL = os.path.join(ROOT, 'input', '情绪_面板_日频.csv')
IDX = os.path.join(ROOT, 'input', '中证1000_000852.SH.csv')
OUT = _P.TRAIN
H = 10
WIN = 250


def roll_pct(s, win=WIN, minp=250):
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
    return ser[~ser.index.duplicated(keep='last')]


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


close = load_idx()
panel = pd.read_csv(PANEL, encoding='utf-8')
panel['日期'] = pd.to_datetime(panel['日期'])
panel = panel.set_index('日期').sort_index().reindex(close.index)
fwd = np.log(close.shift(-H) / close)

pv = load_seed_mean('s').reindex(close.index)
slow = load_seed_mean('slow_s').reindex(close.index)

# 候选池：覆盖足够长（能支持 2019+ 检验），其余字段单列出来只做参考
POOL = ['50ETF_IV平值', '50ETF_IV虚沽', '50ETF_IV虚购', '50ETF_IV沽购比', '50ETF_IV偏斜',
        '50ETF_PCR成交', '50ETF_PCR持仓',
        'IF_会员多空比', 'IC_会员多空比', 'IH_会员多空比',
        '300ETF_IV平值', '300ETF_IV虚沽', '300ETF_IV沽购比', '300ETF_IV偏斜',
        '300ETF_PCR成交', '300ETF_PCR持仓']
SHORT = ['500ETF_IV平值', '500ETF_IV虚沽', '500ETF_IV虚购', '500ETF_IV沽购比', '500ETF_IV偏斜',
         '500ETF_PCR成交', '500ETF_PCR持仓', '500ETF_成交额万元', '300ETF_成交额万元', '300ETF_IV虚购']

P = {f: roll_pct(panel[f]) for f in POOL + SHORT}
print('=' * 112)
print('情绪多字段合成能力测试   label=ln(close_{t+10}/close_t)   信号=250日滚动分位（因果）')
print('=' * 112)

# ---------- 1) 三种合成方式 ----------
print('\n【1】合成方式对比（同一字段集合：16 个覆盖 2015+ 的字段）')
print('    方向由「该集合内每个字段的 IC 符号」决定（这里先用全样本符号，见【2】对偏差的检验）')


def sign_map(fields, mask=None):
    if mask is None:
        mask = fwd.notna()
    out = {}
    for f in fields:
        ic = sp(panel[f][mask].values.astype(float), fwd[mask].values.astype(float))
        out[f] = np.sign(ic) if np.isfinite(ic) else 0.0
    return out


def make(fields, smap, w=None):
    cols = []
    ws = []
    for f in fields:
        s = smap[f]
        if s == 0:
            continue
        cols.append(P[f] if s > 0 else (1 - P[f]))
        ws.append(1.0 if w is None else w.get(f, 0.0))
    if not cols:
        return pd.Series(np.nan, index=close.index)
    M = pd.concat(cols, axis=1)
    anyOK = M.notna().any(axis=1)
    if w is None:
        return M.mean(axis=1).where(anyOK)
    wv = np.array(ws, dtype=float)
    num = M.mul(wv, axis=1).sum(axis=1, min_count=1)
    den = M.notna().mul(wv, axis=1).sum(axis=1)
    return (num / den.replace(0, np.nan)).where(anyOK)


smap_all = sign_map(POOL)
comps = {'等权(16字段)': make(POOL, smap_all)}
# IC 加权（用 |IC| 做权）
icmap = {}
for f in POOL:
    icmap[f] = abs(sp(panel[f].values.astype(float), fwd.values.astype(float)))
tot = sum(icmap.values())
comps['IC加权(16字段)'] = make(POOL, smap_all, {f: icmap[f] for f in POOL})
comps['IF单独'] = make(['IF_会员多空比'], smap_all)
comps['IH单独'] = make(['IH_会员多空比'], smap_all)
comps['50ETF_IV偏斜单独'] = make(['50ETF_IV偏斜'], smap_all)
comps['IV族5字段等权'] = make([f for f in POOL if 'IV' in f and f.startswith('50ETF')], smap_all)
comps['期货4字段等权'] = make([f for f in POOL if '会员多空比' in f], smap_all)
comps['PCR4字段等权'] = make([f for f in POOL if 'PCR' in f], smap_all)

print(f"\n{'合成':<24}{'n':>6}{'IC':>8}{'非重叠IC':>9}{'t':>7}{'IC|y|':>8}")
print('-' * 112)
for nm, c in comps.items():
    m = c.notna() & fwd.notna()
    if m.sum() < 300:
        continue
    a = c[m].values.astype(float)
    y = fwd[m].values.astype(float)
    ic = sp(a, y)
    ico = sp(a[::H], y[::H])
    no = len(a[::H])
    t = ico * np.sqrt(no - 2) / np.sqrt(max(1e-12, 1 - ico ** 2))
    print(f"{nm:<24}{int(m.sum()):>6}{ic:>+8.3f}{ico:>+9.3f}{t:>7.2f}{sp(a, np.abs(y)):>+8.3f}")

# ---------- 2) 贪婪前向选择：全样本 vs 只用历史 ----------
print('\n' + '=' * 112)
print('【2】贪婪前向选字段：全样本选（过拟合版）vs 只用历史选（真实版）')
print('    每步加入使合成 IC 最大的字段；方向由训练段 IC 符号决定')
print('=' * 112)

BASE = fwd.notna() & pv.notna() & slow.notna()  # 与后续回测同一区间


def greedy(train_mask, pool, max_k=8):
    chosen, smap = [], {}
    best_path = []
    for step in range(max_k):
        best_f, best_ic, best_s = None, -9, 0
        for f in pool:
            if f in chosen:
                continue
            s = sign_map([f], train_mask).get(f, 0.0)
            if s == 0:
                continue
            trial = chosen + [f]
            sm2 = dict(smap)
            sm2[f] = s
            c = make(trial, sm2)
            m = train_mask & c.notna()
            if m.sum() < 400:
                continue
            ic = sp(c[m].values.astype(float), fwd[m].values.astype(float))
            if np.isfinite(ic) and abs(ic) > best_ic:
                best_ic, best_f, best_s = abs(ic), f, s
        if best_f is None:
            break
        chosen.append(best_f)
        smap[best_f] = best_s
        best_path.append((best_f, best_ic))
    return chosen, smap, best_path


chosen_all, smap_full, path = greedy(BASE, POOL, max_k=8)
print('\n  全样本贪婪路径（每步的合成 |IC|）:')
for i, (f, v) in enumerate(path, 1):
    print(f'    {i}. +{f:<20} 合成|IC| = {v:.3f}')
c_full = make(chosen_all, smap_full)
m = BASE & c_full.notna()
print(f'\n  全样本最优子集（{len(chosen_all)} 字段）区间 IC = '
      f"{sp(c_full[m].values.astype(float), fwd[m].values.astype(float)):+.3f}")

print('\n  逐年「只用历史选字段」的 OOS 检验：')
print(f"  {'年':<6}{'训练期字段数':>10}{'当年选中':<44}{'OOS IC':>9}{'IF单独 IC':>11}{'全样本子集 IC':>13}")
print('  ' + '-' * 108)
yr_rows = []
for Y in range(2019, 2027):
    tr = BASE & (fwd.index.year < Y)
    te = BASE & (fwd.index.year == Y)
    if tr.sum() < 500 or te.sum() < 50:
        continue
    ch, sm, _ = greedy(tr, POOL, max_k=6)
    c = make(ch, sm)
    mm = te & c.notna()
    if mm.sum() < 50:
        continue
    ic_oos = sp(c[mm].values.astype(float), fwd[mm].values.astype(float))
    if 'IF_会员多空比' not in sm:
        sm = dict(sm)
        sm['IF_会员多空比'] = sign_map(['IF_会员多空比'], tr).get('IF_会员多空比', 0.0)
    c_if = make(['IF_会员多空比'], sm)
    ic_if = sp(c_if[mm].values.astype(float), fwd[mm].values.astype(float))
    ic_full = sp(c_full[mm].values.astype(float), fwd[mm].values.astype(float))
    yr_rows.append((Y, ic_oos, ic_if, ic_full))
    print(f"  {Y:<6}{len(ch):>10}{','.join(x[:12] for x in ch):<44}{ic_oos:>+9.3f}{ic_if:>+11.3f}{ic_full:>+13.3f}")

if yr_rows:
    a = np.array([[r[1], r[2], r[3]] for r in yr_rows], dtype=float)
    print('  ' + '-' * 108)
    print(f"  {'均值':<6}{'':>10}{'':<44}{a[:,0].mean():>+9.3f}{a[:,1].mean():>+11.3f}{a[:,2].mean():>+13.3f}")
    print(f"  {'>0 年数':<6}{'':>10}{'':<44}{int((a[:,0]>0).sum()):>9}{int((a[:,1]>0).sum()):>11}{int((a[:,2]>0).sum()):>13}")
    print(f'\n  >>> 全样本选字段的「表面优势」= {a[:,2].mean():+.3f}（把未来信息当已知）')
    print(f'  >>> 只用历史选字段的真实水平   = {a[:,0].mean():+.3f}')
    print(f'  >>> 干脆只用 IF 一个字段       = {a[:,1].mean():+.3f}')

# ---------- 3) 与 pv/slow 的三因子合成诊断 ----------
print('\n' + '=' * 112)
print('【3】三因子合成诊断：把情绪合成当第三条腿，与 pv/slow 等权相加后的秩IC')
print('=' * 112)
pv_p = roll_pct(pv, 300)
slow_v = ((roll_pct(slow, 300) - 0.30) / 0.35).clip(0, 1)
base2 = 0.5 * pv_p + 0.5 * slow_v

print(f"{'第三腿':<24}{'corr(2因子)':>12}{'2因子IC':>10}{'3因子IC':>10}{'增益':>8}{'残差IC':>9}")
print('-' * 112)
m0 = BASE & base2.notna() & fwd.notna()
ic2 = sp(base2[m0].values.astype(float), fwd[m0].values.astype(float))
for nm, c in comps.items():
    mm = m0 & c.notna()
    if mm.sum() < 400:
        continue
    a = c[mm].values.astype(float)
    b = base2[mm].values.astype(float)
    y = fwd[mm].values.astype(float)
    cp = np.corrcoef(a, b)[0, 1]
    tri = (b - b.mean()) / b.std() + (a - a.mean()) / a.std()
    ic3 = sp(tri, y)
    Az = np.column_stack([np.ones(len(a)), (b - b.mean()) / b.std()])
    beta, *_ = np.linalg.lstsq(Az, a, rcond=None)
    ri = sp(a - Az @ beta, y)
    print(f"{nm:<24}{cp:>+12.3f}{sp(b, y):>+10.3f}{ic3:>+10.3f}{ic3-sp(b,y):>+8.3f}{ri:>+9.3f}")

print('\n' + '=' * 112)
print('完成')
