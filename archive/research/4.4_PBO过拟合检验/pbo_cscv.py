# -*- coding: utf-8 -*-
"""PBO / CSCV 过拟合检验 —— Bailey, Borwein, López de Prado & Zhu (2014)。

补 README 自认的"唯一空白"：策略参数是看着全样本挑出来的，那它有多大可能是过拟合。

## 口径
1. **候选集**：部署层参数网格（`all_configs()`），每组参数用 `core/strategy.py` 生成目标仓位，
   用 `core/replay.py`（唯一绩效口径：T 日信号 → T+1 开盘成交、只在目标变化时交易、单边 3bp）
   回放出 2019-06-04 ~ 2026-09-30 的日净收益。**没有第二个回测引擎**。
2. **CSCV**：把 T 个交易日切成 S 个连续块；枚举全部 C(S, S/2) 种"半数块做样本内、其余做样本外"
   的划分。每次划分里在样本内挑夏普最高的那一组，看它在样本外的排名 ω ∈ (0,1)，取 λ = logit(ω)。
   **PBO = P(λ ≤ 0)**，即"样本内最优"落到样本外中位数以下的频率。
3. **伴随指标**：退化斜率（OOS 夏普 ~ IS 夏普 的 OLS 斜率）、选中组合的亏损概率、
   出厂配置（先验给定、不参与挑选）在各划分里的样本外排名、DSR（Deflated Sharpe Ratio）。
4. **PBO 随候选集扩大而上升**：从 12 组逐级扩到 1152 组，看结论稳不稳。

## 用法
    python archive\\research\\4.6_PBO过拟合检验\\pbo_cscv.py           # 全量
    python archive\\research\\4.6_PBO过拟合检验\\pbo_cscv.py --quick   # 只跑出厂网格

产出（本目录）：`results_pbo.json`、`configs_stats.csv`、`returns_matrix.npz`
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 路径 / 导入
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import _paths as _P  # noqa: E402

sys.path.insert(0, _P.CORE)
import datafeed as dfd          # noqa: E402
import replay as rp             # noqa: E402
import strategy as st           # noqa: E402

START, END = "2019-06-04", "2026-09-30"
ONE_WAY = 3e-4
ANN = 252
EULER = 0.5772156649015329

# ---------------------------------------------------------------- 候选集定义
WINS = [200, 250, 300, 400]
BANDS = [(0.25, 0.75), (0.30, 0.70), (0.30, 0.80), (0.35, 0.85)]
KS = [1, 3, 5]
SATS = [True, False]
W_PVS = [0.30, 0.50, 0.70]
SENTS = [True, False]
GRIDS = [(0.05, 0.10), (0.10, 0.15)]      # (仓位网格, 死区)

SHIPPED = dict(win=300, band=(0.30, 0.80), k=3, sat=True, w_pv=0.50,
               sent=True, grid=(0.05, 0.10), src="main")

# 结构性候选：archive/signals/_train_out 里真实跑过、后被否决的量价口径
STRUCT_SRCS = ["main", "h10", "h20", "d5_10_ic", "d5_10_mse", "d5_10_20_ic"]


def load_variant(tag: str) -> pd.Series | None:
    """读 archive/signals/_train_out/{tag}_{YYYY}.csv（逐年、64 维→行均值）。

    只认 `{tag}_YYYY.csv`，避免 `d5_10_mse` 误吃 `d5_10_mse_s0_2019.csv`。
    """
    pat = re.compile(rf"^{re.escape(tag)}_(\d{{4}})\.csv$")
    parts = []
    for p in sorted(glob.glob(os.path.join(_P.TRAIN, "*.csv"))):
        if not pat.match(os.path.basename(p)):
            continue
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        if not cols:
            continue
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None


def cfg_key(cfg):
    return "{}_win{}_b{:.2f}-{:.2f}_k{}_sat{}_wpv{:.2f}_sent{}_{:.2f}/{:.2f}".format(
        cfg.get("src", "main"),
        cfg["win"], cfg["band"][0], cfg["band"][1], cfg["k"],
        int(cfg["sat"]), cfg["w_pv"], int(cfg["sent"]),
        cfg["grid"][0], cfg["grid"][1])


def all_configs():
    out = []
    for win, band, k, sat, w_pv, sent, gd in itertools.product(
            WINS, BANDS, KS, SATS, W_PVS, SENTS, GRIDS):
        out.append(dict(win=win, band=band, k=k, sat=sat, w_pv=w_pv,
                        sent=sent, grid=gd, src="main"))
    return out


def struct_configs():
    """跨量价口径 × 方向层参数：更接近"当年真实搜索过的空间"。"""
    out = []
    for src, band, k, sat, sent in itertools.product(
            STRUCT_SRCS, BANDS, KS, SATS, SENTS):
        out.append(dict(win=300, band=band, k=k, sat=sat, w_pv=0.50,
                        sent=sent, grid=(0.05, 0.10), src=src))
    return out


# ---------------------------------------------------------------- 快算子
def fast_mode(t: pd.Series, k: int) -> pd.Series:
    """等价于 strategy.pos_mode（值为 0/0.5/1），用累计计数写，快 ~100 倍。"""
    v = np.asarray(t.values, dtype=float)
    n = len(v)
    ok = np.isin(v, [0.0, 0.5, 1.0]) | np.isnan(v)
    if not np.all(ok):
        raise ValueError("pos_mode 输入出现非 {0,0.5,1} 的值")
    c0 = np.concatenate([[0], np.cumsum(v == 0.0)])
    c5 = np.concatenate([[0], np.cumsum(v == 0.5)])
    c1 = np.concatenate([[0], np.cumsum(v == 1.0)])
    out = np.full(n, np.nan)
    for i in range(n):
        lo = i - k + 1
        if lo < 0:
            lo = 0
        a0, a5, a1 = c0[i + 1] - c0[lo], c5[i + 1] - c5[lo], c1[i + 1] - c1[lo]
        if a0 + a5 + a1 == 0:
            continue
        out[i] = 0.0 if a0 >= a5 and a0 >= a1 else (0.5 if a5 >= a1 else 1.0)
    return pd.Series(out, index=t.index)


# ---------------------------------------------------------------- 造矩阵
def build_matrix(configs, verbose=True):
    """返回 (dates, R[T×N], stats_df)。R 是日净收益（3bp 单边），全列同一日期轴。"""
    t0 = time.time()
    ohlc = dfd.load_ohlc()
    raw = dfd.build_raw_signals()
    if_raw = dfd.load_if_ratio().dropna()
    srcs = sorted({c.get("src", "main") for c in configs})
    pv_map = {"main": raw["pv"]}
    for tag in srcs:
        if tag == "main":
            continue
        s = load_variant(tag)
        if s is None:
            raise FileNotFoundError(f"找不到 {tag}_YYYY.csv")
        pv_map[tag] = s
    if verbose:
        print(f"[data] pv/slow {len(raw)} 行 {raw.index.min().date()}~{raw.index.max().date()}"
              f" | IF {len(if_raw)} 行 止于 {if_raw.index.max().date()}"
              f" | 口径 {srcs}")

    # 1) 滚动分位只依赖 (口径, window)
    pct = {}
    for tag in srcs:
        for w in sorted({c["win"] for c in configs}):
            pct[(tag, w)] = (st.roll_pct(pv_map[tag], w), st.roll_pct(raw["slow"], w))
    if_pct = st.roll_pct(if_raw, st.W_SENT, minp=st.MINP_SENT)

    # 2) base：依赖 (src, win, band, k, sat, w_pv)
    base_cache = {}
    for key in sorted({(c.get("src", "main"), c["win"], c["band"], c["k"],
                        c["sat"], c["w_pv"]) for c in configs}):
        src, win, band, k, sat, w_pv = key
        p, s = pct[(src, win)]
        mm = pd.DataFrame({"p": p, "s": s}).dropna()
        ss = st.slow_saturate(mm["s"]) if sat else mm["s"]
        comp = w_pv * mm["p"] + (1.0 - w_pv) * ss
        base_cache[key] = fast_mode(st.tgt3(comp, band[0], band[1]), k)

    # 3) A0：+ 情绪层
    a0_cache = {}
    for key, base in base_cache.items():
        src, win, band, k, sat, w_pv = key
        for sent in (True, False):
            if sent:
                ifp = if_pct.reindex(base.index).ffill()
                a0 = st.at_base(ifp, base)
                out = pd.DataFrame({"base": base, "if": ifp, "a0": a0},
                                   index=base.index).dropna()
            else:
                out = pd.DataFrame({"base": base}, index=base.index).dropna()
                out["a0"] = out["base"]
            a0_cache[key + (sent,)] = out["a0"]

    # 4) 落地 + 回放
    rets, stats = [], []
    idx = None
    for i, cfg in enumerate(configs):
        a0 = a0_cache[(cfg.get("src", "main"), cfg["win"], cfg["band"], cfg["k"],
                       cfg["sat"], cfg["w_pv"], cfg["sent"])]
        pos = st.band(st.grid(a0, cfg["grid"][0]), cfg["grid"][1])
        rep = rp.replay_clean(pos, ohlc, start=START, end=END, one_way=ONE_WAY)
        if idx is None:
            idx = rep.index
        rets.append(rep["净收益"].values)
        nav = rep["净值"].values
        sd = rep["净收益"].std(ddof=1)
        n = len(nav)
        stats.append(dict(
            key=cfg_key(cfg), src=cfg.get("src", "main"),
            win=cfg["win"], band=f"{cfg['band'][0]:.2f}-{cfg['band'][1]:.2f}",
            k=cfg["k"], sat=int(cfg["sat"]), w_pv=cfg["w_pv"], sent=int(cfg["sent"]),
            grid=f"{cfg['grid'][0]:.2f}/{cfg['grid'][1]:.2f}",
            sharpe=round(float(rep["净收益"].mean() / sd * np.sqrt(ANN)), 4) if sd > 0 else np.nan,
            cagr=round(float(nav[-1] ** (ANN / (n - 1)) - 1), 4),
            mdd=round(float((nav / np.maximum.accumulate(nav) - 1).min()), 4),
            avg_pos=round(float(rep["实际仓位"].mean()), 4),
            nchg=int((rep["目标仓位"].diff().abs() > 1e-12).sum()),
        ))
        if verbose and (i + 1) % 300 == 0:
            print(f"  ... {i+1}/{len(configs)}  {time.time()-t0:.0f}s")

    R = np.column_stack(rets)
    assert R.shape[0] == len(idx) and not np.isnan(R).any(), R.shape
    if verbose:
        print(f"[matrix] {R.shape[0]} 交易日 × {R.shape[1]} 组，用时 {time.time()-t0:.0f}s")
    return idx, R, pd.DataFrame(stats)


# ---------------------------------------------------------------- CSCV
def _crit(sum_, sumsq, cnt):
    mu = sum_ / cnt
    var = sumsq / cnt - mu ** 2
    return mu / np.sqrt(np.maximum(var, 1e-18))


def _block_stats(R, S):
    T = R.shape[0]
    edges = np.linspace(0, T, S + 1).astype(int)
    cnt = np.array([edges[i + 1] - edges[i] for i in range(S)], dtype=float)
    bs = np.stack([R[edges[i]:edges[i + 1]].sum(0) for i in range(S)])
    bq = np.stack([(R[edges[i]:edges[i + 1]] ** 2).sum(0) for i in range(S)])
    combos = np.array(list(itertools.combinations(range(S), S // 2)), dtype=np.int16)
    return cnt, bs, bq, combos


def cscv(R, S=16, metric="sharpe", chunk=1500, keep_ranks=None):
    """组合对称交叉验证。返回每次划分的 λ、IS/OOS 夏普、选中列号、指定列的 OOS 相对排名。"""
    T, N = R.shape
    cnt, bs, bq, combos = _block_stats(R, S)
    tot_s, tot_q, tot_n = bs.sum(0), bq.sum(0), cnt.sum()
    nc = len(combos)
    lam = np.empty(nc)
    is_sel = np.empty(nc)
    os_sel = np.empty(nc)
    is_med = np.empty(nc)
    os_med = np.empty(nc)
    sel = np.empty(nc, dtype=np.int32)
    ranks = {j: np.empty(nc) for j in (keep_ranks or [])}

    for a in range(0, nc, chunk):
        c = combos[a:a + chunk]
        m = np.zeros((len(c), S))
        np.put_along_axis(m, c, 1.0, axis=1)
        isn, iss, isq = m @ cnt, m @ bs, m @ bq
        osn, oss, osq = tot_n - isn, tot_s - iss, tot_q - isq
        if metric == "sharpe":
            isc = _crit(iss, isq, isn[:, None])
            osc = _crit(oss, osq, osn[:, None])
        else:                                    # 年化收益
            isc = iss / isn[:, None] * ANN
            osc = oss / osn[:, None] * ANN
        best = isc.argmax(1)
        rows = np.arange(len(c))
        picked = osc[rows, best]
        rank = ((osc < picked[:, None]).sum(1)
                + 0.5 * ((osc == picked[:, None]).sum(1) - 1) + 1)
        w = rank / (N + 1)
        lam[a:a + chunk] = np.log(w / (1 - w))
        is_sel[a:a + chunk] = isc[rows, best]
        os_sel[a:a + chunk] = picked
        is_med[a:a + chunk] = np.median(isc, axis=1)
        os_med[a:a + chunk] = np.median(osc, axis=1)
        sel[a:a + chunk] = best
        for j in ranks:
            v = osc[:, j]
            r2 = ((osc < v[:, None]).sum(1)
                  + 0.5 * ((osc == v[:, None]).sum(1) - 1) + 1)
            ranks[j][a:a + chunk] = r2 / (N + 1)
    return dict(lam=lam, is_sel=is_sel, os_sel=os_sel, sel=sel,
                is_med=is_med, os_med=os_med,
                ranks=ranks, S=S, N=N, nc=nc)


def summarize(res, label):
    lam = res["lam"]
    ie = res["is_sel"] - res["is_med"]
    oe = res["os_sel"] - res["os_med"]
    s = dict(label=label, N=res["N"], S=res["S"], n_comb=res["nc"],
             pbo=round(float((lam <= 0).mean()), 4),
             lam_median=round(float(np.median(lam)), 4),
             lam_p10=round(float(np.percentile(lam, 10)), 4),
             lam_p90=round(float(np.percentile(lam, 90)), 4),
             is_sharpe_mean=round(float(res["is_sel"].mean()), 4),
             oos_sharpe_mean=round(float(res["os_sel"].mean()), 4),
             degrade_slope=round(float(np.polyfit(res["is_sel"], res["os_sel"], 1)[0]), 4),
             degrade_slope_excess=round(float(np.polyfit(ie, oe, 1)[0]), 4),
             is_excess_mean=round(float(ie.mean()), 4),
             oos_excess_mean=round(float(oe.mean()), 4),
             p_oos_loss=round(float((res["os_sel"] <= 0).mean()), 4))
    for j, rk in res["ranks"].items():
        s["fixed_oos_rank_median"] = round(float(np.median(rk)), 4)
        s["fixed_below_median_pct"] = round(float((rk <= 0.5).mean()), 4)
    return s


# ---------------------------------------------------------------- DSR
def deflated_sharpe(rets, n_trials, sr_var):
    from scipy.stats import norm
    r = np.asarray(rets, float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1)                      # 逐日
    g3 = float(pd.Series(r).skew())
    g4 = float(pd.Series(r).kurt()) + 3.0              # 非超额峰度
    sr0 = np.sqrt(sr_var) * ((1 - EULER) * norm.ppf(1 - 1.0 / n_trials)
                             + EULER * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    return dict(sr_daily=round(float(sr), 5), sr_ann=round(float(sr * np.sqrt(ANN)), 4),
                sr0_ann=round(float(sr0 * np.sqrt(ANN)), 4),
                dsr=round(float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den)), 4),
                skew=round(g3, 4), kurt=round(g4, 4))


# ---------------------------------------------------------------- 主流程
LEVELS = [
    ("L1 方向层 band×k", dict(win=[300], w_pv=[0.5], sat=[True], sent=[True], grid=[(0.05, 0.10)])),
    ("L2 +饱和开关", dict(win=[300], w_pv=[0.5], sat=[True, False], sent=[True], grid=[(0.05, 0.10)])),
    ("L3 +情绪开关", dict(win=[300], w_pv=[0.5], sat=[True, False], sent=[True, False], grid=[(0.05, 0.10)])),
    ("L4 +合成权重", dict(win=[300], w_pv=[0.3, 0.5, 0.7], sat=[True, False], sent=[True, False], grid=[(0.05, 0.10)])),
    ("L5 +分位窗口", dict(win=[200, 250, 300, 400], w_pv=[0.3, 0.5, 0.7], sat=[True, False], sent=[True, False], grid=[(0.05, 0.10)])),
    ("L6 +网格/死区", dict(win=[200, 250, 300, 400], w_pv=[0.3, 0.5, 0.7], sat=[True, False], sent=[True, False], grid=[(0.05, 0.10), (0.10, 0.15)])),
]


def pick_cols(space, fixed):
    return [i for i, c in enumerate(space) if all(c[k] in v for k, v in fixed.items())]


def dedupe(R):
    keep, seen = [], set()
    for j in range(R.shape[1]):
        h = R[:, j].astype(np.float32).tobytes()
        if h not in seen:
            seen.add(h)
            keep.append(j)
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t_all = time.time()
    os.makedirs(_HERE, exist_ok=True)

    space = all_configs()
    print(f"候选配置 {len(space)} 组（去重前）")
    dates, R, stats = build_matrix(space)
    np.savez_compressed(os.path.join(_HERE, "returns_matrix.npz"),
                        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
                        R=R.astype(np.float32), keys=stats["key"].values)
    stats.to_csv(os.path.join(_HERE, "configs_stats.csv"), index=False, encoding="utf-8-sig")

    ship_key = cfg_key(SHIPPED)
    ship_idx = int(stats.index[stats["key"] == ship_key][0])
    print(f"出厂配置：{ship_key}  夏普 {stats.loc[ship_idx, 'sharpe']}")

    results = dict(meta=dict(start=START, end=END, one_way=ONE_WAY,
                             n_days=int(R.shape[0]), n_configs=int(R.shape[1]),
                             shipped=ship_key,
                             shipped_sharpe=float(stats.loc[ship_idx, "sharpe"]),
                             generated=time.strftime("%Y-%m-%d %H:%M:%S")),
                   levels=[], robustness=[], dsr={}, shipped_in_grid={})

    # ---- 各层级 PBO（候选集逐级扩大） ----
    for name, fixed in LEVELS:
        cols = pick_cols(space, fixed)
        keep = dedupe(R[:, cols])
        cols2 = [cols[k] for k in keep]
        Rl = R[:, cols2]
        j = cols2.index(ship_idx) if ship_idx in cols2 else None
        Slist = [16] if True else []
        if name.startswith("L6"):
            Slist = [10, 16, 20]
        for S in Slist:
            t0 = time.time()
            res = cscv(Rl, S=S, keep_ranks=[j] if j is not None else None)
            s = summarize(res, f"{name} | S={S}")
            s["n_raw"] = len(cols)
            s["secs"] = round(time.time() - t0, 1)
            results["levels"].append(s)
            print(f"  {s['label']:<24} N={s['N']:<5} PBO={s['pbo']:.3f} "
                  f"λ中位={s['lam_median']:+.2f} 斜率={s['degrade_slope']:+.2f} "
                  f"亏损率={s['p_oos_loss']:.3f} ({s['secs']}s)")

    # ---- 稳健性：判据换成"年化收益"（只在全网格上跑） ----
    cols = pick_cols(space, LEVELS[-1][1])
    keep = dedupe(R[:, cols])
    cols2 = [cols[k] for k in keep]
    Rf = R[:, cols2]
    jf = cols2.index(ship_idx) if ship_idx in cols2 else None
    res = cscv(Rf, S=16, metric="ret")
    s = summarize(res, "L6 全网格 | S=16 | 判据=年化收益")
    results["robustness"].append(s)
    print(f"  {s['label']:<24} PBO={s['pbo']:.3f}")

    # ---- DSR ----
    r_ship = R[:, ship_idx]
    sr_var = float(np.var([R[:, k].mean() / R[:, k].std(ddof=1)
                           for k in range(R.shape[1])], ddof=1))
    results["dsr"] = deflated_sharpe(r_ship, Rf.shape[1], sr_var)
    d = results["dsr"]
    print(f"  DSR: SR年化={d['sr_ann']} SR0={d['sr0_ann']} DSR={d['dsr']}")

    # ---- 出厂配置在全网格里的名次 ----
    ordered = stats.sort_values("sharpe", ascending=False).reset_index(drop=True)
    rank_ship = int(ordered.index[ordered["key"] == ship_key][0]) + 1
    q = ordered["sharpe"].quantile([.1, .5, .9])
    results["shipped_in_grid"] = dict(
        rank=rank_ship, n=int(len(ordered)),
        sharpe=float(ordered.iloc[rank_ship - 1]["sharpe"]),
        top1_key=str(ordered.iloc[0]["key"]),
        top1_sharpe=float(ordered.iloc[0]["sharpe"]),
        sharpe_p10=round(float(q[.1]), 4), sharpe_p50=round(float(q[.5]), 4),
        sharpe_p90=round(float(q[.9]), 4),
    )
    print(f"  出厂配置在 {len(ordered)} 组里夏普排第 {rank_ship}；"
          f"最好的是 {results['shipped_in_grid']['top1_key']} "
          f"({results['shipped_in_grid']['top1_sharpe']})")

    # ---- 结构性候选集：跨量价口径 ----
    print("\n[结构性候选集] 跨量价口径 × 方向层参数")
    scfgs = struct_configs()
    _, Rs, stats_s = build_matrix(scfgs, verbose=False)
    np.savez_compressed(os.path.join(_HERE, "returns_matrix_struct.npz"),
                        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
                        R=Rs.astype(np.float32), keys=stats_s["key"].values)
    stats_s.to_csv(os.path.join(_HERE, "configs_stats_struct.csv"), index=False,
                   encoding="utf-8-sig")
    keep = dedupe(Rs)
    Rs2 = Rs[:, keep]
    ship_idx_s = int(stats_s.index[stats_s["key"] == ship_key][0])
    j = list(keep).index(ship_idx_s) if ship_idx_s in list(keep) else None
    results["struct"] = dict(n_raw=int(len(scfgs)), n_unique=int(Rs2.shape[1]), levels=[])
    for S in (10, 16, 20):
        res = cscv(Rs2, S=S, keep_ranks=[j] if j is not None else None)
        s = summarize(res, f"结构性候选集 | S={S}")
        results["struct"]["levels"].append(s)
        print(f"  {s['label']:<22} N={s['N']:<5} PBO={s['pbo']:.3f} "
              f"λ中位={s['lam_median']:+.2f} 斜率={s['degrade_slope']:+.2f} "
              f"亏损率={s['p_oos_loss']:.3f}")
    o = stats_s.sort_values("sharpe", ascending=False).reset_index(drop=True)
    results["struct"]["shipped_rank"] = int(o.index[o["key"] == ship_key][0]) + 1
    results["struct"]["by_src"] = {
        k: dict(n=int(v["sharpe"].size), sharpe_mean=round(float(v["sharpe"].mean()), 4),
                sharpe_max=round(float(v["sharpe"].max()), 4))
        for k, v in stats_s.groupby("src")}
    for k, v in results["struct"]["by_src"].items():
        print(f"    {k:<12} n={v['n']:<3} 夏普 均值={v['sharpe_mean']:+.3f} 最高={v['sharpe_max']:+.3f}")

    with open(os.path.join(_HERE, "results_pbo.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"完成，总用时 {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
