# -*- coding: utf-8 -*-
"""绩效指标（唯一真身）—— 与 archive/reports/4.6_最终版计算与报告/report_final.py 的
perf() 逐字一致，保证实盘账本指标与回测报告可直接对比。

口径：
    成本      单边 3bp，按当日 |Δ仓位| 计提
    收益      对数收益；年化 = 日对数收益均值 × 252
    夏普      年化收益 / 年化波动（无风险利率取 0）
    回撤      净值序列的最深跌幅
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

ANN = 252
BP = 0.0003          # 单边交易成本


def net_returns(pos: pd.Series, ret: pd.Series, bp: float = BP) -> pd.Series:
    """净收益序列：昨日仓位吃今日收益，减去当日换仓成本。

    ⚠️ 旧口径（T 日收盘成交），保留仅供复现历史报告；现行绩效口径是
    `core/replay.py` 的 replay_clean + `perf_from_returns`。
    """
    warnings.warn("net_returns 是旧口径(T 日收盘成交)，现行口径请用 replay.replay_clean + perf_from_returns",
                  DeprecationWarning, stacklevel=2)
    pr = (pos.shift(1) * ret).fillna(0.0)
    to = pos.diff().abs().fillna(0.0)
    return pr - to * bp


def perf_from_returns(net: pd.Series, pos: pd.Series, label: str = "",
                      turn: float | None = None, cost: float | None = None,
                      nchg: int | None = None) -> dict:
    """直接从(已扣费)收益序列算指标——实盘账本用这条，回测与实盘口径一致。

    net  : 每日净收益（对数口径）
    pos  : 每日实际持仓
    turn : 单边换手/年（缺省由 pos 差分推算）；cost 同理
    """
    net = net.fillna(0.0)
    pos = pos.astype(float)
    cq = net.cumsum()
    ann = net.mean() * ANN
    vol = net.std() * np.sqrt(ANN)
    mdd = float((np.exp(cq - cq.cummax()) - 1).min())
    eq = np.exp(cq)
    peak = eq.cummax()
    yr = len(net) / ANN
    to = pos.diff().abs().fillna(pos.abs().iloc[0] if len(pos) else 0.0)
    pos_mean = float(pos.mean())
    return dict(
        label=label,
        cum=float(np.expm1(cq.iloc[-1])),
        ann=float(ann), vol=float(vol),
        sharpe=float(ann / vol) if vol > 0 else np.nan,
        mdd=mdd,
        calmar=float(ann / abs(mdd)) if mdd else np.nan,
        dd_days=int((eq < peak).astype(int).groupby((eq >= peak).cumsum()).cumsum().max()),
        pos=pos_mean, pos_med=float(pos.median()), pos_std=float(pos.std()),
        unit=float(ann / pos_mean) if pos_mean > 0 else np.nan,
        win=float((net > 0).mean()),
        turn=float(turn if turn is not None else to.sum() / yr),
        cost=float(cost if cost is not None else to.sum() / yr * BP),
        nchg=int(nchg if nchg is not None else (to > 1e-9).sum()),
        nlev=int(len(np.unique(np.round(pos.dropna().values, 6)))),
        net=net, eq=eq,
    )


def perf(pos: pd.Series, ret: pd.Series, label: str = "",
         bp: float = BP, years: float | None = None) -> dict:
    """标准绩效字典。`years` 缺省时按 len/252 估算（用于换手年化）。"""
    net = net_returns(pos, ret, bp)
    to = pos.diff().abs().fillna(0.0)
    yr = years if years else len(net) / ANN
    return perf_from_returns(net, pos, label,
                             turn=float(to.sum() / yr), cost=float(to.sum() / yr * bp),
                             nchg=int((to > 1e-9).sum()))


def bh_perf(ret: pd.Series) -> dict:
    """买入持有基准。"""
    cq = ret.cumsum()
    ann = ret.mean() * ANN
    vol = ret.std() * np.sqrt(ANN)
    mdd = float((np.exp(cq - cq.cummax()) - 1).min())
    eq = np.exp(cq)
    peak = eq.cummax()
    return dict(label="买入持有", cum=float(np.expm1(cq.iloc[-1])), ann=float(ann),
                vol=float(vol), sharpe=float(ann / vol) if vol > 0 else np.nan,
                mdd=mdd, calmar=float(ann / abs(mdd)) if mdd else np.nan,
                dd_days=int((eq < peak).astype(int)
                            .groupby((eq >= peak).cumsum()).cumsum().max()),
                pos=1.0, win=float((ret > 0).mean()), net=ret, eq=eq)


def excess_stats(net: pd.Series, ret: pd.Series) -> dict:
    """相对基准的超额（逐日相减）。"""
    ex = net - ret
    sd = ex.std()
    return dict(
        ann_ex=float(net.mean() * ANN - ret.mean() * ANN),
        cum_ex=float(np.expm1(net.sum()) - np.expm1(ret.sum())),
        day_bp=float(ex.mean() * 1e4),
        ir=float(ex.mean() / sd * np.sqrt(ANN)) if sd else np.nan,
        t=float(ex.mean() / (sd / np.sqrt(len(ex)))) if sd else np.nan,
    )


def monthly(pos: pd.Series, ret: pd.Series, bp: float = BP) -> pd.DataFrame:
    """月度汇总：策略 / 指数 月度对数收益。"""
    net = net_returns(pos, ret, bp)
    df = pd.DataFrame({"strat": net, "bm": ret})
    return df.groupby([df.index.year, df.index.month]).sum()


def capture(mo: pd.DataFrame) -> dict:
    """上行/下行捕获（按月度）。"""
    up = mo["bm"] > 0
    cu = mo.loc[up, "strat"].mean() / mo.loc[up, "bm"].mean() * 100 if up.any() else np.nan
    cd = mo.loc[~up, "strat"].mean() / mo.loc[~up, "bm"].mean() * 100 if (~up).any() else np.nan
    return dict(win_rate=float((mo["strat"] > 0).mean()), n_months=len(mo),
                n_up=int(up.sum()), n_dn=int((~up).sum()),
                cap_up=float(cu), cap_dn=float(cd),
                up_strat=float(mo.loc[up, "strat"].mean() * 100),
                up_bm=float(mo.loc[up, "bm"].mean() * 100),
                dn_strat=float(mo.loc[~up, "strat"].mean() * 100),
                dn_bm=float(mo.loc[~up, "bm"].mean() * 100))


def yearly(pos: pd.Series, ret: pd.Series, base: pd.Series | None = None,
           bp: float = BP) -> pd.DataFrame:
    """分年表：指数 / 策略 / 超额 / 策略夏普 / 双回撤 / 日均仓位 / 调仓次数。"""
    net = net_returns(pos, ret, bp)
    rows = []
    for y, g in net.groupby(net.index.year):
        ix = g.index
        r_y, n_y = ret.loc[ix], g
        cq, cqi = n_y.cumsum(), r_y.cumsum()
        vol = n_y.std() * np.sqrt(ANN)
        to = pos.loc[ix].diff().abs().fillna(0.0)
        rows.append(dict(
            年份=y,
            指数=np.expm1(cqi.iloc[-1]),
            策略=np.expm1(cq.iloc[-1]),
            超额=np.expm1(cq.iloc[-1]) - np.expm1(cqi.iloc[-1]),
            夏普=(n_y.mean() * ANN / vol) if vol > 0 else np.nan,
            策略回撤=float((np.exp(cq - cq.cummax()) - 1).min()),
            指数回撤=float((np.exp(cqi - cqi.cummax()) - 1).min()),
            日均仓位=float(pos.loc[ix].mean()),
            调仓次数=int((to > 1e-9).sum()),
        ))
    df = pd.DataFrame(rows).set_index("年份")
    if base is not None:
        nb = net_returns(base, ret, bp)
        df["基准策略"] = [float(np.expm1(nb.loc[nb.index.year == y].sum())) for y in df.index]
        df["叠加贡献"] = (df["策略"] - df["基准策略"]) * 100
    return df


def summarize(pos: pd.Series, ret: pd.Series, base: pd.Series | None = None,
              bp: float = BP) -> dict:
    """一次性产出报告所需的全部指标（策式字典）。"""
    S = perf(pos, ret, "策略", bp)
    BM = bh_perf(ret)
    out = dict(S=S, BM=BM, ex=excess_stats(S["net"], ret))
    out["mo"] = monthly(pos, ret, bp)
    out["cap"] = capture(out["mo"])
    out["yearly"] = yearly(pos, ret, base, bp)
    if base is not None:
        out["base"] = perf(base, ret, "M3基准", bp)
    return out


def fmt_console(out: dict) -> str:
    """把 summarize() 的结果打成人读表格（实盘日报用）。"""
    S, BM = out["S"], out["BM"]
    lines = []
    w = 74
    lines.append("=" * w)
    lines.append(f"{'指标':<20}{'策略':>18}{'买入持有':>18}")
    lines.append("-" * w)
    rows = [("累计收益", "cum", "{:.1%}"), ("年化收益", "ann", "{:.2%}"),
            ("年化波动", "vol", "{:.2%}"), ("夏普比率", "sharpe", "{:.2f}"),
            ("最大回撤", "mdd", "{:.1%}"), ("Calmar", "calmar", "{:.2f}"),
            ("最长回撤持续", "dd_days", "{:.0f}"), ("日胜率", "win", "{:.1%}"),
            ("日均仓位", "pos", "{:.1%}")]
    for lbl, k, f in rows:
        lines.append(f"{lbl:<20}{f.format(S[k]):>18}{f.format(BM[k]):>18}")
    lines.append("-" * w)
    lines.append(f"{'单边换手/年':<20}{S['turn']:>18.2f}{'—':>18}")
    lines.append(f"{'年化成本':<20}{S['cost']:>18.2%}{'—':>18}")
    lines.append(f"{'调仓次数':<20}{S['nchg']:>18d}{'—':>18}")
    ex, cap = out["ex"], out["cap"]
    lines.append("-" * w)
    lines.append(f"超额年化 {ex['ann_ex']:+.2%}   日均 {ex['day_bp']:+.2f}bp   "
                 f"IR {ex['ir']:.2f}   t {ex['t']:.2f}")
    lines.append(f"月度胜率 {cap['win_rate']:.0%}（{cap['n_months']} 个月）   "
                 f"上行捕获 {cap['cap_up']:.0f}%   下行捕获 {cap['cap_dn']:.0f}%")
    lines.append("=" * w)
    return "\n".join(lines)
