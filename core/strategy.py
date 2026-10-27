# -*- coding: utf-8 -*-
"""M3 策略规则（唯一真身）—— 回测与实盘共用这一份，避免口径分叉。

链路：
    pv 原始信号 ─┐
                 ├→ 各自 300 日滚动分位 → slow 量化饱和 → 等权 0.5:0.5
    slow 原始信号 ┘      → 三档 {0, 0.5, 1.0}（0.30 / 0.80）→ 5 日众数平滑 → base
    base ──(仅当 base == 0.5)→ 用情绪 IF 的 250 日分位替换 → A0
    A0 ─→ 5% 仓位网格 → 10% 死区 → 最终目标仓位

所有函数都是纯函数，唯一带状态的是 `next_target`（需要"当前实际持仓"做死区）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 参数（唯一出处）
W_SIGNAL = 300          # 信号滚动分位窗口（交易日）
MINP_SIGNAL = 250       # 分位最小样本
W_SENT = 250            # 情绪滚动分位窗口
MINP_SENT = 60          # 情绪分位最小样本
K_MODE = 3              # 众数平滑天数（≈ 最少保持 3 日; 2026-10-27 全历史扫描: k=3 捕获率↑ 夏普持平)
BAND_LO, BAND_HI = 0.30, 0.80   # 三档阈值
SAT_LO, SAT_SPAN = 0.30, 0.35   # slow 量化饱和：clip((pct-0.30)/0.35, 0, 1)
W_PV = 0.5              # 合成权重（pv / slow 等权）
GRID = 0.05             # 仓位网格 5%（21 档）
DEADBAND = 0.10         # 死区 10%
SENT_SWITCH = 0.5       # 情绪仅在 base == 此值时接管


# ---------------------------------------------------------------- 基础算子
def roll_pct(s: pd.Series, win: int, minp: int = MINP_SIGNAL) -> pd.Series:
    """滚动分位（严格用"过去" win 个样本，不含当日 → 无前视；保序，不做标准化）。"""
    v = np.asarray(s.values, dtype=float)
    o = np.full(len(v), np.nan)
    mp = min(minp, max(60, win // 3))
    for i in range(len(v)):
        lo = max(0, i - win)
        if i < mp or i - lo < mp:
            continue
        w = v[lo:i]
        w = w[~np.isnan(w)]
        if len(w) < mp or np.isnan(v[i]):
            continue
        o[i] = (w < v[i]).sum() / len(w)
    return pd.Series(o, index=s.index)


def pos_mode(t: pd.Series, k: int = K_MODE) -> pd.Series:
    """最近 k 日众数（消除单日抖动；中位/均值都会改变档位含义，故不用）。"""
    v = np.asarray(t.values, dtype=float)
    o = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        if not len(w):
            continue
        vals, cnt = np.unique(w, return_counts=True)
        o[i] = vals[np.argmax(cnt)]
    return pd.Series(o, index=t.index)


def tgt3(comp: pd.Series, lo: float = BAND_LO, hi: float = BAND_HI) -> pd.Series:
    """三档映射：<0.30 空仓 / [0.30,0.80) 半仓 / >=0.80 满仓。"""
    return pd.Series(np.where(comp < lo, 0.0, np.where(comp < hi, 0.5, 1.0)),
                     index=comp.index).where(comp.notna())


def slow_saturate(pct: pd.Series, lo: float = SAT_LO,
                  span: float = SAT_SPAN) -> pd.Series:
    """slow 分位量化饱和：0.30 以下归零、0.65 以上封顶（修复 5 箱 Q5 为负的缺陷）。"""
    return ((pct - lo) / span).clip(0, 1)


def grid(p, q: float = GRID):
    """仓位量化到 q 网格（默认 5%）。"""
    if q <= 0:
        return p
    return (p / q).round().clip(0, 1 / q) * q


def band(p: pd.Series, bd: float = DEADBAND) -> pd.Series:
    """死区（迟滞）：与"当前实际持仓"偏差不足 bd 就不动。"""
    if bd <= 0:
        return p
    v = np.asarray(p.values, dtype=float)
    out = np.full(len(v), np.nan)
    cur = v[0]
    for i in range(len(v)):
        if np.isnan(v[i]):
            out[i] = cur
            continue
        if abs(v[i] - cur) >= bd - 1e-12:
            cur = v[i]
        out[i] = cur
    return pd.Series(out, index=p.index)


# ---------------------------------------------------------------- 组合链路
def composite(pv_raw: pd.Series, slow_raw: pd.Series,
              win: int = W_SIGNAL) -> pd.DataFrame:
    """pv / slow 原始信号 → 滚动分位 → slow 饱和 → 等权合成。返回 p / s / s_sat / comp。"""
    pv_pct = roll_pct(pv_raw, win)
    sl_pct = roll_pct(slow_raw, win)
    mm = pd.DataFrame({"p": pv_pct, "s": sl_pct}).dropna()
    mm["s_sat"] = slow_saturate(mm["s"])
    mm["comp"] = W_PV * mm["p"] + (1.0 - W_PV) * mm["s_sat"]
    return mm


def build_base(pv_raw: pd.Series, slow_raw: pd.Series,
               win: int = W_SIGNAL, k: int = K_MODE) -> pd.Series:
    """方向层：综合分位 → 三档 → 5 日众数 → base ∈ {0, 0.5, 1}。"""
    mm = composite(pv_raw, slow_raw, win)
    return pos_mode(tgt3(mm["comp"], BAND_LO, BAND_HI), k)


def at_base(a0: pd.Series, base: pd.Series) -> pd.Series:
    """情绪层：仅当 base == 0.5 时用情绪分位替换；0 / 1 档不动。"""
    half = (base - SENT_SWITCH).abs() < 1e-9
    return base.where(~half, a0)


def final_position(a0: pd.Series, q: float = GRID, bd: float = DEADBAND) -> pd.Series:
    """落地：网格 + 死区。"""
    return band(grid(a0, q), bd)


def position_series(pv_raw: pd.Series, slow_raw: pd.Series, if_raw: pd.Series,
                    win: int = W_SIGNAL, win_sent: int = W_SENT,
                    q: float = GRID, bd: float = DEADBAND) -> pd.DataFrame:
    """一次性算出全历史仓位（回测与实盘回填共用）。

    返回列：base / IF分位 / A0 / 目标仓位
    """
    base = build_base(pv_raw, slow_raw, win)
    if_pct = roll_pct(if_raw, win_sent, minp=MINP_SENT).reindex(base.index).ffill()
    a0 = at_base(if_pct, base)
    out = pd.DataFrame({"base": base, "IF分位": if_pct, "A0": a0}, index=base.index).dropna()
    out["目标仓位"] = final_position(out["A0"], q, bd)
    return out


# ---------------------------------------------------------------- 实盘单步
def next_target(a0_today: float, cur_pos: float,
                q: float = GRID, bd: float = DEADBAND) -> float:
    """给定今日 A0 与当前实际持仓，算出今日该调到的目标仓位。"""
    tgt = float(grid(np.array([a0_today]), q)[0])
    if abs(tgt - cur_pos) >= bd - 1e-12:
        return tgt
    return cur_pos
