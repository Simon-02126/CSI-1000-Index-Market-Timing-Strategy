# -*- coding: utf-8 -*-
"""Replay 引擎：T+1 开盘成交口径的份额/现金显式记账。

这是**报告与自检统一使用的口径**，理由：`pos.shift(1)×ret` 那种写法（把整根
close-to-close 收益归给"前一日决策的仓位"）在现实中无法执行 —— 隔夜跳空时你
并不在目标仓位上，只有 T+1 开盘价能成交。

口径定义
--------
* 信号：T 日收盘算出目标仓位 `t`。
* 成交：T+`lag` 日**开盘价**（默认 lag=1，即 T+1 开盘）。
* 盯市：当日**收盘价**。
* 调仓：只有当目标仓位变化时才交易；两次调仓之间仓位随价格**自然漂移**，不做
  固定比例的日度再平衡（那会白付"漂移成本"，实测把年化成本从 0.51% 抬到 1.00%）。
* 成本：单边 `one_way` × |成交后仓位 − 成交前仓位|，从现金里扣，当天不产生盈亏。
* 收益：对数收益；年化 = 日均值 × 252。

自检不变量（`self_test`）
    - 满仓常数目标 + 零成本：首笔建仓之后，净值逐日比值必须严格等于收盘价比值；
    - 满仓常数目标 + 3bp：只在建仓当天扣一次费。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def replay_clean(tg, ohlc, start=None, end=None, one_way=3e-4, lag=1,
                 on_change_only=True, init_cash=1.0) -> pd.DataFrame:
    """按目标仓位序列回放，返回逐日账户明细。

    Parameters
    ----------
    tg : pd.Series
        目标仓位（0~1），索引为信号日。
    ohlc : pd.DataFrame
        需含 `open` / `close` 两列，DatetimeIndex。
    start, end : str or Timestamp or None
        回放区间。
    one_way : float
        单边成本率。
    lag : int
        信号日到成交日的交易日间隔（1 = T+1 开盘成交）。
    on_change_only : bool
        True 时只在目标变化时交易（报告口径）；False 则每日拉回目标。

    Returns
    -------
    DataFrame[净收益(对数), 目标仓位, 实际仓位, 成本(绝对额), 净值]
    """
    d = ohlc
    if start is not None:
        d = d.loc[d.index >= pd.Timestamp(start)]
    if end is not None:
        d = d.loc[d.index <= pd.Timestamp(end)]
    d = d.dropna(subset=["open", "close"])
    idx = d.index
    o = d["open"].to_numpy(float)
    c = d["close"].to_numpy(float)
    t = tg.reindex(idx).ffill().to_numpy(float)
    n = len(idx)
    if n == 0:
        return pd.DataFrame(columns=["净收益", "目标仓位", "实际仓位", "成本", "净值"])

    cash, shares, cur_tgt = float(init_cash), 0.0, 0.0
    val = np.zeros(n)
    tgt_s = np.zeros(n)
    pos = np.zeros(n)
    cst = np.zeros(n)
    val[0] = float(init_cash)
    tgt_s[0] = float(t[0]) if not np.isnan(t[0]) else 0.0
    for i in range(n):
        if i > 0:
            j = i - lag
            want = float(t[j]) if (j >= 0 and not np.isnan(t[j])) else cur_tgt
            eq_open = cash + shares * o[i]
            changed = (abs(want - cur_tgt) > 1e-12) or (i == 1)
            if (changed if on_change_only else True) and eq_open > 0:
                p_open = shares * o[i] / eq_open
                fee = abs(want - p_open) * eq_open * one_way
                # 费用从"成交后权益"里出，避免用满额建仓 + 现金转负造成隐含杠杆
                eq_after = eq_open - fee
                new_sh = want * eq_after / o[i]
                cash = eq_after - new_sh * o[i]
                shares = new_sh
                cst[i] = fee
                cur_tgt = want
            else:
                want = cur_tgt
        eq = cash + shares * c[i]
        val[i] = eq
        pos[i] = (shares * c[i] / eq) if eq > 0 else 0.0
        tgt_s[i] = cur_tgt
    lr = np.zeros(n)
    lr[1:] = np.log(val[1:] / val[:-1])
    return pd.DataFrame({"净收益": lr, "目标仓位": tgt_s, "实际仓位": pos,
                         "成本": cst, "净值": val}, index=idx)


def turnover(tg: pd.Series, index: pd.Index) -> pd.Series:
    """单边换手（按目标仓位变化计，单位：仓位比例）。"""
    s = tg.reindex(index).ffill()
    to = s.diff().abs()
    if len(s):
        to.iloc[0] = abs(float(s.iloc[0]))
    return to.fillna(0.0)


def self_test(ohlc: pd.DataFrame, start=None, end=None) -> dict:
    """引擎不变量自检（无需外部期望值）。

    满仓常数目标下，首笔建仓之后 `净值[i]/净值[i-1]` 必须等于 `close[i]/close[i-1]`；
    零成本与 3bp 的差应恰为一次性建仓费。
    """
    d = ohlc
    if start is not None:
        d = d.loc[d.index >= pd.Timestamp(start)]
    if end is not None:
        d = d.loc[d.index <= pd.Timestamp(end)]
    d = d.dropna(subset=["open", "close"])
    tg = pd.Series(1.0, index=d.index)
    r0 = replay_clean(tg, d, one_way=0.0, lag=1, on_change_only=True)
    r1 = replay_clean(tg, d, one_way=3e-4, lag=1, on_change_only=True)
    ref = d["close"].pct_change().to_numpy()
    got0 = r0["净值"].pct_change().to_numpy()
    got1 = r1["净值"].pct_change().to_numpy()
    e0 = float(np.nanmax(np.abs(got0[2:] - ref[2:]))) if len(d) > 2 else 0.0
    e1 = float(np.nanmax(np.abs(got1[2:] - ref[2:]))) if len(d) > 2 else 0.0
    # 3bp 与零成本：末日净值差应为比例 3bp（建仓费在 100% 仓位上一次扣）
    ratio = float(r1["净值"].iloc[-1] / r0["净值"].iloc[-1]) if len(d) else 1.0
    e2 = abs((1.0 - ratio) - 3e-4)
    return dict(ok=bool(e0 < 1e-10 and e1 < 1e-10 and e2 < 1e-9),
                max_err_zero_cost=e0, max_err_3bp=e1, fee_ratio_err=e2,
                n_days=int(len(d)))


if __name__ == "__main__":
    import os
    import sys

    _d = os.path.dirname(os.path.abspath(__file__))
    while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
        _d = os.path.dirname(_d)
    if _d not in sys.path:
        sys.path.insert(0, _d)
    import datafeed as dfd

    oh = dfd.load_ohlc()
    r = self_test(oh, "2019-06-04", "2026-09-30")
    print("引擎自检：", "OK" if r["ok"] else "FAIL")
    for k, v in r.items():
        print(f"  {k:<18}{v}")
