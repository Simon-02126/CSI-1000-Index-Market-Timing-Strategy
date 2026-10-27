# -*- coding: utf-8 -*-
"""core 自检：用通用模块重算最终版绩效，结果必须与报告一致。

    python core/selftest.py

校验项：区间、样本数、核心绩效、分年收益、交易级结构、M3 基准。
任何一项偏离即报错退出（exit 1）。

口径与 `05_报告与图表/中证1000择时策略说明_最终版.md` 完全一致：
    - 引擎：`core/replay.py`（T 日收盘出信号 → **T+1 开盘价成交** → 收盘价盯市）
    - 平滑：`K_MODE = 3`（3 日众数）
    - 成本：单边 3bp，按真实换手计提
    - 区间：2019-06-04 ~ 2026-09-30，1780 个交易日 / 7.06 年
    - 目标值：累计 328.3%、年化 20.59%、夏普 1.19、回撤 -25.1%、Calmar 0.82、
             日均仓位 49.8%、单边换手 17.09、年化成本 0.51%、调仓 444 次
"""
from __future__ import annotations

import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths as _P  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import datafeed  # noqa: E402
import metrics  # noqa: E402
import replay  # noqa: E402
import strategy  # noqa: E402

START, END = "2019-06-04", "2026-09-30"
BP = 3e-4                                   # 单边成本

EXPECT = {
    "n": 1780,
    "cum": 4.283, "ann": 0.2059, "sharpe": 1.19, "mdd": -0.251,
    "calmar": 0.82, "pos": 0.498, "turn": 17.09, "cost": 0.0051, "nchg": 444,
}
EXPECT_YEAR = {2019: 0.097, 2020: 0.361, 2021: 0.060, 2022: 0.074,
               2023: 0.055, 2024: 0.611, 2025: 0.300, 2026: 0.141}
EXPECT_BASE = {"cum": 2.182, "sharpe": 0.71}      # M3 基准（不加情绪层）
EXPECT_BH = {"cum": 0.389, "sharpe": 0.19}        # 买入持有（含窗口首日的前一日收益口径）


def stats(rep: pd.DataFrame, years: float) -> dict:
    """从 replay 输出算绩效。成本口径 = 真实换手 × 单边成本（不是名义费用累加）。"""
    net = rep["净收益"]
    eq = rep["净值"]
    pos = rep["目标仓位"]
    to = rep["目标仓位"].diff().abs().fillna(rep["目标仓位"].abs())
    ann = float(net.mean() * replay.ANN)
    vol = float(net.std(ddof=1) * np.sqrt(replay.ANN))
    dd = float((eq / eq.cummax() - 1.0).min())
    return dict(
        cum=float(eq.iloc[-1]), ann=ann, vol=vol,
        sharpe=ann / vol if vol > 0 else 0.0, mdd=dd,
        calmar=ann / abs(dd) if dd < 0 else 0.0,
        pos=float(pos.mean()), turn=float(to.sum() / years),
        cost=float(to.sum() * BP / years),
        nchg=int((to > 1e-12).sum()),
    )


def main() -> int:
    print("=" * 88)
    print("core 自检：通用模块 vs 最终版报告（replay 引擎 / K_MODE=%d）" % strategy.K_MODE)
    print("=" * 88)

    data = datafeed.load_all()
    pos = strategy.position_series(data["raw"]["pv"], data["raw"]["slow"], data["if_raw"])
    pos = pos[(pos.index >= pd.Timestamp(START)) & (pos.index <= pd.Timestamp(END))]

    ohlc = datafeed.load_ohlc().loc[pos.index[0]:pos.index[-1]]
    ohlc = ohlc[["open", "high", "low", "close"]].dropna()

    # 引擎不变量自检（满仓常数目标下，建仓后净值日变化 == 收盘价日变化）
    st = replay.self_test(ohlc)
    print(f"引擎自检 {'OK' if st['ok'] else 'FAIL'}"
          f"  残差 零成本 {st['max_err_zero_cost']:.2e} / 3bp {st['max_err_3bp']:.2e}"
          f"  样本 {st['n_days']} 日")
    if not st["ok"]:
        print("自检失败：replay 引擎不变量被破坏")
        return 1

    years = len(pos) / replay.ANN
    rep = replay.replay_clean(pos["目标仓位"], ohlc, one_way=BP)
    rbase = replay.replay_clean(pos["base"], ohlc, one_way=BP)
    S, B = stats(rep, years), stats(rbase, years)

    print(f"区间 {ohlc.index[0].date()} ~ {ohlc.index[-1].date()}，{len(rep)} 个交易日 / {years:.2f} 年")
    print(f"仓位档位 {rep['目标仓位'].round(4).nunique()} 档，调仓 {S['nchg']} 次")

    fails = []

    def chk(name, got, want, rel=0.01, abs_tol=None):
        if abs_tol is not None:
            ok = abs(got - want) <= abs_tol
        else:
            ok = abs(got - want) <= abs(want) * rel + 1e-9
        print(f"  {name:<12}{got:>16.4f}{want:>16.4f}{'OK' if ok else 'MISMATCH':>10}")
        if not ok:
            fails.append(name)

    print("-" * 88)
    print(f"  {'指标':<12}{'重算':>16}{'报告':>16}{'':>10}")
    chk("样本数", len(rep), EXPECT["n"], abs_tol=0)
    chk("累计收益", S["cum"], EXPECT["cum"])
    chk("年化收益", S["ann"], EXPECT["ann"])
    chk("夏普比率", S["sharpe"], EXPECT["sharpe"], abs_tol=0.01)
    chk("最大回撤", S["mdd"], EXPECT["mdd"], abs_tol=0.005)
    chk("Calmar", S["calmar"], EXPECT["calmar"], abs_tol=0.02)
    chk("日均仓位", S["pos"], EXPECT["pos"], abs_tol=0.005)
    chk("单边换手", S["turn"], EXPECT["turn"], abs_tol=0.1)
    chk("年化成本", S["cost"], EXPECT["cost"], abs_tol=0.0005)
    chk("调仓次数", S["nchg"], EXPECT["nchg"], abs_tol=0)

    print("-" * 88)
    # 买入持有口径与报告一致：用全历史差分后按窗口取子集（首日含前一日收益）
    ret_bm = datafeed.log_returns(data["close"]).reindex(ohlc.index).fillna(0.0)
    print(f"  {'年份':<10}{'策略':>11}{'报告':>11}{'指数':>11}{'夏普':>9}{'日均仓位':>11}")
    for y in sorted(set(rep.index.year)):
        m = rep.index.year == y
        s_y = rep.loc[m, "净值"]
        got = float(s_y.iloc[-1] / s_y.iloc[0] * (1.0 + rep.loc[m, "净收益"].iloc[0])
                    - 1.0) if m.any() else 0.0
        # 年度收益 = 年内净值变化（首日净值已含当日起点）
        got = float(s_y.iloc[-1] / (s_y.iloc[0] / np.exp(rep.loc[m, "净收益"].iloc[0])) - 1.0)
        want = EXPECT_YEAR.get(int(y))
        ok = want is None or abs(got - want) <= abs(want) * 0.02 + 1e-9
        ann_y = float(rep.loc[m, "净收益"].mean() * replay.ANN)
        vol_y = float(rep.loc[m, "净收益"].std(ddof=1) * np.sqrt(replay.ANN))
        shp = ann_y / vol_y if vol_y > 0 else 0.0
        bm_y = float(np.exp(ret_bm[m].sum()) - 1.0)
        print(f"  {y:<10}{got:>10.1%}{('—' if want is None else format(want, '.1%')):>11}"
              f"{bm_y:>10.1%}{shp:>9.2f}{rep.loc[m, '目标仓位'].mean():>10.1%}")
        if not ok:
            fails.append(f"{y}年策略")

    print("-" * 88)
    print(f"  M3 基准（不叠加情绪）：累计 {B['cum'] - 1:+.1%}  年化 {B['ann']:.2%}  "
          f"夏普 {B['sharpe']:.2f}  回撤 {B['mdd']:.1%}  调仓 {B['nchg']} 次")
    # 买入持有：与报告同一口径（metrics.bh_perf，含窗口首日的前一日收益）
    BH = metrics.bh_perf(ret_bm)
    BH = dict(cum=BH["cum"], ann=BH["ann"], sharpe=BH["sharpe"], mdd=BH["mdd"])
    print(f"  买入持有：            累计 {BH['cum']:+.1%}  年化 {BH['ann']:.2%}  "
          f"夏普 {BH['sharpe']:.2f}  回撤 {BH['mdd']:.1%}")
    chk("M3累计", B["cum"], EXPECT_BASE["cum"])
    chk("M3夏普", B["sharpe"], EXPECT_BASE["sharpe"], abs_tol=0.01)
    chk("买入持有累计", BH["cum"], EXPECT_BH["cum"])
    chk("买入持有夏普", BH["sharpe"], EXPECT_BH["sharpe"], abs_tol=0.01)
    print("=" * 88)

    if fails:
        print(f"自检失败：{fails}")
        return 1
    print("自检通过：core 模块与报告口径完全一致（replay 引擎 / K_MODE=%d）。" % strategy.K_MODE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
