# -*- coding: utf-8 -*-
"""情绪面数据经济含义校验"""
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

inp = _P.INPUT
w = pd.read_csv(os.path.join(inp, "情绪_面板_日频.csv"), encoding="utf-8-sig", parse_dates=["日期"])
c = pd.read_csv(os.path.join(inp, "中证1000_000852.SH.csv"), encoding="gbk")
c["日期"] = pd.to_datetime(c["日期"])
c = c[["日期", "收盘价"]].rename(columns={"收盘价": "close"})
m = w.merge(c, on="日期", how="left").set_index("日期")
lc = np.log(m["close"])
r = lc.diff()
f5 = (lc.shift(-5) - lc)
f10 = (lc.shift(-10) - lc)

print("=" * 96)
print("A. 与指数收益的相关性（判断方向与领先性）")
print("=" * 96)
print(f"  {'字段':<18}{'同期':>9}{'次日':>9}{'未来5日':>10}{'未来10日':>10}   预期方向")
cols = ["50ETF_IV平值", "50ETF_IV沽购比", "50ETF_IV偏斜", "50ETF_PCR成交", "50ETF_PCR持仓",
        "IF_会员多空比", "IM_会员多空比"]
for col in cols:
    if col not in m.columns:
        continue
    s = m[col]
    print(f"  {col:<18}{s.corr(r):>+9.3f}{s.corr(r.shift(-1)):>+9.3f}"
          f"{s.corr(f5):>+10.3f}{s.corr(f10):>+10.3f}", end="")
    exp = "IV/PCR 应与未来收益负相关" if ("IV" in col or "PCR" in col) else "会员多空比应与未来收益正相关"
    print(f"   {exp}")

print("\n" + "=" * 96)
print("B. 极端行情抽查（IV 应在危机中抬升、PCR 结构变化）")
print("=" * 96)
segs = [("2015-06-15", "2015-08-31", "2015 股灾"),
        ("2018-01-25", "2018-02-09", "2018 回调"),
        ("2020-02-03", "2020-03-23", "2020 疫情"),
        ("2022-04-01", "2022-04-30", "2022 封控"),
        ("2024-01-15", "2024-02-05", "2024 初流动性"),
        ("2024-09-24", "2024-10-08", "2024 924 行情"),
        ("2025-01-01", "2026-10-21", "近两年")]
base = m["50ETF_IV平值"].mean()
print(f"  全样本 IV 平值均值 = {base:.3f}")
print(f"  {'区间':<14}{'IV平值':>9}{'相对全样本':>11}{'PCR成交':>9}{'PCR持仓':>9}{'指数涨跌':>10}")
for d0, d1, lbl in segs:
    seg = m.loc[(m.index >= d0) & (m.index <= d1)]
    if len(seg) == 0:
        continue
    ret = np.expm1(np.log(seg["close"].iloc[-1] / seg["close"].iloc[0]))
    print(f"  {lbl:<14}{seg['50ETF_IV平值'].mean():>9.3f}"
          f"{seg['50ETF_IV平值'].mean()/base:>10.2f}x"
          f"{seg['50ETF_PCR成交'].mean():>9.2f}{seg['50ETF_PCR持仓'].mean():>9.2f}{ret:>10.1%}")

print("\n" + "=" * 96)
print("C. 分位覆盖度（供后续按研报做「反向布林带」用）")
print("=" * 96)
print(f"  {'字段':<18}{'有效样本':>9}{'q10':>9}{'q50':>9}{'q90':>9}{'最新':>9}")
for col in cols:
    if col not in m.columns:
        continue
    s = m[col].dropna()
    print(f"  {col:<18}{len(s):>9}{s.quantile(.1):>9.3f}{s.median():>9.3f}"
          f"{s.quantile(.9):>9.3f}{s.iloc[-1]:>9.3f}")
