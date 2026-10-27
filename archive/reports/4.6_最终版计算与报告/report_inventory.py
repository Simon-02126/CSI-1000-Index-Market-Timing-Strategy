# -*- coding: utf-8 -*-
"""报告用：数据清单。主数据文件逐列 + 原始源文件 + 合成文件 + 模型输出"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import glob
import numpy as np
import pandas as pd

BASE = _P.ROOT
HERE = os.path.dirname(os.path.abspath(__file__))


def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


def rng(df, dc):
    ds = pd.to_datetime(df[dc], errors="coerce").dropna()
    return str(ds.min().date()), str(ds.max().date())


print("=" * 150)
print("【表1】主数据文件 input/中证1000_000852.SH.csv —— 逐列说明（pv 与 slow 两个视图都只读这一个文件）")
print("=" * 150)
main = read_any(os.path.join(BASE, "input", "中证1000_000852.SH.csv"))
print(f"  总行数 {len(main)}，列数 {len(main.columns)}，日期 {rng(main, '日期')[0]} ~ {rng(main, '日期')[1]}")
print(f"  {'列名':<26}{'用途':<10}{'首个非空日':>12}{'缺失率':>9}  说明")
PV = {"开盘价", "最高价", "最低价", "收盘价", "均价(元)", "成交量(股)", "成交额(元)", "换手率(%)"}
SLOW = {"全市场融资余额(元)", "全市场融券余额(元)", "全市场融资净买入额(元)",
        "板块总市值(亿元)", "板块上市股票数"}
DESC = {
    "开盘价": "pv 通道1：log(开盘价 / 窗口末日收盘价)",
    "最高价": "pv 通道2：log(最高价 / 窗口末日收盘价)",
    "最低价": "pv 通道3：log(最低价 / 窗口末日收盘价)",
    "收盘价": "pv 通道4：log(收盘价 / 窗口末日收盘价)", "均价(元)": "pv 通道5：log(均价 / 窗口末日均价)",
    "成交量(股)": "pv 通道6：成交量 / 150日均值", "成交额(元)": "pv 通道7：成交额 / 150日均值",
    "换手率(%)": "pv 通道8：换手率 / 150日均值",
    "全市场融资余额(元)": "slow：存量，取 水平 + 150日变化 2 个特征",
    "全市场融券余额(元)": "slow：存量，取 水平 + 150日变化 2 个特征",
    "板块总市值(亿元)": "slow：存量，取 水平 + 150日变化 2 个特征",
    "板块上市股票数": "slow：存量，取 水平 + 150日变化 2 个特征",
    "全市场融资净买入额(元)": "slow：流量，取 累计流向/融资余额 + 单日强度 2 个特征",
}
for c in main.columns:
    c = str(c)
    first = "—"
    miss = 0.0
    if c not in ("日期",):
        v = pd.to_numeric(main[c], errors="coerce")
        nz = main.loc[v.notna() & (v != 0), "日期"]
        if len(nz):
            first = str(pd.to_datetime(nz).min().date())
        miss = float(v.isna().mean())
    if c == "日期":
        use, desc = "主键", "交易日"
    elif c in PV:
        use, desc = "pv", DESC[c]
    elif c in SLOW:
        use, desc = "slow", DESC[c]
    else:
        use, desc = "未用", "原始指数行情附带列"
    print(f"  {c:<26}{use:<10}{first:>12}{miss:>8.1%}  {desc}")
print()

print("=" * 150)
print("【表2】原始源文件（未加工的落盘数据）")
print("=" * 150)
raws = [
    ("data/两融数据.csv", "Mtdate", "沪深两市融资/融券余额、买入额、净买入额，逐交易所"),
    ("data/市场规模数据.csv", "SgnDate", "各口径板块总市值、上市股票数、换手率等"),
    ("input/情绪_期权每日统计.csv", "日期", "50/300/500ETF 期权成交量、持仓量、PCR、成交额"),
    ("input/情绪_期权IV.csv", "日期", "50/300/500ETF 期权平值/虚值 IV、沽购比、偏斜"),
    ("input/情绪_期货会员持仓.csv", "日期", "IF/IC/IM/IH 前20会员持买单量/持卖单量"),
]
print(f"  {'文件':<30}{'行数':>7}{'起始':>12}{'结束':>12}{'列':>4}  内容")
for rel, dcol, desc in raws:
    df = read_any(os.path.join(BASE, *rel.split("/")))
    if dcol in df.columns:
        d0, d1 = rng(df, dcol)
    else:
        d0 = d1 = "-"
    print(f"  {rel:<30}{len(df):>7}{d0:>12}{d1:>12}{len(df.columns):>4}  {desc}")
print()

print("=" * 150)
print("【表3】合成数据（由表2 派生，不是原始数据）")
print("=" * 150)
print(f"  {'合成文件':<28}{'行数':>7}{'有效起始':>12}{'结束':>12}{'列':>4}  合成方式")
print(f"  {'情绪_面板_日频.csv':<28}{'':>7}")
panel = read_any(os.path.join(BASE, "input", "情绪_面板_日频.csv"))
p0, p1 = rng(panel, "日期")
n_nonnull = panel.drop(columns=["日期"]).notna().sum()
print(f"  {'':<28}{len(panel):>7}{'见下':>12}{p1:>12}{len(panel.columns):>4}")
print("     合成方式：以中证1000 交易日历为左表，把 表2 的"
      "「期权每日统计 / 期权IV / 期货会员持仓」三张长表按 (日期, 标的) 透视成宽表后按日期合并。")
print(f"     注意：左连接后行数={len(panel)}（含 2015-02-09 之前的日历行，这些行的情绪字段为空），"
      f"情绪字段实际有效区间 2015-02-09 ~ {p1}")
print()
print(f"     {'面板列名':<22}{'来源文件':<26}{'合成规则'}")
for c in [str(x) for x in panel.columns if str(x) != "日期"]:
    if c.endswith("_会员多空比"):
        src, rule = "情绪_期货会员持仓.csv", f"{c[:2]} 品种：前20会员持买单量 / 前20会员持卖单量"
    elif "PCR" in c:
        src, rule = "情绪_期权每日统计.csv", ("认沽成交量/认购成交量" if "成交" in c else "认沽持仓量/认购持仓量")
    elif "成交额" in c:
        src, rule = "情绪_期权每日统计.csv", "该 ETF 期权当日总成交额（万元）"
    else:
        key = c.split("IV")[-1] if "IV" in c else ""
        src = "情绪_期权IV.csv"
        rule = {"平值": "平值合约隐含波动率", "虚沽": "虚值认沽隐含波动率",
                "虚购": "虚值认购隐含波动率", "沽购比": "IV_认沽认购比",
                "偏斜": "IV_虚沽 − IV_平值"}.get(key, "隐含波动率衍生")
    print(f"     {c:<22}{src:<26}{rule}")
print()
print("  另有两个派生信号文件（模型输出，非原始数据）：")
sig_files = [
    ("output/signals_v2/中证1000_000852.SH_pv_signals.csv", "pv 量价因子，2 seed 集成，64 维取均值"),
    ("output/signals_v2/中证1000_000852.SH_slow_signals.csv", "slow 资金因子，2 seed 集成，64 维取均值"),
]
print(f"    {'文件':<52}{'行数':>7}{'起始':>12}{'结束':>12}")
for rel, _ in sig_files:
    f = os.path.join(BASE, *rel.split("/"))
    df = read_any(f)
    d0, d1 = rng(df, "日期")
    print(f"    {rel:<52}{len(df):>7}{d0:>12}{d1:>12}")
print()

print("=" * 150)
print("【表4】本报告实际使用的 6-seed 集成信号（研究版，逐 seed 存档）")
print("=" * 150)
for tag, pat in (("量价 pv", "s?_*.csv"), ("资金 slow", "slow_s?_*.csv")):
    fs = sorted(glob.glob(os.path.join(HERE, "_train_out", pat)))
    d0 = d1 = None
    for f in fs:
        x = pd.read_csv(f, encoding="utf-8-sig", parse_dates=["日期"])
        a, b = x["日期"].min(), x["日期"].max()
        d0 = a if d0 is None or a < d0 else d0
        d1 = b if d1 is None or b > d1 else d1
    print(f"  {tag:<10} {len(fs):>3} 个文件（6 seed × 2019~2026 逐年）  {d0.date()} ~ {d1.date()}")
print()

print("=" * 150)
print("【表5】备选标的（本报告未使用，仅列出以供扩展）")
print("=" * 150)
print(f"  {'文件':<30}{'行数':>7}{'起始':>12}{'结束':>12}")
for f in sorted(glob.glob(os.path.join(BASE, "input", "*.csv"))):
    b = os.path.basename(f)
    if "中证1000" in b or "情绪" in b:
        continue
    df = read_any(f)
    d0, d1 = rng(df, "日期")
    print(f"  {b:<30}{len(df):>7}{d0:>12}{d1:>12}")
print("\n完成")
