# -*- coding: utf-8 -*-
"""合并情绪面数据分片，输出到项目 input/ 目录

产出:
  input/情绪_期权每日统计.csv     期权成交/持仓/PCR（按标的，long 格式）
  input/情绪_期权IV.csv           隐含波动率分档与偏斜（按标的，long 格式）
  input/情绪_期货会员持仓.csv      中金所前20会员多空比（按品种，long 格式）
  input/情绪_面板_日频.csv         宽表：50ETF为主 + 300/500ETF辅助 + IF/IC/IM，与指数日期对齐
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
PART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_senti_parts")


def merge(job, key_cols, out_name):
    files = sorted(glob.glob(os.path.join(PART, f"{job}_*.csv")))
    if not files:
        print(f"[{job}] 无分片")
        return None
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")
    if not dfs:
        return None
    d = pd.concat(dfs, ignore_index=True)
    d["日期"] = pd.to_datetime(d["日期"], errors="coerce")
    d = d.dropna(subset=["日期"]).drop_duplicates(subset=["日期"] + key_cols, keep="last")
    d = d.sort_values(["日期"] + key_cols).reset_index(drop=True)
    fp = os.path.join(INPUT, out_name)
    d.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"[{job}] {len(d)} 行 -> {out_name}  "
          f"({d['日期'].min().date()} ~ {d['日期'].max().date()}, {d['日期'].nunique()} 个交易日)")
    return d


print("=" * 90)
stats = merge("stats", ["标的代码"], "情绪_期权每日统计.csv")
risk = merge("risk", ["标的代码"], "情绪_期权IV.csv")
cffex = merge("cffex", ["品种"], "情绪_期货会员持仓.csv")


def brief(d, name, cols):
    if d is None:
        return
    print(f"\n--- {name} 字段完整性 ---")
    print(f"  {'列':<24}{'非空率':>8}{'均值':>14}{'最新值':>14}")
    for c in cols:
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        last = s.dropna().iloc[-1] if s.notna().any() else np.nan
        print(f"  {c:<24}{s.notna().mean():>8.0%}{s.mean():>14.4f}{last:>14.4f}")


if stats is not None:
    brief(stats, "期权每日统计", ["总成交额_万元", "总成交量", "PCR_成交_认沽比认购", "PCR_持仓_认沽比认购"])
    print("\n  各标的可用区间:")
    for sid, g in stats.groupby("标的代码"):
        print(f"    {sid} ({g['标的名称'].iloc[0]}): {g['日期'].min().date()} ~ {g['日期'].max().date()}"
              f"  {len(g)} 日")
if risk is not None:
    brief(risk, "期权IV", ["IV_平值", "IV_虚值认购", "IV_虚值认沽", "IV_认沽认购比", "IV_偏斜_虚沽减平值"])
    print("\n  各标的可用区间:")
    for sid, g in risk.groupby("标的代码"):
        print(f"    {sid}: {g['日期'].min().date()} ~ {g['日期'].max().date()}  {len(g)} 日")
if cffex is not None:
    brief(cffex, "期货会员持仓", ["前20_持买单量", "前20_持卖单量", "会员多空比"])
    print("\n  各品种可用区间:")
    for v, g in cffex.groupby("品种"):
        print(f"    {v}: {g['日期'].min().date()} ~ {g['日期'].max().date()}  {len(g)} 日")

# ---------------- 宽表：与指数日期对齐 ----------------
idx_fp = os.path.join(INPUT, "中证1000_000852.SH.csv")
idx = pd.read_csv(idx_fp, encoding="gbk", usecols=[0])
idx.columns = ["日期"]
idx["日期"] = pd.to_datetime(idx["日期"], errors="coerce")
idx = idx.dropna().drop_duplicates().sort_values("日期")
wide = idx.set_index("日期")

if stats is not None:
    for sid, tag in [("510050", "50ETF"), ("510300", "300ETF"), ("510500", "500ETF")]:
        g = stats[stats["标的代码"].astype(str) == sid].set_index("日期")
        for c, cn in [("PCR_成交_认沽比认购", "PCR成交"), ("PCR_持仓_认沽比认购", "PCR持仓"),
                      ("总成交额_万元", "成交额万元")]:
            if c in g.columns:
                wide[f"{tag}_{cn}"] = g[c].reindex(wide.index)
if risk is not None:
    for sid, tag in [("510050", "50ETF"), ("510300", "300ETF"), ("510500", "500ETF")]:
        g = risk[risk["标的代码"].astype(str) == sid].set_index("日期")
        for c, cn in [("IV_平值", "IV平值"), ("IV_虚值认沽", "IV虚沽"),
                      ("IV_虚值认购", "IV虚购"), ("IV_认沽认购比", "IV沽购比"),
                      ("IV_偏斜_虚沽减平值", "IV偏斜")]:
            if c in g.columns:
                wide[f"{tag}_{cn}"] = g[c].reindex(wide.index)
if cffex is not None:
    for v in ("IF", "IC", "IM", "IH"):
        g = cffex[cffex["品种"] == v].set_index("日期")
        if "会员多空比" in g.columns:
            wide[f"{v}_会员多空比"] = g["会员多空比"].reindex(wide.index)

wide = wide.reset_index()
fp = os.path.join(INPUT, "情绪_面板_日频.csv")
wide.to_csv(fp, index=False, encoding="utf-8-sig")
print(f"\n[面板] {len(wide)} 行 × {len(wide.columns)} 列 -> 情绪_面板_日频.csv")
print(f"  日期 {wide['日期'].min().date()} ~ {wide['日期'].max().date()}")
nn = wide.drop(columns=["日期"]).notna().mean().sort_values(ascending=False)
print("  各列非空率（前 12）:")
for c, v in nn.head(12).items():
    first = wide.loc[wide[c].notna(), "日期"].min()
    print(f"    {c:<22}{v:>7.1%}   起始 {first.date() if pd.notna(first) else '-'}")
