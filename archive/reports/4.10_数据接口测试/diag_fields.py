# -*- coding: utf-8 -*-
"""两处异常定位：① 中金所表里是否有「合计」行 ② 换手率差异最大的那天"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import socket
socket.setdefaulttimeout(40)
import pandas as pd
import requests
import akshare as ak

BASE = _P.ROOT
SEP = "=" * 108


def rd(fp):
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(fp)


print(SEP)
print("① 中金所 get_cffex_rank_table 的原始表结构（2026-10-21 与 2026-10-23）")
print(SEP)
for d in ["20261021", "20261023"]:
    obj = ak.get_cffex_rank_table(date=d, vars_list=["IF"])
    print(f"\n   日期 {d}：合约数 {len(obj)}，键 = {list(obj.keys())}")
    for k, v in obj.items():
        print(f"   --- {k}  行数 {len(v)}  rank 范围 {v['rank'].min()}~{v['rank'].max()} ---")
        print(v[["rank", "long_party_name", "long_open_interest",
                 "short_party_name", "short_open_interest"]].to_string(index=False))
        print(f"       sum(long) 全部行 = {v['long_open_interest'].sum():.0f}")
        print(f"       sum(long) rank<=20 = {v[v['rank'] <= 20]['long_open_interest'].sum():.0f}")
        print(f"       sum(short) 全部行 = {v['short_open_interest'].sum():.0f}")
        print(f"       sum(short) rank<=20 = {v[v['rank'] <= 20]['short_open_interest'].sum():.0f}")
        break   # 只看第一个合约，避免刷屏

    lo_all = sum(float(v["long_open_interest"].sum()) for v in obj.values())
    so_all = sum(float(v["short_open_interest"].sum()) for v in obj.values())
    lo_20 = sum(float(v[v["rank"] <= 20]["long_open_interest"].sum()) for v in obj.values())
    so_20 = sum(float(v[v["rank"] <= 20]["short_open_interest"].sum()) for v in obj.values())
    print(f"\n   >>> 汇总：全部行 long={lo_all:.0f} short={so_all:.0f} "
          f"比={lo_all/so_all:.6f}  行数合计={sum(len(v) for v in obj.values())}")
    print(f"   >>> 汇总：rank<=20 long={lo_20:.0f} short={so_20:.0f} "
          f"比={lo_20/so_20:.6f}")

fut = rd(f"{BASE}\\input\\情绪_期货会员持仓.csv")
fut["dt"] = pd.to_datetime(fut["日期"]).dt.strftime("%Y-%m-%d")
for d in ["2026-10-21", "2026-10-23"]:
    r = fut[(fut["dt"] == d) & (fut["品种"] == "IF")]
    if not r.empty:
        r = r.iloc[0]
        print(f"   项目 CSV {d}: long={r['前20_持买单量']:.0f} short={r['前20_持卖单量']:.0f} "
              f"比={r['会员多空比']:.6f} 合约条数={r['合约条数']}")

print()
print(SEP)
print("② 换手率差异最大的那一天（东财 vs 项目）")
print(SEP)
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
j = requests.get("http://push2his.eastmoney.com/api/qt/stock/kline/get",
                 params=dict(secid="1.000852", fields1="f1,f2,f3,f4,f5,f6",
                             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                             klt=101, fqt=1, beg="20260801", end="20500101", lmt=10000),
                 headers=H, timeout=25).json()
rows = [x.split(",") for x in j["data"]["klines"]]
em = pd.DataFrame(rows, columns=["日期", "开", "收", "高", "低", "量", "额", "振幅",
                                 "涨跌幅", "涨跌额", "换手"]).set_index("日期")
em["换手"] = pd.to_numeric(em["换手"])
em["量"] = pd.to_numeric(em["量"]) * 100
em["额"] = pd.to_numeric(em["额"])

p = rd(f"{BASE}\\input\\中证1000_000852.SH.csv")
p["dt"] = pd.to_datetime(p["日期"]).dt.strftime("%Y-%m-%d")
p = p.set_index("dt")
j2 = pd.concat([em["换手"].rename("东财"), p["换手率(%)"].rename("项目")], axis=1).dropna()
j2["绝对差"] = (j2["东财"] - j2["项目"]).abs()
j2["相对差%"] = j2["绝对差"] / j2["项目"] * 100
print(j2.sort_values("绝对差", ascending=False).head(8).to_string())
print(f"\n   绝对差 中位 {j2['绝对差'].median():.4f} pct  均值 {j2['绝对差'].mean():.4f} pct  "
      f">0.005 pct 的天数 {int((j2['绝对差'] > 0.005).sum())}/{len(j2)}")
