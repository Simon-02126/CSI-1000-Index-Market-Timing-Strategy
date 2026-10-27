# -*- coding: utf-8 -*-
"""第五轮：把免费接口拉到的数，与项目现有 CSV 做对数校验（文档遗留事项 2）"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import socket
import time

socket.setdefaulttimeout(40)
import pandas as pd
import requests

BASE = _P.ROOT
SEP = "=" * 100


def hdr(s):
    print()
    print(SEP)
    print(f"【{s}】", flush=True)


def read_csv_any(fp):
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取 {fp}")


# ---------------------------------------------------------------- 两融 CSV 结构
hdr("1. 项目现有 两融数据.csv 的结构")
m = read_csv_any(f"{BASE}\\data\\两融数据.csv")
print(f"   形状 {m.shape}")
print(f"   列名 {list(m.columns)}")
print(m.tail(4).to_string())
dcol = [c for c in m.columns if str(c).lower().startswith("mt")][0]
print(f"   日期列 = {dcol}，类型 {m[dcol].dtype}，最后 3 值 {list(m[dcol].tail(3))}")

# ---------------------------------------------------------------- 市场规模 CSV 结构
hdr("2. 项目现有 市场规模数据.csv 的结构")
s = read_csv_any(f"{BASE}\\data\\市场规模数据.csv")
print(f"   形状 {s.shape}")
print(f"   列名 {list(s.columns)}")
print(s.tail(3).to_string())
scol = [c for c in s.columns if str(c).lower().startswith("sgn")][0]
print(f"   日期列 = {scol}，最后 3 值 {list(s[scol].tail(3))}")

# ---------------------------------------------------------------- 量价 CSV 结构
hdr("3. 项目现有 中证1000 主数据的结构（用于量价对数校验）")
p = read_csv_any(f"{BASE}\\input\\中证1000_000852.SH.csv")
print(f"   形状 {p.shape}")
print(f"   列名 {list(p.columns)}")
print(p.tail(3).to_string())

# ---------------------------------------------------------------- 4. 量价：东财 http vs 项目 CSV
hdr("4. 对数校验 A：东财 http K线（1.000852）vs 项目 CSV")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://quote.eastmoney.com/"}
j = requests.get("http://push2his.eastmoney.com/api/qt/stock/kline/get",
                 params=dict(secid="1.000852", fields1="f1,f2,f3,f4,f5,f6",
                             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                             klt=101, fqt=1, beg="20260920", end="20500101", lmt=1000),
                 headers=H, timeout=25).json()
kl = j["data"]["klines"]
rows = []
for x in kl:
    a = x.split(",")
    rows.append(dict(日期=a[0], 开=float(a[1]), 收=float(a[2]), 高=float(a[3]),
                     低=float(a[4]), 量=float(a[5]), 额=float(a[6]), 换手=float(a[10])))
em = pd.DataFrame(rows)
print("   东财返回（末 5 行）：")
print(em.tail(5).to_string(index=False))

p2 = p.copy()
p2["dt"] = pd.to_datetime(p2[[c for c in p2.columns if "日期" in c][0]]).dt.strftime("%Y-%m-%d")
cc = [c for c in p2.columns if "收盘" in c][0]
oo = [c for c in p2.columns if "开盘" in c][0]
hh = [c for c in p2.columns if "最高" in c][0]
ll = [c for c in p2.columns if "最低" in c][0]
vol = [c for c in p2.columns if "成交量" in c][0]
amt = [c for c in p2.columns if "成交额" in c][0]
tur = [c for c in p2.columns if "换手" in c][0]
w = p2[p2["dt"].isin(em["日期"])].copy()
w = w.rename(columns={cc: "收_p", oo: "开_p", hh: "高_p", ll: "低_p",
                      vol: "量_p", amt: "额_p", tur: "换手_p"})
cmp = em.merge(w[["dt", "开_p", "收_p", "高_p", "低_p", "量_p", "额_p", "换手_p"]],
               left_on="日期", right_on="dt", how="left")
print()
print("   逐日对比（东财 vs 项目）：")
for _, r in cmp.iterrows():
    print(f"     {r['日期']}  收 {r['收']:.2f}/{r['收_p']:.2f}  开 {r['开']:.2f}/{r['开_p']:.2f}  "
          f"量 {r['量']:.0f}/{r['量_p']:.0f}  额 {r['额']:.0f}/{r['额_p']:.0f}  "
          f"换手 {r['换手']:.2f}/{r['换手_p']:.2f}")
print(f"   收盘最大相对误差 {((cmp['收'] - cmp['收_p']).abs() / cmp['收_p']).max()*100:.6f}%")
print(f"   成交量最大相对误差 {((cmp['量'] - cmp['量_p']).abs() / cmp['量_p'].replace(0, pd.NA)).max()*100:.6f}%")
print(f"   成交额最大相对误差 {((cmp['额'] - cmp['额_p']).abs() / cmp['额_p'].replace(0, pd.NA)).max()*100:.6f}%")
print(f"   换手率最大相对误差 {((cmp['换手'] - cmp['换手_p']).abs() / cmp['换手_p'].replace(0, pd.NA)).max()*100:.4f}%")

print()
print(SEP)
print("第五轮结束")
