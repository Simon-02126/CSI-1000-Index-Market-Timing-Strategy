# -*- coding: utf-8 -*-
"""第七轮：东财 http 通道覆盖全部 9 个指数 + 回测区间深度"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import socket
import time

socket.setdefaulttimeout(30)
import requests

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://quote.eastmoney.com/"}
SEP = "=" * 100

CODES = [
    ("上证指数", "1.000001"), ("上证50", "1.000016"), ("科创50", "1.000688"),
    ("深证成指", "0.399001"), ("创业板指", "0.399006"), ("沪深300", "1.000300"),
    ("中证500", "1.000905"), ("中证1000", "1.000852"), ("中证2000", "2.932000"),
]

print(SEP)
print("东财 http K线通道 —— 9 个指数逐个测试（beg=19900101 拉全历史）")
print(SEP)
for name, sid in CODES:
    t0 = time.time()
    try:
        j = requests.get("http://push2his.eastmoney.com/api/qt/stock/kline/get",
                         params=dict(secid=sid, fields1="f1,f2,f3,f4,f5,f6",
                                     fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                                     klt=101, fqt=1, beg="19900101", end="20500101",
                                     lmt=100000),
                         headers=H, timeout=25).json()
        d = j.get("data")
        if not d or not d.get("klines"):
            print(f"  ❌ {name:8} secid={sid:10} 空 (rc={j.get('rc')})")
            continue
        kl = d["klines"]
        first, last = kl[0].split(","), kl[-1].split(",")
        print(f"  ✅ {name:8} secid={sid:10} name={d['name']:8} {len(kl):5d} 根  "
              f"{first[0]} ~ {last[0]}  末日收{last[2]} 换手{last[10]}%  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  ❌ {name:8} secid={sid:10} {type(e).__name__}: {str(e)[:60]}")
    time.sleep(1.0)

print()
print(SEP)
print("附：项目现有 9 个指数 CSV 的日期覆盖")
print(SEP)
import glob
import os
import pandas as pd

for fp in sorted(glob.glob(_os_path.path.join(_P.INPUT, "*指数*.csv")) +
                 glob.glob(_os_path.path.join(_P.INPUT, "*.SH.csv")) +
                 glob.glob(_os_path.path.join(_P.INPUT, "*.SZ.csv"))):
    try:
        df = None
        for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
            try:
                df = pd.read_csv(fp, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            continue
        dc = [c for c in df.columns if "日期" in c][0]
        d = pd.to_datetime(df[dc])
        print(f"  {os.path.basename(fp):32} {len(df):5d} 行  {d.min().date()} ~ {d.max().date()}  "
              f"{len(df.columns)} 列")
    except Exception as e:
        print(f"  {os.path.basename(fp):32} ❌ {type(e).__name__}: {str(e)[:50]}")
