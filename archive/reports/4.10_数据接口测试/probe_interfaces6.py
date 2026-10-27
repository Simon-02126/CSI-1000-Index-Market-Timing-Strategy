# -*- coding: utf-8 -*-
"""第六轮：两融 + 市场规模 的对数校验（同日期逐字段）"""
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

import akshare as ak

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
    raise RuntimeError(fp)


m = read_csv_any(f"{BASE}\\data\\两融数据.csv")
s = read_csv_any(f"{BASE}\\data\\市场规模数据.csv")

DATES = ["2026-10-21", "2026-10-20", "2026-10-19", "2026-10-16", "2026-10-15"]

# ---------------------------------------------------------------- 市场规模口径编码
hdr("0. 市场规模 CSV 里最近日期出现过的 DataSgnCode")
last = s[s["SgnDate"] == s["SgnDate"].max()]
print(f"   最新日期 {s['SgnDate'].max()}")
print(last[["DataSgnCode", "DataSgn", "ListedCoNum", "StockNum", "MarketValue",
            "CirculatedMktValue", "Amount"]].to_string(index=False))

# ---------------------------------------------------------------- 1. 两融-上交所
hdr("1. 对数校验 B-1：两融 上交所（akshare）vs CSV 的 SHSE 行")
sse = ak.stock_margin_sse(start_date="20261015", end_date="20261024")
print("   akshare 返回列：", list(sse.columns))
ok = 0
for _, r in sse.iterrows():
    d = str(r["信用交易日期"])
    dt = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    c = m[(m["Mtdate"] == dt) & (m["Exchangecode"] == "SHSE")]
    if c.empty:
        print(f"   {dt}  CSV 无 SHSE 行")
        continue
    c = c.iloc[0]
    pairs = [("融资余额", "Financebalance"), ("融资买入额", "Financeamount"),
             ("融券余量金额", "Shortbalance"), ("融资融券余额", "Mtbalance"),
             ("融券余量", "Secshortbalance")]
    diffs = []
    for a_, b_ in pairs:
        x = float(r[a_]); y = float(c[b_])
        rel = abs(x - y) / abs(y) if y else float("nan")
        diffs.append(f"{b_}={rel*100:.8f}%")
    same = all(abs(float(r[a_]) - float(c[b_])) / abs(float(c[b_])) < 1e-6
               for a_, b_ in pairs if float(c[b_]) != 0)
    print(f"   {dt}  " + ("✅完全一致 " if same else "⚠ 有差异  ") + " ".join(diffs))
    ok += same
print(f"   → {ok}/{len(sse)} 天完全一致")

# ---------------------------------------------------------------- 2. 两融-深交所
hdr("2. 对数校验 B-2：两融 深交所（akshare，单位亿元）vs CSV 的 SZSE 行")
for dt in DATES:
    try:
        z = ak.stock_margin_szse(date=dt.replace("-", ""))
        r = z.iloc[0]
        c = m[(m["Mtdate"] == dt) & (m["Exchangecode"] == "SZSE")]
        if c.empty:
            print(f"   {dt}  CSV 无 SZSE 行")
            continue
        c = c.iloc[0]
        out = []
        for a_, b_, sc in [("融资余额", "Financebalance", 1e8),
                           ("融资买入额", "Financeamount", 1e8),
                           ("融券余额", "Shortbalance", 1e8),
                           ("融资融券余额", "Mtbalance", 1e8)]:
            x = float(r[a_]) * sc
            y = float(c[b_])
            out.append(f"{b_}: {x:.0f} vs {y:.0f} ({(x-y)/y*100:+.6f}%)")
        print(f"   {dt}  " + " | ".join(out))
    except Exception as e:
        print(f"   {dt}  ❌ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.5)

# ---------------------------------------------------------------- 3. 市场规模-上交所
hdr("3. 对数校验 C-1：市场规模 上交所 daily 概况 vs CSV")
for dt in DATES:
    try:
        x = ak.stock_sse_deal_daily(date=dt.replace("-", ""))
        row = x[x["单日情况"] == "挂牌数"]
        mkt = x[x["单日情况"] == "市价总值"]
        if row.empty:
            print(f"   {dt}  无挂牌数行")
            continue
        print(f"   {dt}  上交所: 挂牌数(股票)={float(row.iloc[0]['股票']):.0f}  "
              f"市价总值(股票)={float(mkt.iloc[0]['股票']):.2f} 亿元  "
              f"科创板: 挂牌数={float(row.iloc[0]['科创板']):.0f} "
              f"市值={float(mkt.iloc[0]['科创板']):.2f}")
        c = s[(s["SgnDate"] == dt)]
        for _, rr in c.iterrows():
            if rr["DataSgnCode"] in (1, 2, 4, 11, 12, 14):
                print(f"        CSV code={rr['DataSgnCode']:<3} {rr['DataSgn']:<12} "
                      f"ListedCoNum={rr['ListedCoNum']} StockNum={rr['StockNum']} "
                      f"MarketValue={rr['MarketValue']:.2f} "
                      f"CirculMkt={rr['CirculatedMktValue']:.2f}")
    except Exception as e:
        print(f"   {dt}  ❌ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.5)

# ---------------------------------------------------------------- 4. 市场规模-深交所
hdr("4. 对数校验 C-2：市场规模 深交所总貌 vs CSV")
for dt in DATES:
    try:
        x = ak.stock_szse_summary(date=dt.replace("-", ""))
        r = x[x["证券类别"] == "股票"]
        if r.empty:
            print(f"   {dt}  无股票行")
            continue
        r = r.iloc[0]
        print(f"   {dt}  深交所: 数量={r['数量']}  总市值={r['总市值']/1e8:.2f} 亿元  "
              f"流通市值={r['流通市值']/1e8:.2f} 亿元")
        c = s[s["SgnDate"] == dt]
        for _, rr in c.iterrows():
            if rr["DataSgnCode"] in (5, 8, 11, 14):
                print(f"        CSV code={rr['DataSgnCode']:<3} {rr['DataSgn']:<12} "
                      f"StockNum={rr['StockNum']} MarketValue={rr['MarketValue']:.2f}")
    except Exception as e:
        print(f"   {dt}  ❌ {type(e).__name__}: {str(e)[:80]}")
    time.sleep(1.5)

print()
print(SEP)
print("第六轮结束")
