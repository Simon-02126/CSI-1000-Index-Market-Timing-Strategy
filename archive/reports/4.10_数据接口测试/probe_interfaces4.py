# -*- coding: utf-8 -*-
"""第四轮：baostock 重试 / 东财分片域名绕过 / 其他量价备选源"""
import socket
import time

socket.setdefaulttimeout(60)
import requests

SEP = "=" * 100


def hdr(s):
    print()
    print(SEP)
    print(f"【{s}】", flush=True)


# ---------------------------------------------------------------- 1. baostock 重试
hdr("1. baostock 重试（更长的 socket 超时，3 次）")
try:
    import baostock as bs
    for i in range(1, 4):
        print(f"   第 {i} 次 login ...", flush=True)
        t0 = time.time()
        lg = bs.login()
        print(f"      code={lg.error_code} msg={lg.error_msg}  ({time.time()-t0:.1f}s)", flush=True)
        if lg.error_code == "0":
            rs = bs.query_history_k_data_plus(
                "sh.000852",
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date="2026-10-15", end_date="2026-10-24",
                frequency="d", adjustflag="3")
            print(f"      query error_code={rs.error_code} msg={rs.error_msg}", flush=True)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            print(f"      行数 {len(rows)}  fields={rs.fields}", flush=True)
            for r in rows:
                print("        " + " | ".join(r), flush=True)
            bs.logout()
            print("   ✅ baostock 可用", flush=True)
            break
        time.sleep(5)
    else:
        print("   ❌ baostock 三次登录均失败", flush=True)
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}", flush=True)

# ---------------------------------------------------------------- 2. 东财分片域名
hdr("2. 东方财富 push2his 分片域名尝试（secid=1.000852）")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
     "Referer": "https://quote.eastmoney.com/"}
P = dict(secid="1.000852", fields1="f1,f2,f3,f4,f5,f6",
         fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
         klt=101, fqt=1, beg="20260901", end="20500101", lmt=10000)
for host in ["push2his.eastmoney.com",
             "1.push2his.eastmoney.com", "23.push2his.eastmoney.com",
             "45.push2his.eastmoney.com", "99.push2his.eastmoney.com"]:
    for scheme in (["http", "https"] if host == "push2his.eastmoney.com" else ["https"]):
        url = f"{scheme}://{host}/api/qt/stock/kline/get"
        print(f"   {url} ...", end="", flush=True)
        try:
            r = requests.get(url, params=P, headers=H, timeout=20)
            j = r.json()
            if j.get("data") and j["data"].get("klines"):
                kl = j["data"]["klines"]
                print(f"      ✅ {len(kl)} 根, 末日 {kl[-1].split(',')[0]}")
                break
            else:
                print(f"      ⚠ HTTP {r.status_code}, rc={j.get('rc')} data为空")
        except Exception as e:
            print(f"      ❌ {type(e).__name__}: {str(e)[:70]}")
        time.sleep(3)

# ---------------------------------------------------------------- 3. 其他量价备选
hdr("3. 其他免费量价备选源（akshare 内置）")
import akshare as ak

TESTS = [
    ("sina stock_zh_index_daily(sh000852)",
     lambda: ak.stock_zh_index_daily(symbol="sh000852")),
    ("tencent stock_zh_index_daily_tx(sh000852)",
     lambda: ak.stock_zh_index_daily_tx(symbol="sh000852")),
    ("csindex 全历史 stock_zh_index_hist_csindex(000852, 2012起)",
     lambda: ak.stock_zh_index_hist_csindex(symbol="000852",
                                            start_date="20120101", end_date="20261024")),
    ("csindex 中证2000(932000)",
     lambda: ak.stock_zh_index_hist_csindex(symbol="932000",
                                            start_date="20260901", end_date="20261024")),
    ("csindex 科创50(000688)",
     lambda: ak.stock_zh_index_hist_csindex(symbol="000688",
                                            start_date="20260901", end_date="20261024")),
]
for name, fn in TESTS:
    print(f"   {name} ...", end="", flush=True)
    t0 = time.time()
    try:
        df = fn()
        print(f"      ✅ {df.shape}  {time.time()-t0:.1f}s")
        print(f"         列 {list(df.columns)}")
        print(f"         首行 {df.iloc[0].to_dict()}")
        print(f"         末行 {df.iloc[-1].to_dict()}")
    except Exception as e:
        print(f"      ❌ {type(e).__name__}: {str(e)[:90]}  {time.time()-t0:.1f}s")
    time.sleep(2)

print()
print(SEP)
print("第四轮结束")
