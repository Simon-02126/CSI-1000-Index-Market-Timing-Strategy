# -*- coding: utf-8 -*-
"""第三轮：baostock（现行量价源）+ 东财限流复核 + 中证2000 secid 解析"""
import socket
import time
import traceback

socket.setdefaulttimeout(30)
import requests

SEP = "=" * 100


def hdr(s):
    print()
    print(SEP)
    print(f"【{s}】")


# ---------------------------------------------------------------- 1. baostock
hdr("1. baostock 日频接口（文档中的现行量价数据源）")
try:
    import baostock as bs
    lg = bs.login()
    print(f"   login: code={lg.error_code} msg={lg.error_msg}")
    if lg.error_code != "0":
        raise RuntimeError(f"登录失败: {lg.error_msg}")

    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    for code in ["sh.000852", "sh.000688", "sh.932000", "sh.000001"]:
        print()
        print(f"   --- {code} ---")
        t0 = time.time()
        rs = bs.query_history_k_data_plus(
            code, fields, start_date="2026-09-01", end_date="2026-10-24",
            frequency="d", adjustflag="3")
        print(f"   error_code={rs.error_code} msg={rs.error_msg}")
        if rs.error_code != "0":
            continue
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        print(f"   行数 {len(rows)}（{time.time()-t0:.1f}s）")
        print(f"   列 {rs.fields}")
        for r in rows[:2] + rows[-3:]:
            print("     " + " | ".join(r))
    # 换手率到底有没有
    print()
    print("   换手率专项复测（仅 turn 一列，最近 5 日）：")
    rs = bs.query_history_k_data_plus("sh.000852", "date,close,turn",
                                      start_date="2026-10-15", end_date="2026-10-24",
                                      frequency="d", adjustflag="3")
    while rs.next():
        print("     " + " | ".join(rs.get_row_data()))
    bs.logout()
    print("   ✅ baostock 测试完成")
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}")
    for ln in traceback.format_exc().strip().splitlines()[-4:]:
        print(f"     {ln}")

# ---------------------------------------------------------------- 2. 东财限流复核
hdr("2. 东方财富 push2his 限流复核（间隔 8s，各试 2 次）")
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                  "Referer": "https://quote.eastmoney.com/"})


def em_kline(secid):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    p = dict(secid=secid, fields1="f1,f2,f3,f4,f5,f6",
             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
             klt=101, fqt=1, beg="20260901", end="20500101", lmt=10000)
    j = S.get(url, params=p, timeout=25).json()
    if not j.get("data") or not j["data"].get("klines"):
        raise RuntimeError(f"空 (rc={j.get('rc')})")
    d = j["data"]
    print(f"      name={d.get('name')} code={d.get('code')} 共 {len(d['klines'])} 根")
    print("      最后1根: " + d["klines"][-1])


for sid in ["1.000852", "1.000688", "1.932000"]:
    for attempt in (1, 2):
        print(f"   secid={sid} 第{attempt}次 ...", end="", flush=True)
        try:
            em_kline(sid)
            print("      ✅ 可用")
            break
        except Exception as e:
            print(f"      ❌ {type(e).__name__}: {str(e)[:80]}")
            time.sleep(8)

# ---------------------------------------------------------------- 3. 东财 secid 解析
hdr("3. 用东财行情快照接口判断 932000 属于哪个市场")
for sid in ["1.932000", "2.932000", "0.932000", "1.000852", "1.000688"]:
    try:
        j = S.get("https://push2.eastmoney.com/api/qt/stock/get",
                  params=dict(secid=sid, fields="f57,f58,f43,f107"),
                  timeout=15).json()
        d = j.get("data")
        if d:
            print(f"   secid={sid}  f57(code)={d.get('f57')}  f58(name)={d.get('f58')} "
                  f"f43(price)={d.get('f43')} f107(市场)={d.get('f107')}")
        else:
            print(f"   secid={sid}  空 (rc={j.get('rc')})")
    except Exception as e:
        print(f"   secid={sid}  ❌ {type(e).__name__}: {str(e)[:70]}")
    time.sleep(3)

print()
print(SEP)
print("第三轮结束")
