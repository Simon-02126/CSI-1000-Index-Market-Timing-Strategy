# -*- coding: utf-8 -*-
"""第二轮：补测失败的接口 + 参数纠正"""
import socket
import time
import traceback

socket.setdefaulttimeout(30)
import requests

LAST = "20261023"
SEP = "=" * 100


def hdr(s):
    print()
    print(SEP)
    print(f"【{s}】")


# ---------------------------------------------------------------- 1. 中证2000 东财 secid
hdr("1. 东财搜索接口解析 932000（中证2000）的真实 secid")
try:
    r = requests.get("https://searchapi.eastmoney.com/api/suggest/get",
                     params=dict(input="932000", type="14",
                                 token="TOKEN_HERE"),
                     timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    j = r.json()
    items = (j.get("QuotationCodeTable") or {}).get("Data") or []
    print(f"   命中 {len(items)} 条")
    for it in items[:8]:
        print(f"     code={it.get('Code')} name={it.get('Name')} "
              f"QuoteID={it.get('QuoteID')} MktNum={it.get('MktNum')} "
              f"SecurityType={it.get('SecurityTypeName')}")
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}")


def em_kline(secid, beg="20260901"):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    p = dict(secid=secid, fields1="f1,f2,f3,f4,f5,f6",
             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
             klt=101, fqt=1, beg=beg, end="20500101", lmt=10000)
    j = requests.get(url, params=p, timeout=25,
                     headers={"User-Agent": "Mozilla/5.0"}).json()
    if not j.get("data") or not j["data"].get("klines"):
        raise RuntimeError(f"空 (rc={j.get('rc')})")
    d = j["data"]
    print(f"   name={d.get('name')} code={d.get('code')} 共 {len(d['klines'])} 根")
    for x in d["klines"][-2:]:
        print("     " + x)


for sid in ["2.932000", "0.932000", "1.932000"]:
    hdr(f"2. 东财 push2his  secid={sid}")
    t0 = time.time()
    try:
        em_kline(sid)
        print(f"   ✅ {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   ❌ {time.time()-t0:.1f}s  {type(e).__name__}: {e}")

# ---------------------------------------------------------------- 3. B2 重试
hdr("3. akshare index_zh_a_hist 重试（000852）")
for attempt in range(1, 4):
    try:
        import akshare as ak
        df = ak.index_zh_a_hist(symbol="000852", period="daily",
                                start_date="20260901", end_date=LAST)
        print(f"   ✅ 第 {attempt} 次成功  形状 {df.shape}")
        print(f"   列名 {list(df.columns)}")
        print(df.tail(3).to_string())
        break
    except Exception as e:
        print(f"   ❌ 第 {attempt} 次  {type(e).__name__}: {str(e)[:120]}")
        time.sleep(2)

# ---------------------------------------------------------------- 4. E2 dict 结构
hdr("4. 中金所会员持仓 get_cffex_rank_table 返回结构")
try:
    import akshare as ak
    for dt in ["20261023", "20261022"]:
        obj = ak.get_cffex_rank_table(date=dt, vars_list=["IF"])
        print(f"   date={dt}  type={type(obj)}  len={len(obj) if hasattr(obj,'__len__') else '?'}")
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:6]:
                print(f"      key={k!r}  value_type={type(v)}  "
                      f"len={len(v) if hasattr(v,'__len__') else '?'}")
                if hasattr(v, "columns"):
                    print(f"         columns={list(v.columns)}")
                    print(v.head(3).to_string())
                elif isinstance(v, (list, tuple)) and v:
                    print(f"         首个元素: {repr(v[0])[:300]}")
        elif hasattr(obj, "columns"):
            print(f"      columns={list(obj.columns)}")
            print(obj.head(3).to_string())
except Exception as e:
    print(f"   ❌ {type(e).__name__}: {e}")
    for ln in traceback.format_exc().strip().splitlines()[-4:]:
        print(f"     {ln}")

print()
print(SEP)
print("第二轮结束")
