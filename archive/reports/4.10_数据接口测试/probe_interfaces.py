# -*- coding: utf-8 -*-
"""逐个测试《数据来源与更新说明.md》里提到的数据接口是否可用"""
import socket
import time
import traceback

socket.setdefaulttimeout(30)

import requests

LAST = "20261023"      # 最近一个交易日
START = "20260901"
SEP = "=" * 100


def show(df, n=3, cols=None):
    if df is None:
        print("   返回 None")
        return
    print(f"   形状 {df.shape}")
    if cols:
        print(f"   列名 {list(df.columns)}")
    print(df.head(n).to_string())
    print("   ... 尾部")
    print(df.tail(n).to_string())


def t(name, fn):
    print()
    print(SEP)
    print(f"【{name}】")
    t0 = time.time()
    try:
        r = fn()
        print(f"   ✅ 成功，耗时 {time.time()-t0:.1f}s")
        return r
    except Exception as e:
        print(f"   ❌ 失败，耗时 {time.time()-t0:.1f}s")
        print(f"   {type(e).__name__}: {e}")
        tb = traceback.format_exc().strip().splitlines()
        for ln in tb[-4:]:
            print(f"     {ln}")
        return None


# ---------------------------------------------------------------- A. 东方财富 push2his
def em_kline(secid, beg=START, end="20500101"):
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    p = dict(secid=secid, fields1="f1,f2,f3,f4,f5,f6",
             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
             klt=101, fqt=1, beg=beg, end=end, lmt=10000)
    r = requests.get(url, params=p, timeout=25,
                     headers={"User-Agent": "Mozilla/5.0"})
    j = r.json()
    if not j.get("data") or not j["data"].get("klines"):
        raise RuntimeError(f"返回为空: code={j.get('rc')} data={j.get('data')}")
    kl = j["data"]["klines"]
    print(f"   name={j['data'].get('name')} code={j['data'].get('code')} 共 {len(kl)} 根K线")
    print("   最后 3 行（日期,开,收,高,低,量,额,振幅,涨跌幅,涨跌额,换手率）:")
    for x in kl[-3:]:
        print("     " + x)
    return kl


# ---------------------------------------------------------------- B. akshare csindex
def ak_csindex():
    import akshare as ak
    df = ak.stock_zh_index_hist_csindex(symbol="000852", start_date="20260901", end_date=LAST)
    show(df, cols=True)
    return df


def ak_csindex_em():
    import akshare as ak
    df = ak.index_zh_a_hist(symbol="000852", period="daily",
                            start_date="20260901", end_date=LAST)
    show(df, cols=True)
    return df


# ---------------------------------------------------------------- C. 两融
def ak_margin_sse():
    import akshare as ak
    df = ak.stock_margin_sse(start_date="20260901", end_date=LAST)
    show(df, cols=True)
    return df


def ak_margin_szse():
    import akshare as ak
    df = ak.stock_margin_szse(date=LAST)
    show(df, cols=True)
    return df


# ---------------------------------------------------------------- D. 市场规模
def ak_sse_summary():
    import akshare as ak
    df = ak.stock_sse_summary()
    show(df, cols=True)
    return df


def ak_sse_deal_daily():
    import akshare as ak
    df = ak.stock_sse_deal_daily(date=LAST)
    show(df, n=8, cols=True)
    return df


def ak_szse_summary():
    import akshare as ak
    df = ak.stock_szse_summary(date=LAST)
    show(df, n=8, cols=True)
    return df


# ---------------------------------------------------------------- E. 情绪（已用过，复核）
def ak_sse_option():
    import akshare as ak
    df = ak.option_risk_indicator_sse(date=LAST)
    show(df, cols=True)
    return df


def ak_cffex():
    import akshare as ak
    df = ak.get_cffex_rank_table(date=LAST, vars_list=["IF"])
    show(df, cols=True)
    return df


print(SEP)
print(f"接口可用性测试    最近交易日 = {LAST}")
print(SEP)

t("A1. 东方财富 push2his K线  secid=1.000688（科创50）", lambda: em_kline("1.000688"))
t("A2. 东方财富 push2his K线  secid=1.932000（中证2000）", lambda: em_kline("1.932000"))
t("B1. akshare 中证指数官网 csindex（000852 中证1000）", ak_csindex)
t("B2. akshare 东财指数历史 index_zh_a_hist（000852）", ak_csindex_em)
t("C1. akshare 两融-上交所 stock_margin_sse", ak_margin_sse)
t("C2. akshare 两融-深交所 stock_margin_szse", ak_margin_szse)
t("D1. akshare 上交所每日概况 stock_sse_summary", ak_sse_summary)
t("D2. akshare 上交所每日成交概况 stock_sse_deal_daily", ak_sse_deal_daily)
t("D3. akshare 深交所市场总貌 stock_szse_summary", ak_szse_summary)
t("E1. akshare 上交所期权风险指标 option_risk_indicator_sse", ak_sse_option)
t("E2. akshare 中金所会员持仓 get_cffex_rank_table", ak_cffex)

print()
print(SEP)
print("测试结束")
