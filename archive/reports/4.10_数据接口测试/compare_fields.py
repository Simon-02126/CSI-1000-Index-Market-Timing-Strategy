# -*- coding: utf-8 -*-
"""入模字段级爬取对比：把策略实际用到的每个字段，用免费接口重新抓一遍，与项目 CSV 逐日核对

策略实际用到的字段：
  量价 pv 8 通道 : 开盘价 最高价 最低价 收盘价 均价(元) 成交量(股) 成交额(元) 换手率(%)
  资金 slow 5 字段 : 全市场融资余额 全市场融券余额 全市场融资净买入额 板块总市值 板块上市股票数
  情绪 IF 1 字段  : IF_会员多空比（前20会员持买单量 / 持卖单量）
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import socket
import sys
import time

socket.setdefaulttimeout(40)
import pandas as pd
import requests
import akshare as ak

BASE = _P.ROOT
WIN_S, WIN_E = "2026-09-01", "2026-10-23"     # 对外接口的抓取窗口
SEP = "=" * 108
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://quote.eastmoney.com/"}


def hdr(s):
    print()
    print(SEP)
    print(f"【{s}】", flush=True)


def rd(fp):
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(fp)


# 结果收集：字段 -> dict(匹配天, 比对天, 最大相对误差, 说明)
RES = []


def record(field, src, proj, note=""):
    """proj / src 都是按日期索引的 Series"""
    j = pd.concat([src.rename("src"), proj.rename("proj")], axis=1).dropna()
    if j.empty:
        RES.append((field, "无重叠", 0, float("nan"), float("nan"), "日期对不上"))
        print(f"   ⚠ {field}: 无重叠日期")
        return
    rel = (j["src"] - j["proj"]).abs() / j["proj"].abs().replace(0, pd.NA)
    ab = (j["src"] - j["proj"]).abs()
    mx_rel = float(rel.max()) if rel.notna().any() else float("nan")
    mx_abs = float(ab.max())
    n_ok = int((rel.fillna(0) < 1e-6).sum())
    flag = "✅" if mx_rel < 1e-4 else ("🟡" if mx_rel < 1e-2 else "❌")
    print(f"   {flag} {field:22} n={len(j):3d}  完全一致 {n_ok:3d}/{len(j):3d}  "
          f"最大相对误差 {mx_rel*100:.8f}%  最大绝对差 {mx_abs:.6g}  {note}")
    RES.append((field, src.name if src.name else "", len(j), mx_rel, mx_abs, note))


# ================================================================ 读项目数据
hdr("0. 读取项目现有数据")
p = rd(f"{BASE}\\input\\中证1000_000852.SH.csv")
p["dt"] = pd.to_datetime(p["日期"]).dt.strftime("%Y-%m-%d")
p = p.set_index("dt").sort_index()
print(f"   中证1000 主数据 {p.shape}  {p.index.min()} ~ {p.index.max()}")

mrg = rd(f"{BASE}\\data\\两融数据.csv")
mrg["dt"] = pd.to_datetime(mrg["Mtdate"]).dt.strftime("%Y-%m-%d")

sz = rd(f"{BASE}\\data\\市场规模数据.csv")
sz["dt"] = pd.to_datetime(sz["SgnDate"]).dt.strftime("%Y-%m-%d")

fut = rd(f"{BASE}\\input\\情绪_期货会员持仓.csv")
print(f"   两融 {mrg.shape}  市场规模 {sz.shape}  期货持仓 {fut.shape}")
print(f"   期货持仓列 {list(fut.columns)}")

# 项目内部恒等式自检
hdr("0b. 项目 CSV 内部恒等式自检（用来判断免费源缺项能否推导）")
chk = mrg.pivot_table(index="dt", columns="Exchangecode", values="Financebalance", aggfunc="sum")
if {"SHSE", "SZSE"}.issubset(chk.columns):
    tot2 = chk["SHSE"] + chk["SZSE"]
    tot3 = tot2 + chk.get("BSE", 0)
    proj_tot = p["全市场融资余额(元)"]
    j2 = pd.concat([tot2.rename("沪深"), proj_tot.rename("项目")], axis=1).dropna().tail(30)
    print(f"   ① 全市场融资余额 = 沪+深 ?  末30日中位相对差 "
          f"{((j2['沪深']-j2['项目']).abs()/j2['项目']).median()*100:.6f}%")
    j3 = pd.concat([tot3.rename("沪深北"), proj_tot.rename("项目")], axis=1).dropna().tail(30)
    print(f"   ② 全市场融资余额 = 沪+深+北 ? 末30日中位相对差 "
          f"{((j3['沪深北']-j3['项目']).abs()/j3['项目']).median()*100:.6f}%")

net = p["全市场融资净买入额(元)"]
d_bal = p["全市场融资余额(元)"].diff()
j4 = pd.concat([d_bal.rename("Δ余额"), net.rename("净买入")], axis=1).dropna().tail(60)
diff = (j4["Δ余额"] - j4["净买入"]).abs()
print(f"   ③ 融资净买入额 = Δ融资余额 ?  末60日 绝对差 中位 {diff.median():.4g} 最大 {diff.max():.4g}")
print(f"      相对差 中位 {(diff/j4['净买入'].abs().replace(0,pd.NA)).median()*100:.6f}%")
mb = (p["全市场融资余额(元)"] + p["全市场融券余额(元)"])
j5 = pd.concat([mb.rename("融资+融券"), p["全市场两融余额(元)"].rename("项目")], axis=1).dropna()
print(f"   ④ 全市场两融余额 = 融资余额 + 融券余额 ? 最大相对差 "
      f"{((j5['融资+融券']-j5['项目']).abs()/j5['项目']).max()*100:.8f}%")

# ================================================================ 1. 量价
hdr("1. 量价 8 通道：东财 http K线 vs 项目 CSV")
j = requests.get("http://push2his.eastmoney.com/api/qt/stock/kline/get",
                 params=dict(secid="1.000852", fields1="f1,f2,f3,f4,f5,f6",
                             fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                             klt=101, fqt=1, beg="20260801", end="20500101", lmt=10000),
                 headers=H, timeout=25).json()
kl = [x.split(",") for x in j["data"]["klines"]]
em = pd.DataFrame(kl, columns=["日期", "开", "收", "高", "低", "量_手", "额", "振幅",
                               "涨跌幅", "涨跌额", "换手"]).set_index("日期")
for c in em.columns:
    em[c] = pd.to_numeric(em[c], errors="coerce")
em["量_股"] = em["量_手"] * 100
em["均价"] = em["额"] / em["量_股"]
print(f"   东财 {len(em)} 根  {em.index.min()} ~ {em.index.max()}")

record("开盘价", em["开"], p["开盘价"])
record("最高价", em["高"], p["最高价"])
record("最低价", em["低"], p["最低价"])
record("收盘价", em["收"], p["收盘价"])
record("成交量(股)", em["量_股"], p["成交量(股)"], "东财单位=手，×100")
record("成交额(元)", em["额"], p["成交额(元)"])
record("换手率(%)", em["换手"], p["换手率(%)"], "东财只给2位小数")
record("均价(元)", em["均价"], p["均价(元)"], "推导=额/量")

# ================================================================ 2. 两融
hdr(f"2. 两融：akshare 沪/深 vs 项目 CSV（{WIN_S} ~ {WIN_E}）")
sse = ak.stock_margin_sse(start_date=WIN_S.replace("-", ""), end_date=WIN_E.replace("-", ""))
sse["dt"] = pd.to_datetime(sse["信用交易日期"].astype(str)).dt.strftime("%Y-%m-%d")
sse = sse.set_index("dt").sort_index()
print(f"   上交所 {len(sse)} 天  {sse.index.min()} ~ {sse.index.max()}")

sh_proj = mrg[mrg["Exchangecode"] == "SHSE"].set_index("dt")
record("沪市融资余额", sse["融资余额"].astype(float), sh_proj["Financebalance"].astype(float))
record("全市场融资买入额(沪部分)", sse["融资买入额"].astype(float),
       sh_proj["Financeamount"].astype(float), "仅用于拆解核对")
record("沪市融券余额", sse["融券余量金额"].astype(float), sh_proj["Shortbalance"].astype(float))
record("沪市两融余额", sse["融资融券余额"].astype(float), sh_proj["Mtbalance"].astype(float))

sz_proj = mrg[mrg["Exchangecode"] == "SZSE"].set_index("dt")
days = [d for d in sse.index if d <= WIN_E]
sz_rows = {}
for d in days:
    try:
        z = ak.stock_margin_szse(date=d.replace("-", ""))
        sz_rows[d] = z.iloc[0]
    except Exception as e:
        print(f"      {d} 深交所失败 {type(e).__name__}: {str(e)[:50]}")
    time.sleep(0.4)
szdf = pd.DataFrame(sz_rows).T.sort_index()
print(f"   深交所 {len(szdf)} 天")
if not szdf.empty:
    record("深市融资余额", szdf["融资余额"].astype(float) * 1e8,
           sz_proj["Financebalance"].astype(float), "单位亿元×1e8")
    record("深市融券余额", szdf["融券余额"].astype(float) * 1e8,
           sz_proj["Shortbalance"].astype(float), "单位亿元×1e8")
    record("深市两融余额", szdf["融资融券余额"].astype(float) * 1e8,
           sz_proj["Mtbalance"].astype(float), "单位亿元×1e8")

# 全市场 = 沪 + 深（+北）
if not szdf.empty:
    tot_fin = (sse["融资余额"].astype(float)
               + pd.to_numeric(szdf["融资余额"], errors="coerce") * 1e8).dropna()
    record("全市场融资余额(缺北交所)", tot_fin, p["全市场融资余额(元)"], "沪+深，不含北交所")
    tot_sh = (sse["融券余量金额"].astype(float)
              + pd.to_numeric(szdf["融券余额"], errors="coerce") * 1e8).dropna()
    record("全市场融券余额(缺北交所)", tot_sh, p["全市场融券余额(元)"], "沪+深，不含北交所")

# ================================================================ 3. 市场规模
hdr(f"3. 市场规模：交易所每日统计 vs 项目 CSV（{WIN_S} ~ {WIN_E}，口径 code=11 沪深股票）")
print("   抓取中（沪深各一请求/天）...", flush=True)
sse_d, szse_d = {}, {}
for d in days:
    d8 = d.replace("-", "")
    try:
        x = ak.stock_sse_deal_daily(date=d8)
        r = x[x["单日情况"] == "挂牌数"]
        mv = x[x["单日情况"] == "市价总值"]
        sse_d[d] = dict(挂牌数=float(r.iloc[0]["股票"]), 市价总值=float(mv.iloc[0]["股票"]))
    except Exception as e:
        print(f"      {d} 上交所失败 {type(e).__name__}: {str(e)[:40]}")
    try:
        y = ak.stock_szse_summary(date=d8)
        r2 = y[y["证券类别"] == "股票"]
        szse_d[d] = dict(数量=float(r2.iloc[0]["数量"]),
                         总市值=float(r2.iloc[0]["总市值"]) / 1e8)
    except Exception as e:
        print(f"      {d} 深交所失败 {type(e).__name__}: {str(e)[:40]}")
    time.sleep(0.4)

sd = pd.DataFrame(sse_d).T.sort_index()
zd = pd.DataFrame(szse_d).T.sort_index()
print(f"   上交所 {len(sd)} 天，深交所 {len(zd)} 天")

# 项目 CSV 的 code=11
c11 = sz[sz["DataSgnCode"] == 11].set_index("dt").sort_index()
c1 = sz[sz["DataSgnCode"] == 1].set_index("dt").sort_index()
c5 = sz[sz["DataSgnCode"] == 5].set_index("dt").sort_index()
record("沪 市价总值(code1)", sd["市价总值"], c1["MarketValue"].astype(float), "亿元")
record("深 总市值(code5)", zd["总市值"], c5["MarketValue"].astype(float), "亿元")
record("沪 挂牌数(code1)", sd["挂牌数"], c1["StockNum"].astype(float), "家")
record("深 数量(code5)", zd["数量"], c5["StockNum"].astype(float), "家")

if not sd.empty and not zd.empty:
    tot_mv = (sd["市价总值"].astype(float) + zd["总市值"].astype(float)).dropna()
    record("板块总市值", tot_mv, c11["MarketValue"].astype(float), "沪+深 = code11")
    tot_n = (sd["挂牌数"].astype(float) + zd["数量"].astype(float)).dropna()
    record("板块上市股票数", tot_n, c11["StockNum"].astype(float), "沪+深 = code11")
    # 也核对主数据文件里的两份冗余列
    record("板块总市值(主数据)", tot_mv, p["板块总市值(亿元)"], "同一数值在主数据里的副本")
    record("板块上市股票数(主数据)", tot_n, p["板块上市股票数"], "同一数值在主数据里的副本")

# ================================================================ 4. 情绪 IF 会员多空比
hdr(f"4. 情绪：中金所 IF 会员多空比 vs 项目 CSV（{WIN_S} ~ {WIN_E}）")
fut["dt"] = pd.to_datetime(fut["日期"]).dt.strftime("%Y-%m-%d")
f_if = fut[fut["品种"] == "IF"].set_index("dt").sort_index()
print(f"   项目 CSV IF 行 {len(f_if)}  列 {list(fut.columns)}")
print(f"   项目 CSV 末 3 行：")
print(f_if.tail(3).to_string())

cffex = {}
for d in days:
    d8 = d.replace("-", "")
    try:
        obj = ak.get_cffex_rank_table(date=d8, vars_list=["IF"])
        if not isinstance(obj, dict) or not obj:
            print(f"      {d} 返回空")
            continue
        lo = sum(float(v["long_open_interest"].sum()) for v in obj.values())
        so = sum(float(v["short_open_interest"].sum()) for v in obj.values())
        cffex[d] = dict(前20_持买单量=lo, 前20_持卖单量=so,
                        会员多空比=lo / so if so else float("nan"), 合约数=len(obj))
    except Exception as e:
        print(f"      {d} 中金所失败 {type(e).__name__}: {str(e)[:50]}")
    time.sleep(0.4)

cf = pd.DataFrame(cffex).T.sort_index()
print(f"   抓取 {len(cf)} 天")
if not cf.empty:
    record("前20_持买单量", cf["前20_持买单量"].astype(float),
           f_if["前20_持买单量"].astype(float), "")
    record("前20_持卖单量", cf["前20_持卖单量"].astype(float),
           f_if["前20_持卖单量"].astype(float), "")
    record("IF_会员多空比", cf["会员多空比"].astype(float),
           f_if["会员多空比"].astype(float), "逐合约求和后相除")
    record("合约条数", cf["合约数"].astype(float), f_if["合约条数"].astype(float), "")

# ================================================================ 汇总
hdr("汇总：每个入模字段的爬取一致性")
print(f"   {'字段':28} {'比对天':>6} {'最大相对误差':>16} {'最大绝对差':>16}  说明")
print("   " + "-" * 100)
for field, _src, n, rel, ab, note in RES:
    if isinstance(n, str):
        print(f"   {field:28} {'-':>6} {'-':>16} {'-':>16}  {note}")
        continue
    print(f"   {field:28} {n:6d} {rel*100:15.8f}% {ab:16.6g}  {note}")
print()
print("   ✅ = 最大相对误差 < 0.01%（万分之一门槛）   🟡 = < 1%   ❌ = ≥ 1%")
print()
print(SEP)
print("完成")
