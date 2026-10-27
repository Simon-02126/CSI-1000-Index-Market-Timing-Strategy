# -*- coding: utf-8 -*-
"""抓取情绪面数据（期权 PCR / 期权 IV / 股指期货会员持仓）

用法:
    python fetch_sentiment.py stats 20150209 20261024
    python fetch_sentiment.py risk  20150209 20180930
    python fetch_sentiment.py cffex 20150209 20201231

产出中间分片到 _senti_parts/，用 merge_sentiment.py 合并成 input/ 下的最终文件。
特点: 断点续跑（已有日期自动跳过）、失败重试、限流
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
import akshare as ak  # noqa: E402

BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
PART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_senti_parts")
os.makedirs(PART, exist_ok=True)

JOB = sys.argv[1]
START, END = sys.argv[2], sys.argv[3]


def trading_dates(start, end):
    fp = os.path.join(INPUT, "中证1000_000852.SH.csv")
    df = pd.read_csv(fp, encoding="gbk", usecols=[0])
    d = pd.to_datetime(df.iloc[:, 0], errors="coerce").dropna()
    d = d[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))]
    return d.dt.strftime("%Y%m%d").sort_values().tolist()


def part_file(job):
    return os.path.join(PART, f"{job}_{START}_{END}.csv")


def load_done(fp):
    if not os.path.exists(fp):
        return set()
    try:
        return set(pd.read_csv(fp, encoding="utf-8-sig", usecols=["日期"])["日期"].astype(str))
    except Exception:
        return set()


def append(fp, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    hdr = not os.path.exists(fp)
    df.to_csv(fp, mode="a", header=hdr, index=False, encoding="utf-8-sig")


# ---------------- 期权每日统计（PCR） ----------------
def fetch_stats(d):
    x = ak.option_daily_stats_sse(date=d)
    if x is None or len(x) == 0:
        return []
    out = []
    for _, r in x.iterrows():
        call_vol = float(r["认购成交量"])
        put_vol = float(r["认沽成交量"])
        call_oi = float(r["未平仓认购合约数"])
        put_oi = float(r["未平仓认沽合约数"])
        out.append({
            "日期": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "标的代码": str(r["合约标的代码"]),
            "标的名称": str(r["合约标的名称"]),
            "合约数量": float(r["合约数量"]),
            "总成交额_万元": float(r["总成交额"]),
            "总成交量": float(r["总成交量"]),
            "认购成交量": call_vol,
            "认沽成交量": put_vol,
            "PCR_成交_认沽比认购": put_vol / call_vol if call_vol > 0 else np.nan,
            "未平仓合约总数": float(r["未平仓合约总数"]),
            "认购未平仓": call_oi,
            "认沽未平仓": put_oi,
            "PCR_持仓_认沽比认购": put_oi / call_oi if call_oi > 0 else np.nan,
        })
    return out


# ---------------- 期权风险指标（隐含波动率） ----------------
def fetch_risk(d):
    x = ak.option_risk_indicator_sse(date=d)
    if x is None or len(x) == 0:
        return []
    x = x.copy()
    # CONTRACT_ID 例: 510050C2506A02500 -> 标的 510050, 类型 C/P
    cid = x["CONTRACT_ID"].astype(str)
    x["标的代码"] = cid.str[:6]
    x["类型"] = cid.str[6:7].str.upper()          # C=认购 P=认沽
    x["IV"] = pd.to_numeric(x["IMPLC_VOLATLTY"], errors="coerce")
    x["DELTA"] = pd.to_numeric(x["DELTA_VALUE"], errors="coerce")
    x = x[(x["IV"] > 0) & x["IV"].notna() & x["DELTA"].notna()]
    if len(x) == 0:
        return []
    out = []
    for sid, g in x.groupby("标的代码"):
        call = g[g["类型"] == "C"]
        put = g[g["类型"] == "P"]
        # 虚值认购: 0.15<=delta<=0.35 ; 平值: 0.45<=delta<=0.55 ; 虚值认沽: -0.35<=delta<=-0.15
        otm_c = call[(call["DELTA"] >= 0.15) & (call["DELTA"] <= 0.35)]["IV"]
        atm_c = call[(call["DELTA"] >= 0.45) & (call["DELTA"] <= 0.55)]["IV"]
        atm_p = put[(put["DELTA"] >= -0.55) & (put["DELTA"] <= -0.45)]["IV"]
        otm_p = put[(put["DELTA"] >= -0.35) & (put["DELTA"] <= -0.15)]["IV"]
        iv_otm_c = otm_c.median() if len(otm_c) else np.nan
        iv_atm = np.nanmedian([atm_c.median() if len(atm_c) else np.nan,
                               atm_p.median() if len(atm_p) else np.nan])
        iv_otm_p = otm_p.median() if len(otm_p) else np.nan
        out.append({
            "日期": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "标的代码": sid,
            "IV_平值": iv_atm,
            "IV_虚值认购": iv_otm_c,
            "IV_虚值认沽": iv_otm_p,
            "IV_认沽认购比": (iv_otm_p / iv_otm_c) if (iv_otm_c and iv_otm_c > 0) else np.nan,
            "IV_偏斜_虚沽减平值": (iv_otm_p - iv_atm) if pd.notna(iv_otm_p) and pd.notna(iv_atm) else np.nan,
            "有效合约数": int(len(g)),
        })
    return out


# ---------------- 股指期货会员持仓 ----------------
def fetch_cffex(d):
    dd = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    try:
        tab = ak.get_cffex_rank_table(date=d, vars_list=["IF", "IC", "IM", "IH"])
    except Exception:
        return []
    if not tab:
        return []
    rows = []
    agg = {}
    for sym, df_ in tab.items():
        if df_ is None or len(df_) == 0:
            continue
        var = sym[:2]
        a = agg.setdefault(var, {"long": 0.0, "short": 0.0, "n": 0})
        d20 = df_.head(20)
        a["long"] += pd.to_numeric(d20["long_open_interest"], errors="coerce").sum()
        a["short"] += pd.to_numeric(d20["short_open_interest"], errors="coerce").sum()
        a["n"] += len(d20)
    for var, a in agg.items():
        rows.append({
            "日期": dd,
            "品种": var,
            "前20_持买单量": a["long"],
            "前20_持卖单量": a["short"],
            "会员多空比": a["long"] / a["short"] if a["short"] > 0 else np.nan,
            "合约条数": a["n"],
        })
    return rows


FN = {"stats": fetch_stats, "risk": fetch_risk, "cffex": fetch_cffex}
SLEEP = {"stats": 0.15, "risk": 0.0, "cffex": 0.25}

dates = trading_dates(START, END)
fp = part_file(JOB)
done = load_done(fp)
todo = [d for d in dates if d not in done]
print(f"[{JOB}] {START}~{END} 交易日 {len(dates)}，已完成 {len(done)}，待抓 {len(todo)}", flush=True)

buf, ok, fail = [], 0, []
t0 = time.time()
for i, d in enumerate(todo, 1):
    for attempt in range(3):
        try:
            rows = FN[JOB](d)
            buf += rows
            ok += 1
            break
        except Exception as e:
            if attempt == 2:
                fail.append(d)
            else:
                time.sleep(1.5 * (attempt + 1))
    if i % 100 == 0 or i == len(todo):
        append(fp, buf)
        buf = []
        el = time.time() - t0
        print(f"  {i}/{len(todo)} ok={ok} fail={len(fail)} {el:.0f}s "
              f"eta={el/max(i,1)*(len(todo)-i):.0f}s", flush=True)
    time.sleep(SLEEP[JOB])
append(fp, buf)

with open(os.path.join(PART, f"{JOB}_{START}_{END}_failed.txt"), "w") as f:
    f.write("\n".join(fail))
print(f"[{JOB}] 完成 ok={ok} fail={len(fail)} 输出={fp}", flush=True)
