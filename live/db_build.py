# -*- coding: utf-8 -*-
"""从 raw 层重建派生数据:
  1. features_main: 按"可获取日期"(effective_date) 对齐的入模真相 (审计/看板用)
     date=T 行 -> 各源 effective_date<=T 的最新值; 净买入=该对齐序列差分
  2. state/work/features.csv: 模型引擎工作文件 (trade_date 对齐, GBK 28列)
     与训练口径一致 (模型内部自己做滞后), 属临时计算产物, 每次重建覆盖
幂等: 全量重建。用法: python live/db_build.py
"""
from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

if not (isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding == "utf-8"):
    # 防重入: 被替换的旧 wrapper 析构会关闭底层 buffer, 重复包裹会让 print 失效
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
BASE = _d
DB = os.path.join(BASE, "db", "timing.db")
WORK = os.path.join(BASE, "state", "work", "features.csv")

FEAT_COLS = ["开盘价", "最高价", "最低价", "收盘价", "换手率(%)", "成交量(股)", "成交额(元)", "均价(元)",
             "沪市融资余额(元)", "深市融资余额(元)", "全市场融资余额(元)",
             "全市场融资买入额(元)", "全市场融资净买入额(元)", "全市场融券余额(元)",
             "沪市两融余额(元)", "深市两融余额(元)", "全市场两融余额(元)",
             "板块上市公司数", "板块上市股票数", "板块总股本(亿股)", "板块流通股本(亿股)",
             "板块总市值(亿元)", "板块流通市值(亿元)",
             "板块成交量(亿股)", "板块成交额(亿元)", "板块换手率(%)", "板块流通换手率(%)"]


def load_raw():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    idx = pd.DataFrame([dict(r) for r in con.execute(
        "SELECT * FROM index_daily WHERE code='000852.SH' AND status='final' ORDER BY date")])
    mg = pd.DataFrame([dict(r) for r in con.execute(
        "SELECT * FROM margin_daily WHERE exch='ALL' ORDER BY date")])
    sz = pd.DataFrame([dict(r) for r in con.execute(
        "SELECT * FROM market_size_daily WHERE sgn_code=11 ORDER BY date")])
    con.close()
    for df, col in ((idx, "date"), (mg, "date"), (mg, "effective_date"), (sz, "date"), (sz, "effective_date")):
        df[col] = pd.to_datetime(df[col])
    # ---- 东财指数K线缺换手率/均价(个股才有), 派生补齐:
    #   均价   = 成交额/成交量 (定义式, 精确)
    #   换手率 = 成交量/流通股本×100, 流通股本用"最近一个有换手率的交易日"反推
    #   (中证1000 流通股本近似恒定 1e12 股, 日际漂移<0.1%, 经 x/mean(x) 标准化后可忽略)
    idx["vwap"] = idx["vwap"].fillna(
        idx["amount"] / idx["volume"].where(idx["volume"] > 0))
    has_to = idx.dropna(subset=["turnover", "volume"])
    if len(has_to):
        last = has_to.iloc[-1]
        float_shares = float(last["volume"]) / (float(last["turnover"]) / 100.0)
        idx["turnover"] = idx["turnover"].fillna(
            idx["volume"] / float_shares * 100.0)
        print(f"    指数换手率派生补齐 (流通股本={float_shares:.3e} 股, "
              f"参考日 {last['date'].date()})")
    return idx, mg, sz


def asof_take(src: pd.DataFrame, dates: pd.DatetimeIndex, val_cols: list[str]) -> pd.DataFrame:
    """取 src 中 effective_date <= T 的各 val_cols 最新值, 对齐到 dates"""
    out = pd.DataFrame(index=dates)
    for c in val_cols:
        s = src.dropna(subset=[c]).drop_duplicates("effective_date").set_index("effective_date")[c]
        out[c] = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    return out


def main():
    idx, mg, sz = load_raw()
    dates = pd.DatetimeIndex(idx["date"])
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(WORK), exist_ok=True)

    # ---- 对齐视图 (features_main): 两融/规模 取 effective<=T 最新
    mg_al = asof_take(mg, dates, ["fin_balance", "short_balance", "fin_buy"])
    mg_al["fin_net"] = mg_al["fin_balance"].diff()
    sz_al = asof_take(sz, dates, ["stock_num", "mkt_value"])

    # ---- trade_date 视图 (工作CSV): 两融/规模 按 date=T 直取 (模型内部自带滞后)
    mg_td = mg.set_index("date").reindex(dates)[["fin_balance", "short_balance", "fin_buy"]]
    mg_td["fin_net"] = mg_td["fin_balance"].diff()
    sz_td = sz.drop_duplicates("date").set_index("date").reindex(dates)[["stock_num", "mkt_value"]]

    con = sqlite3.connect(DB)
    con.execute("BEGIN")
    con.execute("DELETE FROM features_main")
    n = 0
    for i, d in enumerate(dates):
        ds = d.strftime("%Y-%m-%d")
        r = idx.iloc[i]
        h = hashlib.sha256()
        h.update(str(tuple(r.values)).encode())
        h.update(str(tuple(mg_al.iloc[i].values)).encode())
        h.update(str(tuple(sz_al.iloc[i].values)).encode())
        con.execute(f"""INSERT INTO features_main
            (date,{','.join('"' + c + '"' for c in FEAT_COLS)},build_hash,built_at)
            VALUES ({','.join('?' * (len(FEAT_COLS) + 3))})""",
            (ds, r["open"], r["high"], r["low"], r["close"], r["turnover"], r["volume"],
             r["amount"], r["vwap"],
             None, None, mg_al["fin_balance"].iloc[i], mg_al["fin_buy"].iloc[i],
             mg_al["fin_net"].iloc[i], mg_al["short_balance"].iloc[i],
             None, None, None,
             None, sz_al["stock_num"].iloc[i], None, None,
             sz_al["mkt_value"].iloc[i], None, None, None, None, None,
             h.hexdigest()[:16], now))
        n += 1
    con.execute("COMMIT")
    con.close()

    # ---- 工作 CSV (模型引擎读): 28 列, GBK, 列名与原主数据 CSV 完全一致
    work = pd.DataFrame({"日期": [d.strftime("%Y-%m-%d") for d in dates]})
    for cn, src_c in zip(FEAT_COLS[:8], ["open", "high", "low", "close", "turnover", "volume", "amount", "vwap"]):
        work[cn] = idx[src_c].to_numpy()
    work["全市场融资余额(元)"] = mg_td["fin_balance"].to_numpy()
    work["全市场融资买入额(元)"] = mg_td["fin_buy"].to_numpy()
    work["全市场融资净买入额(元)"] = mg_td["fin_net"].to_numpy()
    work["全市场融券余额(元)"] = mg_td["short_balance"].to_numpy()
    work["板块上市股票数"] = sz_td["stock_num"].to_numpy()
    work["板块总市值(亿元)"] = sz_td["mkt_value"].to_numpy()
    work = work.reindex(columns=["日期"] + FEAT_COLS)
    work.to_csv(WORK, index=False, encoding="gbk")

    print(f"features_main 重建 {n} 行 ({dates[0].date()} ~ {dates[-1].date()})")
    print(f"工作CSV -> {WORK} ({len(work)} 行)")
    # 对齐缺口提示: 末日仍缺两融/规模的行数
    lack = int(mg_al["fin_balance"].isna().sum())
    print(f"对齐后两融缺失行: {lack} (早期历史无两融, 正常)")


if __name__ == "__main__":
    main()
