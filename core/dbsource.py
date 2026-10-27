# -*- coding: utf-8 -*-
"""数据库数据源（`db/timing.db`）—— 本项目的唯一事实来源。

设计要点
--------
* **只读**打开（`file:...?mode=ro`），不会因为回测/训练而改动库。
* 复现主数据 CSV 的**同日口径**（`trade_date`）：`index_daily` 取价格量额，
  `margin_daily` / `market_size_daily` 按 `date == T` 直取。
  （`features_main` 是 `effective_date <= T` 的 as-of 口径，比同日口径**晚一个交易日**，
   那是给审计/看板用的，不要拿来喂模型，否则等于多滞后一天。）
* 列名与 `input/中证1000_000852.SH.csv` 完全一致，调用方无感切换。
* 库不存在或表为空时 `available()` 返回 False，调用方回落到 CSV。

用法
----
    import dbsource as dbs
    if dbs.available():
        df = dbs.load_features()      # 列名同 CSV，DatetimeIndex 名为「日期」
        ifr = dbs.load_if_ratio()     # 情绪 IF 多空比（if_sentiment）
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

# ---- 路径自举（与 core 其它模块一致：向上找 _paths.py）
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
if _d not in sys.path:
    sys.path.insert(0, _d)
import _paths as _P  # noqa: E402

DB_FP = os.path.join(_P.ROOT, "db", "timing.db")
INDEX_CODE = "000852.SH"

#: 与主数据 CSV 同序的 27 个特征列（日期另列）
FEAT_COLS = ["开盘价", "最高价", "最低价", "收盘价", "换手率(%)", "成交量(股)", "成交额(元)", "均价(元)",
             "沪市融资余额(元)", "深市融资余额(元)", "全市场融资余额(元)",
             "全市场融资买入额(元)", "全市场融资净买入额(元)", "全市场融券余额(元)",
             "沪市两融余额(元)", "深市两融余额(元)", "全市场两融余额(元)",
             "板块上市公司数", "板块上市股票数", "板块总股本(亿股)", "板块流通股本(亿股)",
             "板块总市值(亿元)", "板块流通市值(亿元)",
             "板块成交量(亿股)", "板块成交额(亿元)", "板块换手率(%)", "板块流通换手率(%)"]


def db_path(fp: str | None = None) -> str:
    return fp or DB_FP


def connect(fp: str | None = None) -> sqlite3.Connection:
    """只读连接。"""
    return sqlite3.connect(f"file:{db_path(fp)}?mode=ro", uri=True)


def available(fp: str | None = None) -> bool:
    """库存在且 `index_daily` 有 final 行。"""
    fp = db_path(fp)
    if not os.path.isfile(fp):
        return False
    try:
        con = connect(fp)
        n = con.execute("SELECT COUNT(*) FROM index_daily "
                        "WHERE code=? AND status='final'", (INDEX_CODE,)).fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


def _q(con, sql, *args) -> pd.DataFrame:
    cur = con.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def load_raw(fp: str | None = None, status: str | None = "final"):
    """原始三表（同日口径，未做任何滞后）。

    status='final' 只取定稿行（默认）；None 表示 final+provisional 都取（盘后试算用）。
    """
    con = connect(fp)
    if status == "final":
        idx = _q(con, "SELECT * FROM index_daily WHERE code=? AND status='final' ORDER BY date", INDEX_CODE)
    else:
        # provisional 优先（同一天既有 provisional 又有 final 时取 provisional，用于盘中试算）
        idx = _q(con, "SELECT * FROM index_daily WHERE code=? ORDER BY date, "
                      "CASE status WHEN 'provisional' THEN 0 ELSE 1 END", INDEX_CODE)
        idx = idx.drop_duplicates("date", keep="first")
    mg = _q(con, "SELECT * FROM margin_daily WHERE exch='ALL' ORDER BY date")
    sz = _q(con, "SELECT * FROM market_size_daily WHERE sgn_code=11 ORDER BY date")
    con.close()
    for df, col in ((idx, "date"), (mg, "date"), (mg, "effective_date"), (sz, "date"), (sz, "effective_date")):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return idx, mg, sz


def _derive_index(idx: pd.DataFrame) -> pd.DataFrame:
    """补齐东财 K 线缺的均价/换手率（与 live/db_build.py 同一套派生规则）。"""
    if idx.empty:
        return idx
    idx = idx.copy()
    if "vwap" in idx.columns:
        idx["vwap"] = idx["vwap"].fillna(idx["amount"] / idx["volume"].where(idx["volume"] > 0))
    has = idx.dropna(subset=["turnover", "volume"]) if "turnover" in idx.columns else idx.iloc[0:0]
    if len(has):
        last = has.iloc[-1]
        fs = float(last["volume"]) / (float(last["turnover"]) / 100.0)
        if fs > 0:
            idx["turnover"] = idx["turnover"].fillna(idx["volume"] / fs * 100.0)
    return idx


def load_features(fp: str | None = None, status: str | None = "final") -> pd.DataFrame:
    """主数据（28 列，列名与 CSV 一致，DatetimeIndex 名「日期」，同日口径）。"""
    idx, mg, sz = load_raw(fp, status=status)
    if idx.empty:
        return pd.DataFrame(columns=["日期"] + FEAT_COLS).set_index("日期")
    idx = _derive_index(idx)
    dates = pd.DatetimeIndex(idx["date"])

    mg_td = mg.drop_duplicates("date").set_index("date").reindex(dates) if len(mg) else pd.DataFrame(index=dates)
    sz_td = sz.drop_duplicates("date").set_index("date").reindex(dates) if len(sz) else pd.DataFrame(index=dates)

    out = pd.DataFrame(index=dates)
    for cn, src in zip(["开盘价", "最高价", "最低价", "收盘价", "换手率(%)",
                        "成交量(股)", "成交额(元)", "均价(元)"],
                       ["open", "high", "low", "close", "turnover", "volume", "amount", "vwap"]):
        out[cn] = pd.to_numeric(idx[src], errors="coerce").to_numpy() if src in idx.columns else np.nan

    def _g(df, col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=dates)

    fin_bal = _g(mg_td, "fin_balance")
    out["全市场融资余额(元)"] = fin_bal.to_numpy()
    out["全市场融资买入额(元)"] = _g(mg_td, "fin_buy").to_numpy()
    # 净买入 = 余额差分（与训练口径一致）
    out["全市场融资净买入额(元)"] = fin_bal.diff().to_numpy()
    short_bal = _g(mg_td, "short_balance")
    out["全市场融券余额(元)"] = short_bal.to_numpy()
    out["全市场两融余额(元)"] = (fin_bal + short_bal).to_numpy()

    out["板块上市股票数"] = _g(sz_td, "stock_num").to_numpy()
    out["板块总市值(亿元)"] = _g(sz_td, "mkt_value").to_numpy()

    out.index.name = "日期"
    return out.reindex(columns=FEAT_COLS)


def load_if_ratio(fp: str | None = None) -> pd.Series:
    """情绪 IF 会员多空比（`if_sentiment`，同日口径）。"""
    con = connect(fp)
    try:
        df = _q(con, "SELECT date, ratio FROM if_sentiment WHERE variety='IF' ORDER BY date")
    finally:
        con.close()
    if df.empty:
        return pd.Series(dtype=float, name="IF_会员多空比")
    s = pd.Series(pd.to_numeric(df["ratio"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df["date"]), name="IF_会员多空比")
    return s.dropna()


def nav_ledger(fp: str | None = None) -> pd.DataFrame:
    """账本（只读），列名与 `state/ledger.csv` 语义一致。"""
    con = connect(fp)
    try:
        df = _q(con, "SELECT * FROM v_nav ORDER BY exec_date")
    finally:
        con.close()
    if not df.empty:
        df["exec_date"] = pd.to_datetime(df["exec_date"])
        df = df.set_index("exec_date")
    return df


if __name__ == "__main__":
    if not available():
        print(f"DB 不可用: {db_path()}")
        raise SystemExit(1)
    f = load_features()
    print(f"DB {db_path()}")
    print(f"  load_features  {f.index.min().date()} ~ {f.index.max().date()}  {len(f)} 行 × {len(f.columns)} 列")
    for c in ["收盘价", "全市场融资余额(元)", "板块总市值(亿元)", "板块上市股票数"]:
        s = f[c].dropna()
        print(f"    {c:<16} 非空 {len(s):>5}  首 {s.iloc[0]:>18,.2f}  末 {s.iloc[-1]:>18,.2f}")
    ifr = load_if_ratio()
    print(f"  IF 多空比       {ifr.index.min().date()} ~ {ifr.index.max().date()}  {len(ifr)} 行  末值 {ifr.iloc[-1]:.5f}")
    nv = nav_ledger()
    if len(nv):
        print(f"  账本            {nv.index.min().date()} ~ {nv.index.max().date()}  {len(nv)} 行  末净值 {float(nv['nav'].iloc[-1]):.4f}")
