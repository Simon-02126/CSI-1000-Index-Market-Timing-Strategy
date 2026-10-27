# -*- coding: utf-8 -*-
"""每日抓取 -> timing.db raw 层 (upsert + 状态机 + ingest_log)
与 fetch_daily.py 的通道一致 (2026-10 实测可用), 但写入 SQLite:
  量价   东财 http K线: 指数 000852 (secid 1.000852) + ETF 512100 (secid 1.512100)
  两融   akshare stock_margin_account_info 全市场 (亿元 -> 元), effective_date=T+1
  规模   沪 stock_sse_deal_daily + 深 stock_szse_summary 全口径
  IF     get_cffex_rank_table, rank=999 合计行 -> cffex_rank_raw + if_sentiment
状态机: 抓取日 == 今天 且当前 < 15:10 -> provisional; 否则 final。
        final 永远覆盖 provisional / 同日值变化记 values_changed。
幂等: 重复运行不产生重复行; ingest_log 每次一行。
用法: python live/db_fetch.py            # 抓最近可得
      python live/db_fetch.py 2026-10-26  # 抓到指定日 (回补历史)
"""
from __future__ import annotations

import io
import json
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
SCHEMA = os.path.join(BASE, "db", "schema.sql")

INDEX_CODE, INDEX_SECID = "000852.SH", "1.000852"
ETF_CODE, ETF_SECID = "512100.SH", "1.512100"
FINALIZE_AFTER = 1510          # 15:10 后抓当日 bar 视为 final
S = ("指数", "ETF", "两融", "规模", "IF")


def connect():
    con = sqlite3.connect(DB)
    con.executescript(open(SCHEMA, encoding="utf-8").read())
    con.isolation_level = None      # 手动事务
    return con


def is_final_bar(d: str) -> bool:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    if d < today:
        return True
    if d > today:
        return False
    return int(time.strftime("%H%M")) >= FINALIZE_AFTER


def upsert_bars(con, table: str, code: str, df: pd.DataFrame) -> tuple[int, int]:
    """东财K线 -> {table}; 返回 (新增, 更新)"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur = con.cursor()
    n_new = n_upd = 0
    for _, r in df.iterrows():
        d = r["日期"].strftime("%Y-%m-%d")
        st = "final" if is_final_bar(d) else "provisional"
        fin_at = now if st == "final" else None
        old = cur.execute(f"SELECT status FROM {table} WHERE code=? AND date=?",
                          (code, d)).fetchone()
        if old is None:
            cur.execute(f"""INSERT INTO {table}
                (code,date,open,high,low,close,volume,amount,status,source,fetched_at,finalized_at)
                VALUES (?,?,?,?,?,?,?,?,?,'eastmoney',?,?)""",
                        (code, d, r["open"], r["high"], r["low"], r["close"],
                         r["volume"], r["amount"], st, now, fin_at))
            n_new += 1
        else:
            if old[0] == "final" and st == "provisional":
                continue                       # provisional 不覆盖 final
            cur.execute(f"""UPDATE {table} SET open=?,high=?,low=?,close=?,volume=?,
                            amount=?,status=?,source='eastmoney',
                            finalized_at=COALESCE(finalized_at,?) WHERE code=? AND date=?""",
                        (r["open"], r["high"], r["low"], r["close"], r["volume"],
                         r["amount"], st, fin_at, code, d))
            n_upd += 1
    return n_new, n_upd


def fetch_eastmoney_kline(secid: str, beg: str, end: str, allow_empty: bool = False) -> pd.DataFrame:
    import requests
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    p = {"secid": secid, "klt": 101, "fqt": 0, "beg": beg.replace("-", ""),
         "end": end.replace("-", ""), "fields1": "f1,f2,f3,f4,f5,f6",
         "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"}
    js = requests.get(url, params=p, timeout=30,
                      headers={"User-Agent": "Mozilla/5.0"}).json()
    kl = (js.get("data") or {}).get("klines") or []
    if not kl:
        if allow_empty:
            return pd.DataFrame(columns=["日期", "open", "close", "high", "low", "volume", "amount"])
        raise RuntimeError(f"东财无数据 secid={secid} ({beg}~{end})")
    rows = []
    for ln in kl:
        f = ln.split(",")
        rows.append({"日期": pd.Timestamp(f[0]), "open": float(f[1]), "close": float(f[2]),
                     "high": float(f[3]), "low": float(f[4]),
                     "volume": float(f[5]) * 100, "amount": float(f[6])})
    return pd.DataFrame(rows)


def fetch_margin(con) -> tuple[int, int, int]:
    """全市场两融(akshare汇总表) -> margin_daily exch='ALL'; 返回 (行, 末trade_date, values_changed)"""
    import akshare as ak
    mg = ak.stock_margin_account_info()
    mg["日期"] = pd.to_datetime(mg["日期"])
    # 只维护近90天窗口: 更早的历史以 CSMAR 底档为准, 避免双源数值抖动误报 values_changed
    mg = mg[mg["日期"] >= pd.Timestamp.today() - pd.Timedelta(days=90)]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur = con.cursor()
    # 交易日历: 用 index_daily
    cal = [r[0] for r in cur.execute(
        "SELECT date FROM index_daily WHERE code=? ORDER BY date", (INDEX_CODE,)).fetchall()]
    cal = pd.DatetimeIndex(cal)

    def eff(d: pd.Timestamp) -> str:
        nxt = cal[cal > d]
        return (nxt[0] if len(nxt) else d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    n = chg = 0
    for _, r in mg.iterrows():
        d = r["日期"]
        ds = d.strftime("%Y-%m-%d")
        fb = float(r["融资余额"]) * 1e8
        sb = float(r["融券余额"]) * 1e8
        by = float(r["融资买入额"]) * 1e8
        old = cur.execute("SELECT fin_balance,short_balance,fin_buy FROM margin_daily WHERE date=? AND exch='ALL'",
                          (ds,)).fetchone()
        if old is None:
            cur.execute("""INSERT INTO margin_daily
                (date,exch,fin_balance,short_balance,fin_buy,effective_date,source,fetched_at)
                VALUES (?,'ALL',?,?,?,?, 'akshare', ?)""",
                        (ds, fb, sb, by, eff(d), now))
            n += 1
        elif (old[0], old[1], old[2]) != (fb, sb, by):
            cur.execute("""UPDATE margin_daily SET fin_balance=?,short_balance=?,fin_buy=?,
                           source='akshare',fetched_at=? WHERE date=? AND exch='ALL'""",
                        (fb, sb, by, now, ds))
            n += 1
            chg = 1
    mx = cur.execute("SELECT MAX(date) FROM margin_daily WHERE exch='ALL'").fetchone()[0]
    return n, mx, chg


def fetch_market_size(con) -> int:
    """交易所统计 -> market_size_daily (1=沪股票 5=深股票 8=创业板 11=沪深加总)"""
    import akshare as ak
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur = con.cursor()
    cal = pd.DatetimeIndex([r[0] for r in cur.execute(
        "SELECT date FROM index_daily WHERE code=? ORDER BY date", (INDEX_CODE,)).fetchall()])

    def eff(d: pd.Timestamp) -> str:
        nxt = cal[cal > d]
        return (nxt[0] if len(nxt) else d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    today = pd.Timestamp.today().normalize()
    have = pd.Timestamp(cur.execute("SELECT MAX(date) FROM market_size_daily WHERE sgn_code=11").fetchone()[0] or "2000-01-01")
    days = [d for d in cal if d > have - pd.Timedelta(days=1) and d <= today]
    if not days:
        days = [cal[-1]]
    n = 0
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        ds2 = d.strftime("%Y%m%d")          # 交易所接口要求 YYYYMMDD
        try:
            t = ak.stock_sse_deal_daily(date=ds2).set_index("单日情况")
            sh_n = float(t.loc["挂牌数", "股票"]); sh_mv = float(t.loc["市价总值", "股票"])
        except Exception:
            continue
        try:
            z = ak.stock_szse_summary(date=ds2)
            z_all = z[z["证券类别"] == "股票"].iloc[0]
            z_cyb = z[z["证券类别"] == "创业板A股"].iloc[0]
            sz_n = float(z_all["数量"]); sz_mv = float(z_all["总市值"]) / 1e8
            cyb_n = float(z_cyb["数量"]); cyb_mv = float(z_cyb["总市值"]) / 1e8
        except Exception:
            continue
        rows = [(ds, 1, "沪股票", sh_n, sh_mv), (ds, 5, "深股票", sz_n, sz_mv),
                (ds, 8, "创业板", cyb_n, cyb_mv), (ds, 11, "沪深股票", sh_n + sz_n, sh_mv + sz_mv)]
        for ds2, code, name, num, mv in rows:
            cur.execute("""INSERT INTO market_size_daily
                (date,sgn_code,sgn_name,stock_num,mkt_value,effective_date,source,fetched_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(date,sgn_code) DO UPDATE SET
                stock_num=excluded.stock_num, mkt_value=excluded.mkt_value,
                source=excluded.source, fetched_at=excluded.fetched_at""",
                        (ds2, code, name, num, mv, eff(d), "exchange", now))
            n += 1
        time.sleep(0.4)
    return n


def fetch_if(con) -> int:
    import akshare as ak
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cur = con.cursor()
    mx = cur.execute("SELECT MAX(date) FROM cffex_rank_raw").fetchone()[0] or "2000-01-01"
    cal = pd.DatetimeIndex([r[0] for r in cur.execute(
        "SELECT date FROM index_daily WHERE code=? ORDER BY date", (INDEX_CODE,)).fetchall()])
    days = [d for d in cal if d.strftime("%Y-%m-%d") > mx and d <= pd.Timestamp.today().normalize()]
    n = 0
    for d in days:
        ds = d.strftime("%Y-%m-%d")
        try:
            t = ak.get_cffex_rank_table(date=ds.replace("-", ""), vars_list=["IF"])
            buy = sum(float(x.loc[x["rank"].astype(str) == "999", "long_open_interest"].iloc[0]) for x in t.values())
            sell = sum(float(x.loc[x["rank"].astype(str) == "999", "short_open_interest"].iloc[0]) for x in t.values())
            if not (buy > 0 and sell > 0):
                continue                       # 盘中/未挂网日数据不全, 跳过
        except Exception:
            continue
        cur.execute("""INSERT OR REPLACE INTO cffex_rank_raw
            (date,variety,contract,rank,long_open_interest,short_open_interest,fetched_at)
            VALUES (?,'IF','ALL',999,?,?,?)""", (ds, buy, sell, now))
        cur.execute("""INSERT OR REPLACE INTO if_sentiment
            (date,variety,buy_top20,sell_top20,ratio,n_contracts) VALUES (?,'IF',?,?,?,?)""",
                    (ds, buy, sell, buy / sell, 20 * len(t)))
        n += 1
        time.sleep(0.4)
    return n


def log(con, table, d_from, d_to, n_new, n_upd, changed, check):
    """写一条入库日志。

    区间倒置（`d_from > d_to`，即"库里已经是最新、今天没有新交易日"）时**不写**：
    那种情况没有发生任何抓取动作，写进去只会让 ingest_log 累积一堆
    `date_from=2026-10-28 / date_to=2026-10-27` 的假记录。调用方若要留痕，
    用 `check={"skip": "up-to-date"}` 打日志到 stdout 即可。
    """
    if d_from and d_to and str(d_from) > str(d_to):
        return
    con.execute("""INSERT INTO ingest_log
        (table_name,date_from,date_to,rows_new,rows_updated,values_changed,check_result,started_at,finished_at)
        VALUES (?,?,?,?,?,?,?,?,?)""",
                (table, d_from, d_to, n_new, n_upd, changed, json.dumps(check, ensure_ascii=False),
                 time.strftime("%Y-%m-%d %H:%M:%S"), time.strftime("%Y-%m-%d %H:%M:%S")))


def main(to_date: str | None = None):
    to = pd.Timestamp(to_date) if to_date else pd.Timestamp.today().normalize()
    con = connect()
    cur = con.cursor()
    report = {}
    try:
        con.execute("BEGIN")
        # 1. 量价: 指数 + ETF (从库内末日+1 抓到 to)
        for table, code, secid in (("index_daily", INDEX_CODE, INDEX_SECID),
                                   ("etf_daily", ETF_CODE, ETF_SECID)):
            mx = cur.execute(f"SELECT MAX(date) FROM {table} WHERE code=?", (code,)).fetchone()[0]
            if mx is None and table == "etf_daily":
                # 首次: 按年分段拉全历史 (512100 2013年上市), 防长区间截断
                frames = []
                for y in range(2013, to.year + 1):
                    fr = fetch_eastmoney_kline(
                        secid, f"{y}-01-01", min(to.strftime("%Y-%m-%d"), f"{y}-12-31"),
                        allow_empty=True)
                    if len(fr):
                        frames.append(fr)
                    time.sleep(0.5)
                df = pd.concat(frames, ignore_index=True).drop_duplicates("日期")
                start = "2013-01-01"
            else:
                start_ts = ((pd.Timestamp(mx) + pd.Timedelta(days=1)) if mx
                            else pd.Timestamp("2012-01-01"))
                if start_ts > to:
                    # 库内末日已是今天: 还要看它是 provisional 还是 final。
                    # 盘中(15:10 前)抓过的当日 bar 会一直是 provisional, 若只按日期判"已最新"
                    # 就永远不会被 final 覆盖 -> features_main 卡在昨天, 账本停摆。
                    st = cur.execute(
                        f"SELECT status FROM {table} WHERE code=? AND date=?",
                        (code, mx)).fetchone()
                    if st and st[0] != "final":
                        start_ts = pd.Timestamp(mx)          # 从当天起重抓, 让 final 覆盖
                        print(f"[{S[0] if table=='index_daily' else S[1]}] {code} "
                              f"末日 {mx} 仍是 {st[0]}, 重抓当日")
                    else:
                        # 真·已最新 (末日已定稿), 无新交易日可抓
                        log(con, table, mx, to.strftime("%Y-%m-%d"), 0, 0, 0,
                            {"rows": 0, "skip": "up-to-date"})
                        print(f"[{S[0] if table=='index_daily' else S[1]}] {code} 已最新 ({mx}), 跳过")
                        report[table] = {"new": 0, "upd": 0, "to": mx}
                        continue
                start = start_ts.strftime("%Y-%m-%d")
                df = fetch_eastmoney_kline(secid, start, to.strftime("%Y-%m-%d"))
            nn, nu = upsert_bars(con, table, code, df)
            log(con, table, start, to.strftime("%Y-%m-%d"), nn, nu, 0, {"rows": len(df)})
            print(f"[{S[0] if table=='index_daily' else S[1]}] {code} 新增{nn} 更新{nu} ({start}~{to.date()})")
            report[table] = {"new": nn, "upd": nu, "to": df["日期"].max().strftime("%Y-%m-%d")}

        # 2. 两融
        n, mx, chg = fetch_margin(con)
        log(con, "margin_daily", "", mx, n, 0, chg, {})
        print(f"[{S[2]}] 全市场 {n} 行, 末日 {mx}, values_changed={chg}")
        report["margin_daily"] = {"rows": n, "to": mx}

        # 3. 市场规模
        n = fetch_market_size(con)
        mx = cur.execute("SELECT MAX(date) FROM market_size_daily").fetchone()[0]
        log(con, "market_size_daily", "", mx, n, 0, 0, {})
        print(f"[{S[3]}] {n} 行, 末日 {mx}")
        report["market_size_daily"] = {"rows": n, "to": mx}

        # 4. IF 持仓
        n = fetch_if(con)
        mx = cur.execute("SELECT MAX(date) FROM if_sentiment").fetchone()[0]
        log(con, "cffex_rank_raw", "", mx, n, 0, 0, {})
        print(f"[{S[4]}] {n} 行, 末日 {mx}")
        report["if_sentiment"] = {"rows": n, "to": mx}

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    print(json.dumps({"ok": True, "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
