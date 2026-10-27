# -*- coding: utf-8 -*-
"""一次性导入: 项目 CSV -> timing.db
统一截止 CUTOFF (默认 2026-10-20), 之后日期留给 fetch 链路做测试窗口。
来源全部取自本项目内部文件 (自包含原则):
  index_daily      input/中证1000_000852.SH.csv (GBK, 28列)     status=final, source=csmar_merged
  margin_daily     同上 (全市场两融3列 -> exch='ALL'), effective_date=次交易日
  market_size_daily data/市场规模数据.csv (全口径, 单位: 亿/万股), effective_date=次交易日
  cffex_rank_raw   input/情绪_期货会员持仓.csv -> 聚合行 contract='ALL' rank=999
  if_sentiment     同上物化 (含合约条数)
  features_main    input 主数据 CSV 28 列 + build_hash(当日raw行哈希)
  emotion_panel    input/情绪_面板_日频.csv (33列)
  ledger           state/ledger.csv (append-only 历史)
  signal_freeze    state/signals/*.json (仅 <= CUTOFF)
  model_files      扫描 models/**/*.pt
幂等: 重复运行先 DELETE 再插入, 不产生重复行。
用法: python db/import_csv.py [--cutoff 2026-10-20]
"""
import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
BASE = _d
DB = os.path.join(BASE, "db", "timing.db")
CUTOFF_DEFAULT = "2026-10-20"


def connect():
    con = sqlite3.connect(DB)
    con.executescript(open(os.path.join(BASE, "db", "schema.sql"), encoding="utf-8").read())
    return con


def next_trade_day(dates_index: pd.DatetimeIndex, d: pd.Timestamp) -> str:
    nxt = dates_index[dates_index > d]
    return (nxt[0] if len(nxt) else d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def table_report(con, name):
    n, mn, mx = con.execute(
        f"SELECT COUNT(*), MIN(date), MAX(date) FROM {name}").fetchone() \
        if name != "ledger" else con.execute(
        "SELECT COUNT(*), MIN(exec_date), MAX(exec_date) FROM ledger").fetchone()
    print(f"  {name:20s} {n:6d} 行  {mn} ~ {mx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=CUTOFF_DEFAULT)
    a = ap.parse_args()
    cutoff = a.cutoff
    t0 = time.time()

    main_fp = os.path.join(BASE, "input", "中证1000_000852.SH.csv")
    df = pd.read_csv(main_fp, encoding="gbk")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] <= cutoff].sort_values("日期").reset_index(drop=True)
    cal = pd.DatetimeIndex(df["日期"])
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    con = connect()
    cur = con.cursor()

    # ---------- 清空 (幂等重跑)
    for t in ["index_daily", "margin_daily", "market_size_daily", "cffex_rank_raw",
              "if_sentiment", "features_main", "emotion_panel", "ledger",
              "signal_freeze", "model_files"]:
        cur.execute(f"DELETE FROM {t}")

    # ---------- 1. index_daily
    rows = [( "000852.SH", r["日期"].strftime("%Y-%m-%d"), r["开盘价"], r["最高价"], r["最低价"],
              r["收盘价"], r["成交量(股)"], r["成交额(元)"], r["换手率(%)"], r["均价(元)"],
              "final", "csmar_merged", now, now) for _, r in df.iterrows()]
    cur.executemany("""INSERT OR REPLACE INTO index_daily
        (code,date,open,high,low,close,volume,amount,turnover,vwap,status,source,fetched_at,finalized_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"index_daily: {len(rows)} 行 (<= {cutoff})")

    # ---------- 2. margin_daily (exch='ALL', 净买入不存)
    rows = []
    for _, r in df.iterrows():
        d = r["日期"]
        rows.append((d.strftime("%Y-%m-%d"), "ALL", r["全市场融资余额(元)"],
                     r["全市场融券余额(元)"], r["全市场融资买入额(元)"],
                     next_trade_day(cal, d), "csmar_merged", now))
    cur.executemany("""INSERT OR REPLACE INTO margin_daily
        (date,exch,fin_balance,short_balance,fin_buy,effective_date,source,fetched_at)
        VALUES (?,?,?,?,?,?,?,?)""", rows)
    print(f"margin_daily: {len(rows)} 行")

    # ---------- 3. market_size_daily (全口径, data/市场规模数据.csv)
    ms = pd.read_csv(os.path.join(BASE, "data", "市场规模数据.csv"), encoding="utf-8-sig")
    ms["SgnDate"] = pd.to_datetime(ms["SgnDate"])
    ms = ms[ms["SgnDate"] <= cutoff]
    def clean(v):
        return None if (v is None or (not isinstance(v, str) and pd.isna(v))) else v

    rows = []
    for _, r in ms.iterrows():
        d = r["SgnDate"]
        rows.append((d.strftime("%Y-%m-%d"), int(r["DataSgnCode"]), str(r["DataSgn"]),
                     clean(r.get("ListedCoNum")), clean(r.get("StockNum")), clean(r.get("TotalShares")),
                     clean(r.get("CirculatedShares")), clean(r.get("MarketValue")),
                     clean(r.get("CirculatedMktValue")),
                     clean(r.get("TurnoverRate1")), clean(r.get("TurnoverRate2")),
                     next_trade_day(cal, d), "csmar_archive", now))
    cur.executemany("""INSERT OR REPLACE INTO market_size_daily
        (date,sgn_code,sgn_name,listed_co,stock_num,total_shares,circ_shares,
         mkt_value,circ_value,turnover1,turnover2,effective_date,source,fetched_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"market_size_daily: {len(rows)} 行 ({ms['DataSgnCode'].nunique()} 个口径)")

    # ---------- 4/5. cffex_rank_raw (聚合行) + if_sentiment
    rk = pd.read_csv(os.path.join(BASE, "input", "情绪_期货会员持仓.csv"), encoding="utf-8-sig")
    rk["日期"] = pd.to_datetime(rk["日期"])
    rk = rk[rk["日期"] <= cutoff]
    raw_rows, sent_rows = [], []
    for _, r in rk.iterrows():
        d = r["日期"].strftime("%Y-%m-%d")
        raw_rows.append((d, r["品种"], "ALL", 999, r["前20_持买单量"], r["前20_持卖单量"], now))
        sent_rows.append((d, r["品种"], r["前20_持买单量"], r["前20_持卖单量"],
                          r["会员多空比"], int(r["合约条数"])))
    cur.executemany("""INSERT OR REPLACE INTO cffex_rank_raw
        (date,variety,contract,rank,long_open_interest,short_open_interest,fetched_at)
        VALUES (?,?,?,?,?,?,?)""", raw_rows)
    cur.executemany("""INSERT OR REPLACE INTO if_sentiment
        (date,variety,buy_top20,sell_top20,ratio,n_contracts) VALUES (?,?,?,?,?,?)""", sent_rows)
    print(f"cffex_rank_raw / if_sentiment: {len(raw_rows)} 行 ({rk['品种'].nunique()} 个品种)")

    # ---------- 6. features_main (28列 + build_hash)
    feat_cols = [c for c in df.columns if c != "日期"]
    rows = []
    ms11 = ms[ms["DataSgnCode"] == 11].set_index("SgnDate")
    mg = {r["日期"]: r for _, r in df.iterrows()}
    for _, r in df.iterrows():
        d = r["日期"]
        h = hashlib.sha256()
        h.update(str(tuple(r.values)).encode())
        h.update(str(tuple(ms11.loc[d].values)).encode() if d in ms11.index else b"")
        vals = [d.strftime("%Y-%m-%d")] + [r[c] for c in feat_cols] + [h.hexdigest()[:16], now]
        rows.append(tuple(vals))
    cur.executemany(f"""INSERT OR REPLACE INTO features_main
        (date,{','.join('"' + c + '"' for c in feat_cols)},build_hash,built_at)
        VALUES ({','.join('?' * (len(feat_cols) + 3))})""", rows)
    print(f"features_main: {len(rows)} 行 x {len(feat_cols)} 列")

    # ---------- 7. emotion_panel (33列)
    ep = pd.read_csv(os.path.join(BASE, "input", "情绪_面板_日频.csv"), encoding="utf-8-sig")
    ep["日期"] = pd.to_datetime(ep["日期"], errors="coerce")
    ep = ep[ep["日期"].notna() & (ep["日期"] <= cutoff)].sort_values("日期")
    ep_cols = list(ep.columns)
    rows = [tuple([r["日期"].strftime("%Y-%m-%d")] + [clean(r[c]) for c in ep_cols[1:]])
            for _, r in ep.iterrows()]
    cur.executemany(f"""INSERT OR REPLACE INTO emotion_panel
        (date,{','.join('"' + c + '"' for c in ep_cols[1:])})
        VALUES ({','.join('?' * len(ep_cols))})""", rows)
    print(f"emotion_panel: {len(rows)} 行 x {len(ep_cols)} 列")

    # ---------- 8. ledger (append-only 历史)
    led = pd.read_csv(os.path.join(BASE, "state", "ledger.csv"), encoding="utf-8-sig",
                      parse_dates=["日期", "信号日"])
    led = led[led["日期"] <= cutoff]
    rows = [(r["日期"].strftime("%Y-%m-%d"),
             r["信号日"].strftime("%Y-%m-%d") if pd.notna(r["信号日"]) else None,
             r["目标仓位"], r["成交价"], r["收盘价"], r["持仓比例"], r["份额"], r["现金"],
             r["权益"], r["净值"], r["日收益"], r["成本"], None) for _, r in led.iterrows()]
    cur.executemany("""INSERT INTO ledger
        (exec_date,signal_date,target_pos,exec_price,mark_price,position,shares,cash,
         equity,nav,daily_ret,cost,strategy_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"ledger: {len(rows)} 行 (起点 {rows[0][0]})")

    # ---------- 9. signal_freeze (仅 <= cutoff 的已有快照)
    n_sig = 0
    sig_dir = os.path.join(BASE, "state", "signals")
    if os.path.isdir(sig_dir):
        for fn in sorted(os.listdir(sig_dir)):
            if not fn.endswith(".json"):
                continue
            d = fn[:-5]
            if d > cutoff:
                print(f"  跳过 {fn} (> {cutoff}, 留给实盘链路测试)")
                continue
            j = json.load(open(os.path.join(sig_dir, fn), encoding="utf-8"))
            cur.execute("""INSERT OR REPLACE INTO signal_freeze
                (signal_date,base,if_pct,a0,target_pos,input_hash,model_hash,frozen_at,invalidated)
                VALUES (?,?,?,?,?,?,?,?,0)""",
                        (j.get("日期", d), j.get("base"), j.get("IF分位"), j.get("A0"),
                         j.get("目标仓位"), None, None, j.get("请求日", d)))
            n_sig += 1
    print(f"signal_freeze: {n_sig} 条")

    # ---------- 10. model_files (扫描权重)
    n_m = 0
    for view in ("pv", "slow"):
        vdir = os.path.join(BASE, "models", view)
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".pt"):
                continue
            parts = fn[:-3].split("_")
            # 形如 pv_2026_s0
            year, seed = int(parts[1]), parts[2]
            fp = os.path.join(vdir, fn)
            md5 = hashlib.md5(open(fp, "rb").read()).hexdigest()
            cur.execute("""INSERT OR REPLACE INTO model_files
                (view,year,seed,file_path,train_window_hash,train_data_end,trained_at)
                VALUES (?,?,?,?,?,?,?)""",
                        (view, year, seed, os.path.relpath(fp, BASE), md5, None,
                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(fp)))))
            n_m += 1
    print(f"model_files: {n_m} 个检查点")

    con.commit()

    # ---------- 验收自检
    print("\n===== 验收自检 =====")
    bad = 0
    date_col = {"signal_freeze": "signal_date"}
    for t in ["index_daily", "margin_daily", "market_size_daily", "cffex_rank_raw",
              "if_sentiment", "features_main", "emotion_panel", "signal_freeze"]:
        dc = date_col.get(t, "date")
        mx = con.execute(f"SELECT MAX({dc}) FROM {t}").fetchone()[0]
        if mx and mx > cutoff:
            print(f"  [FAIL] {t} 末日 {mx} 超过 cutoff"); bad += 1
    mx = con.execute("SELECT MAX(exec_date) FROM ledger").fetchone()[0]
    if mx and mx > cutoff:
        print(f"  [FAIL] ledger 末日 {mx} 超过 cutoff"); bad += 1
    # v_features_final 行数 == features_main 行数 (全 final)
    n1 = con.execute("SELECT COUNT(*) FROM features_main").fetchone()[0]
    n2 = con.execute("SELECT COUNT(*) FROM v_features_final").fetchone()[0]
    print(f"  v_features_final {n2} / features_main {n1} 行 {'OK' if n1 == n2 else 'FAIL'}")
    bad += 0 if n1 == n2 else 1
    # v_if_ratio 与 if_sentiment 一致 (重叠日期)
    n3 = con.execute("""SELECT COUNT(*) FROM v_if_ratio v JOIN if_sentiment s
        ON v.date=s.date AND v.variety=s.variety AND ABS(v.ratio-s.ratio)>1e-9""").fetchone()[0]
    print(f"  v_if_ratio vs if_sentiment 不一致行: {n3} {'OK' if n3 == 0 else 'FAIL'}")
    bad += n3
    # margin 余额差分(前一交易日) == features 净买入, 抽 3 日
    cal_days = [r[0] for r in con.execute(
        "SELECT date FROM index_daily WHERE code='000852.SH' ORDER BY date").fetchall()]
    for d in ["2026-10-15", "2026-10-16", "2026-10-20"]:
        i = cal_days.index(d)
        b0 = con.execute("SELECT fin_balance FROM margin_daily WHERE date=? AND exch='ALL'",
                         (cal_days[i - 1],)).fetchone()[0]
        b1 = con.execute("SELECT fin_balance FROM margin_daily WHERE date=? AND exch='ALL'",
                         (d,)).fetchone()[0]
        nz = con.execute('SELECT "全市场融资净买入额(元)" FROM features_main WHERE date=?',
                         (d,)).fetchone()[0]
        ok = abs((b1 - b0) - nz) < max(abs(nz) * 0.01, 1e-6)
        print(f"  净买入自洽 {d}: 差分 {b1-b0:,.0f} vs 存值 {nz:,.0f} {'OK' if ok else 'FAIL'}")
        bad += 0 if ok else 1
    print(f"\n库文件: {DB} ({os.path.getsize(DB)/1e6:.1f} MB), 用时 {time.time()-t0:.1f}s")
    print("总结:", "全部通过 ✓" if bad == 0 else f"{bad} 项失败 ✗")
    con.close()


if __name__ == "__main__":
    main()
