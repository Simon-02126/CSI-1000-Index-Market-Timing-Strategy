# -*- coding: utf-8 -*-
"""db_update: 看板「更新交易」按钮背后的一键编排。

链路: (可选抓数) -> db_build 重建特征 -> 计算对齐日 T_max ->
      生成/冻结缺口日信号 (archive 历史 + 2026 模型尾段推理) ->
      按 ETF 开盘价逐日记账 (与 live/portfolio.append_day 同口径) ->
      打印人类日志, 最后一行输出 JSON 报告 (看板消费)。

口径要点 (已与用户对齐):
  * 对齐日 T_max = min(指数final末日, IF末日, 两融max(effective_date), 规模sgn11 max(effective_date))
    —— 两融 T 日生效 T+1 披露, 按"可获取日期"与其他数据对齐, 无未来函数。
  * 信号日 S 用 S 日收盘数据生成 -> T+1(下一交易日) 以 512100 开盘价成交, 收盘价盯市。
  * 首个换汇建仓日: 上一账本行的 equity 全部转现金, 按当日开盘价建仓。
  * 信号冻结与记账在同一个事务里, 失败整体回滚。

用法:
  python live/db_update.py                       # 增量续跑 (特征重建+缺口信号+记账)
  python live/db_update.py --fetch               # 先抓数再增量续跑
  python live/db_update.py --rebuild-oos [--fetch]  # 样本外重建: 信号/账本从 2026 年初
                                                   # 起点按初始资金空仓建仓 (ETF 口径), 清旧账
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import sys
import time
import traceback

import numpy as np
import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
BASE = _d
for _p in (BASE, os.path.join(BASE, "core"), os.path.join(BASE, "live")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import live.config as C                      # noqa: E402
import core.datafeed as dfd                  # noqa: E402
import core.strategy as st                   # noqa: E402
import core.model_io as mio                  # noqa: E402
from live import db_fetch, db_build          # noqa: E402

# db_fetch/db_build 在 import 时会确保 stdout 为 utf-8 wrapper；这里再保险一次，
# 防重入（被替换的旧 wrapper 析构会关闭底层 buffer，重复包裹会让 print 失效）。
if not (isinstance(sys.stdout, io.TextIOWrapper) and sys.stdout.encoding == "utf-8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
_STDOUT_KEEP = sys.stdout          # 防 GC 析构关 buffer

DB = db_build.DB
WORK = db_build.WORK
ETF = C.ETF_CODE + ".SH"
IDX = C.INDEX_CODE + ".SH"
TAIL_YEAR = 2026                             # 模型逐年末重训, 尾年信号靠推理延长
STRATEGY_HASH = "db_update_v1"


# ---------------------------------------------------------------- 对齐日
def t_max(con: sqlite3.Connection) -> str:
    """各源可获取口径的最小末日 = 信号日上限。均取 ISO 日期字符串。"""
    d_idx = con.execute("SELECT MAX(date) FROM index_daily WHERE code=? AND status='final'",
                        (IDX,)).fetchone()[0]
    d_if = con.execute("SELECT MAX(date) FROM if_sentiment WHERE variety='IF'").fetchone()[0]
    d_mg = con.execute("SELECT MAX(effective_date) FROM margin_daily WHERE exch='ALL'").fetchone()[0]
    d_sz = con.execute("SELECT MAX(effective_date) FROM market_size_daily WHERE sgn_code=11"
                       ).fetchone()[0]
    return min(x for x in (d_idx, d_if, d_mg, d_sz) if x)


# ---------------------------------------------------------------- 信号
def build_signal_frame(con: sqlite3.Connection) -> pd.DataFrame:
    """组装 pv/slow/if 全序列并算出 base/IF分位/A0/目标仓位 (position_series 口径)。"""
    print("[3/5] 组装模型信号 (archive 历史 + %d 年尾段推理)..." % TAIL_YEAR)
    raw = dfd.build_raw_signals()                       # archive 权威历史
    tails = {}
    for view in ("pv", "slow"):
        tail = mio.predict_series(None, TAIL_YEAR, view)
        keep = tail[tail.index > raw[view].index.max()]  # archive 优先, 只补尾部缺口
        tails[view] = pd.concat([raw[view], keep]).sort_index()
        print(f"    {view}: archive 至 {raw[view].index.max().date()}, "
              f"尾段补 {len(keep)} 日至 {tails[view].index.max().date()}")
    if_raw = pd.Series(
        [r[0] for r in con.execute("SELECT ratio FROM if_sentiment WHERE variety='IF' ORDER BY date")],
        index=pd.to_datetime([r[0] for r in con.execute(
            "SELECT date FROM if_sentiment WHERE variety='IF' ORDER BY date")]),
        name="if", dtype=float)
    pos = st.position_series(tails["pv"], tails["slow"], if_raw)
    print(f"    仓位序列 {pos.index[0].date()} ~ {pos.index[-1].date()} 共 {len(pos)} 行")
    return pos


def model_hash_2026() -> str:
    """当年 12 个模型文件的 md5 指纹, 写入 freeze.model_hash。"""
    h = hashlib.md5()
    for view in ("pv", "slow"):
        for seed in range(6):
            fp = mio.model_path(TAIL_YEAR, view, seed)
            if os.path.isfile(fp):
                h.update(os.path.basename(fp).encode())
                with open(fp, "rb") as f:
                    h.update(f.read(1 << 20))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    do_fetch = "--fetch" in argv
    rebuild_oos = "--rebuild-oos" in argv   # 样本外重建: 信号/账本从 2026 年初重来, 起点建仓
    rep = {"ok": False, "fetched": do_fetch, "signals_frozen": [], "fills": [],
           "pending": [], "T_max": None}
    t0 = time.time()
    try:
        if do_fetch:
            print("[1/5] 抓取最新数据 (东财/交易所/中金所)...")
            db_fetch.main()
        else:
            print("[1/5] 跳过抓数 (--fetch 可开启)")

        print("[2/5] 重建 features_main 与工作 CSV...")
        db_build.main()

        con = sqlite3.connect(DB)
        con.row_factory = sqlite3.Row
        tmax = t_max(con)
        rep["T_max"] = tmax
        print(f"    对齐日 T_max = {tmax}")

        pos = build_signal_frame(con)
        trade_dates = [d.strftime("%Y-%m-%d") for d in pos.index]

        # ---- 起始状态: 续跑读上一账本行; --rebuild-oos 从样本外起点重来
        if rebuild_oos:
            oos_start = next((d for d in trade_dates if d >= f"{TAIL_YEAR}-01-01"), None)
            if not oos_start:
                raise RuntimeError(f"仓位序列中没有 {TAIL_YEAR} 年的数据, 无法重建样本外")
            i0 = trade_dates.index(oos_start)
            if i0 == 0:
                raise RuntimeError("样本外起点前无交易日, 无法生成建仓信号")
            sig_start = trade_dates[i0 - 1]          # 样本外前一生成日的信号 -> 首个样本外交易日开盘建仓
            last_sig = trade_dates[i0 - 2] if i0 >= 2 else ""   # 冻结范围: sig_start ~ T_max
            cur, shares_p, cash_p, nav_p = 0.0, 0.0, C.INIT_CAPITAL, 1.0
            print(f"    [重建] 样本外起点={oos_start}, 建仓信号日={sig_start}, "
                  f"初始资金={C.INIT_CAPITAL:,.0f} (仓位从 0 起步)")
        else:
            led_last = con.execute(
                "SELECT exec_date, signal_date, target_pos, shares, cash, equity, nav, strategy_hash "
                "FROM ledger ORDER BY exec_date DESC LIMIT 1").fetchone()
            if led_last is None:
                raise RuntimeError("ledger 为空: 请先 --rebuild-oos 重建样本外账本")
            last_sig = led_last["signal_date"]
            cur = float(led_last["target_pos"])      # 实盘 walk: 当前实际持仓=上一日目标
            nav_p = float(led_last["nav"])
            if led_last["strategy_hash"] == STRATEGY_HASH:
                shares_p, cash_p = float(led_last["shares"]), float(led_last["cash"])
            else:
                # 换汇建仓日: 旧账本行是"指数口径"存档, 份额/现金不可沿用,
                # 首日将全部权益转为现金, 按当日 ETF 开盘价重新建仓
                shares_p, cash_p = 0.0, float(led_last["equity"])
                print(f"    注意: 账本末日 {led_last['exec_date']} 为指数口径存档, "
                      f"{last_sig} 之后的首次执行将换汇建仓 (现金={cash_p:.0f})")

        # ---- ETF 交易日历 (final  bar)
        etf = {r["date"]: dict(open=r["open"], close=r["close"], status=r["status"])
               for r in con.execute("SELECT date, open, close, status FROM etf_daily WHERE code=?",
                                    (ETF,))}

        mhash = model_hash_2026()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        frozen = set(r[0] for r in con.execute("SELECT signal_date FROM signal_freeze"))

        con.execute("BEGIN")
        try:
            if rebuild_oos:
                con.execute("DELETE FROM ledger")
                con.execute("DELETE FROM signal_freeze WHERE signal_date >= ?", (sig_start,))
                print(f"    [重建] 已清空 ledger, 并清除 {sig_start} 起的旧冻结信号")

            # ---- [4/5] 冻结缺口日信号: 从上一信号日+1 走到 T_max
            print(f"[4/5] 冻结信号 {last_sig} 之后至 {tmax} ...")
            for S in trade_dates:
                if S <= last_sig or S > tmax:
                    continue
                a0 = float(pos.loc[pd.Timestamp(S), "A0"])
                tgt = st.next_target(a0, cur)
                bh = con.execute("SELECT build_hash FROM features_main WHERE date=?", (S,)) \
                        .fetchone()
                con.execute("""INSERT OR REPLACE INTO signal_freeze
                    (signal_date, base, if_pct, a0, target_pos, input_hash, model_hash,
                     frozen_at, invalidated)
                    VALUES (?,?,?,?,?,?,?,?,0)""",
                    (S, float(pos.loc[pd.Timestamp(S), "base"]),
                     float(pos.loc[pd.Timestamp(S), "IF分位"]), a0, tgt,
                     bh["build_hash"] if bh else None, mhash, now))
                frozen.add(S)
                rep["signals_frozen"].append(
                    {"signal_date": S, "a0": round(a0, 4), "target_pos": tgt})
                print(f"    信号 {S}: base={float(pos.loc[pd.Timestamp(S), 'base']):.1f} "
                      f"A0={a0:+.4f} -> 目标 {tgt:.0%} "
                      f"({'由 ' + format(cur, '.0%') + ' 调整' if tgt != cur else '维持'})")
                cur = tgt

            # ---- [5/5] 执行: 每个新信号 S -> 下一交易日 E 以开盘价成交
            print("[5/5] 执行调仓 (ETF 开盘成交, 收盘盯市)...")
            new_signals = sorted(S for S in frozen if S > last_sig and S <= tmax)
            etf_dates = sorted(etf)                      # ETF 交易日历 (含 provisional)
            for S in new_signals:
                E = next((d for d in etf_dates if d > S), None)
                if E is None:
                    rep["pending"].append({"signal_date": S, "reason": "无下一交易日"})
                    continue
                bar = etf.get(E)
                if bar is None:
                    rep["pending"].append({"signal_date": S, "exec_date": E,
                                           "reason": "ETF 无行情"})
                    continue
                if bar["status"] != "final":
                    rep["pending"].append({"signal_date": S, "exec_date": E,
                                           "reason": f"ETF {bar['status']} 待收盘确认"})
                    continue
                if con.execute("SELECT 1 FROM ledger WHERE exec_date=?", (E,)).fetchone():
                    continue                                   # 幂等: 已记账
                tgt = float(con.execute("SELECT target_pos FROM signal_freeze WHERE signal_date=?",
                                        (S,)).fetchone()[0])
                open_px, close_px = float(bar["open"]), float(bar["close"])
                eq_open = cash_p + shares_p * open_px
                cur_open = shares_p * open_px / eq_open if eq_open > 0 else 0.0
                fee = abs(tgt - cur_open) * eq_open * C.ONE_WAY_COST
                shares = tgt * eq_open / open_px
                cash = cash_p + shares_p * open_px - shares * open_px - fee
                eq = cash + shares * close_px
                nav = eq / C.INIT_CAPITAL
                dret = nav / nav_p - 1 if nav_p else 0.0
                pos_after = shares * close_px / eq if eq else 0.0
                con.execute("""INSERT INTO ledger
                    (exec_date, signal_date, target_pos, exec_price, mark_price, position,
                     shares, cash, equity, nav, daily_ret, cost, strategy_hash)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (E, S, tgt, open_px, close_px, pos_after, shares, cash, eq,
                     nav, dret, fee, STRATEGY_HASH))
                rep["fills"].append({
                    "exec_date": E, "signal_date": S, "target_pos": tgt,
                    "exec_price": open_px, "shares_delta": round(shares - shares_p, 2),
                    "shares": round(shares, 2), "cash": round(cash, 2),
                    "equity": round(eq, 2), "nav": round(nav, 6),
                    "daily_ret": round(dret, 6), "cost": round(fee, 2)})
                print(f"    成交 {E} (信号{S}): 目标 {tgt:.0%} @ {open_px:.3f} "
                      f"Δ份额 {shares - shares_p:+.2f} 费 {fee:.2f} "
                      f"净值 {nav:.4f} ({dret:+.2%})")
                shares_p, cash_p, nav_p = shares, cash, nav
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        # ---- 汇总
        last = con.execute("SELECT nav, equity, position FROM ledger ORDER BY exec_date DESC LIMIT 1") \
                  .fetchone()
        rep["nav"] = round(last["nav"], 6)
        rep["equity"] = round(last["equity"], 2)
        rep["position"] = round(last["position"], 4)
        rep["ledger_last"] = con.execute("SELECT MAX(exec_date) FROM ledger").fetchone()[0]
        con.close()
        rep["ok"] = True
        rep["seconds"] = round(time.time() - t0, 1)
        print(f"完成: 冻结 {len(rep['signals_frozen'])} 日信号, 成交 {len(rep['fills'])} 笔, "
              f"待执行 {len(rep['pending'])} 笔, 净值 {rep['nav']:.4f}, 耗时 {rep['seconds']}s")
    except Exception as e:                                      # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        print("失败:", rep["error"])
        traceback.print_exc()
        print(json.dumps(rep, ensure_ascii=False))               # 失败也输出 JSON
        return 1
    print("__REPORT__" + json.dumps(rep, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
