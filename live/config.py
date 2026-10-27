# -*- coding: utf-8 -*-
"""实盘配置（唯一出处）。

改标的、改账户、改调度时间，都只改这里。
"""
from __future__ import annotations

import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths as _P  # noqa: E402

# ---------------------------------------------------------------- 标的
INDEX_NAME = "中证1000"
INDEX_CODE = "000852"
EASTMONEY_SECID = "1.000852"        # 东财行情：0.399xxx=深证, 1.000xxx=上证, 2.932xxx=中证2000
ETF_CODE = "512100"                 # 512100 中证1000ETF（现金替代/小资金）

# ---------------------------------------------------------------- 账户
INIT_CAPITAL = 1_000_000.0          # 初始模拟资金（元）
ONE_WAY_COST = 0.0003               # 单边成本（回测口径 3bp）
SLIPPAGE = 0.0000                   # 额外滑点（默认 0，需要时调）

# ---------------------------------------------------------------- 策略参数（引用 core，勿在此覆盖）
# 网格 / 死区 / 分位窗口等一律以 core/strategy.py 为准，实盘不要另设一套。

# ---------------------------------------------------------------- 时间
BOOTSTRAP_START = "2019-06-04"      # 模拟盘回填起点（= 回测起点）
EXEC_AT_OPEN = True                 # True: T 日收盘出信号 → T+1 开盘成交（唯一可执行口径）

# ---------------------------------------------------------------- 每日调度（本地时间，用于文档与提示）
SCHEDULE = [
    ("15:05", "量价（东财 http K 线）"),
    ("15:40", "情绪（中金所会员持仓）"),
    ("20:00", "市场规模（交易所统计）"),
    ("21:00", "两融（交易所披露，T+1 挂网）→ 数据齐备后出信号"),
    ("T+1 09:15~09:25", "集合竞价按目标仓位下单"),
]

# ---------------------------------------------------------------- 状态文件
LEDGER_CSV = os.path.join(_P.STATE, "ledger.csv")          # 模拟盘账本
SIGNAL_JSON = os.path.join(_P.STATE, "signals")            # 每日信号快照
SIGNALS_LOG = os.path.join(_P.STATE, "signals_log.csv")    # 逐日信号明细（base/A0/目标）
EQUITY_CSV = os.path.join(_P.STATE, "equity.csv")          # 逐日净值
REPORT_PNG = os.path.join(_P.STATE, "模拟盘净值.png")
REPORT_MD = os.path.join(_P.STATE, "模拟盘日报.md")

for _p in (SIGNAL_JSON,):
    os.makedirs(_p, exist_ok=True)
