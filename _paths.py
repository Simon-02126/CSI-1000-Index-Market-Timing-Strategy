# -*- coding: utf-8 -*-
"""项目路径统一入口（唯一真相）

设计目标：整个文件夹复制到任何位置都能直接运行。
所有路径都由本文件所在目录（即项目根）推导，不含任何绝对路径。

项目分两块：
    core/   + live/     实盘与通用（保留维护）
    archive/            训练、回测、研究（归档留存，仍可运行）

用法：
    import _paths as _P
    _P.INPUT / _P.DATA / _P.OUTPUT         数据层
    _P.CORE / _P.LIVE / _P.ARCHIVE         代码块
    _P.MODELS / _P.STATE                   实盘模型与账本状态
    _P.TRAIN                               历史 6-seed 信号（实盘 bootstrap 用）
"""
import os
import sys

# 项目根 = 本文件所在目录
ROOT = os.path.dirname(os.path.abspath(__file__))


def _d(*parts):
    """返回项目根下的路径；若目录不存在则尝试创建（失败也不报错，只读介质可运行）。"""
    p = os.path.join(ROOT, *parts)
    try:
        os.makedirs(p, exist_ok=True)
    except OSError:
        pass
    return p


# —— 数据层 ——
INPUT = _d("input")                      # 中证1000 主数据、情绪原始表、情绪面板
DATA = _d("data")                        # 两融、市场规模
DATA_RAW = _d("data", "raw")             # 东财 K 线缓存等中间件
OUTPUT = _d("output")                    # 生产信号根目录
SIGNALS = _d("output", "signals")        # timing_model 默认输出目录
SIGNALS_V2 = _d("output", "signals_v2")  # 2-seed 生产信号

# —— 代码块 ——
CORE = _d("core")                        # 通用：模型、策略规则、指标、数据读取
LIVE = _d("live")                        # 实盘：抓数、模型注册、信号、账本、报告
ARCHIVE = _d("archive")                  # 归档：训练/回测/研究脚本
RESEARCH = _d("archive", "research")     # 归档的研究脚本（4.0~4.5）
REPORT_SCRIPTS = _d("archive", "reports")  # 归档的报告脚本（4.6~4.10）
LEGACY_MODEL = _d("archive", "legacy_model")
DATAPROC = _d("archive", "data_process")  # 历史全量抓取脚本

# 向后兼容别名（归档脚本里可能仍在用 _P.MODEL）
MODEL = CORE

# —— 实盘状态 ——
MODELS = _d("models")                    # 年度模型：models/pv/pv_{year}_s{seed}.pt
MODELS_PV = _d("models", "pv")
MODELS_SLOW = _d("models", "slow")
STATE = _d("state")                      # 账本、信号序列、每日快照
STATE_RAW = _d("state", "raw")           # L0 原始快照：state/raw/YYYY-MM-DD/*.json

# —— 历史信号（实盘 bootstrap 与归档脚本共用） ——
TRAIN = _d("archive", "signals", "_train_out")   # 172 个 6-seed 逐年信号

# —— 附件 ——
SCRIPTS = ARCHIVE
REPORTS = os.path.join(ROOT, "05_报告与图表")
LOGS = os.path.join(ARCHIVE, "logs")
REFERENCE = os.path.join(ROOT, "07_参考资料")

# 主数据文件名（唯一入口，避免各脚本各写一份）
INDEX_CSV = os.path.join(INPUT, "中证1000_000852.SH.csv")
SENTI_PANEL = os.path.join(INPUT, "情绪_面板_日频.csv")
FUT_POS_CSV = os.path.join(INPUT, "情绪_期货会员持仓.csv")
MARGIN_CSV = os.path.join(DATA, "两融数据.csv")
SIZE_CSV = os.path.join(DATA, "市场规模数据.csv")

# 让 `import timing_model` / `import strategy` 在任何目录下都能解析
for _p in (CORE, LIVE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def read_csv_any(path, **kw):
    """按常见编码依次尝试读取 CSV（GBK → GB18030 → UTF-8-SIG → UTF-8）。

    项目原始数据是 GBK 编码，但复制/另存后可能变成 UTF-8，这里做兼容。
    """
    import pandas as pd
    last = None
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, LookupError) as e:
            last = e
            continue
    raise last


if __name__ == "__main__":
    print(f"ROOT = {ROOT}\n")
    groups = [
        ("数据层", ("INPUT", "DATA", "DATA_RAW", "OUTPUT", "SIGNALS", "SIGNALS_V2")),
        ("代码块", ("CORE", "LIVE", "ARCHIVE", "RESEARCH", "REPORT_SCRIPTS",
                    "LEGACY_MODEL", "DATAPROC")),
        ("实盘状态", ("MODELS", "MODELS_PV", "MODELS_SLOW", "STATE", "STATE_RAW")),
        ("历史/附件", ("TRAIN", "REPORTS", "LOGS", "REFERENCE")),
    ]
    for title, keys in groups:
        print(f"[{title}]")
        for k in keys:
            p = globals()[k]
            n = len(os.listdir(p)) if os.path.isdir(p) else -1
            print(f"  {k:15} = {p}   ({n} 项)")
        print()
    print("[入口文件]")
    for k in ("INDEX_CSV", "SENTI_PANEL", "FUT_POS_CSV", "MARGIN_CSV", "SIZE_CSV"):
        p = globals()[k]
        print(f"  {k:15} {'OK ' if os.path.isfile(p) else 'MISS'} {p}")
