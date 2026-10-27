-- -*- coding: utf-8 -*-
-- timing.db 数据层 schema —— 四层结构: raw / derived / state / meta
-- 规则: raw 层幂等 upsert; derived 层 DROP 可重建; state 层 append-only; 训练只读 v_features_final
-- 初始导入截止 2026-10-20, 之后日期由 fetch 链路写入 (测试窗口)

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ================================================================ raw 原始层

-- 量价主表: 单一事实源。status 状态机隔离盘中 provisional bar (final 永不被 provisional 覆盖)
CREATE TABLE IF NOT EXISTS index_daily (
    code        TEXT    NOT NULL,               -- 品种代码, 如 000852.SH
    date        TEXT    NOT NULL,               -- 交易日 ISO8601
    open        REAL, high REAL, low REAL, close REAL,
    volume      REAL,                           -- 成交量(股)
    amount      REAL,                           -- 成交额(元)
    turnover    REAL,                           -- 换手率(%)
    vwap        REAL,                           -- 均价 = 成交额/成交量
    status      TEXT    NOT NULL DEFAULT 'final'
                CHECK (status IN ('provisional','final')),
    source      TEXT,                           -- eastmoney / baostock / tencent
    fetched_at  TEXT,                           -- 首次抓取时间
    finalized_at TEXT,                          -- 翻转为 final 的时间
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_index_daily_date ON index_daily(date);

-- 两融: trade_date 与 effective_date (T+1披露) 分列, as_of 查询防未来函数
CREATE TABLE IF NOT EXISTS margin_daily (
    date            TEXT NOT NULL,              -- 数据所属交易日 T
    exch            TEXT NOT NULL,              -- SH / SZ / BSE / ALL
    fin_balance     REAL,                       -- 融资余额(元)
    short_balance   REAL,                       -- 融券余额(元)
    fin_buy         REAL,                       -- 融资买入额(元); 净买入不存, 由余额差分派生
    effective_date  TEXT,                       -- 实际可得交易日 = T+1
    source          TEXT,
    fetched_at      TEXT,
    PRIMARY KEY (date, exch)
);
CREATE INDEX IF NOT EXISTS idx_margin_eff ON margin_daily(effective_date);

-- 市场规模: 一行一个 (日期, 口径), 沿用 DataSgnCode
CREATE TABLE IF NOT EXISTS market_size_daily (
    date            TEXT NOT NULL,
    sgn_code        INTEGER NOT NULL,           -- 1沪股票 2沪主板A 4科创板 5深股票 8创业板 11沪深
    sgn_name        TEXT,
    listed_co       REAL,                       -- 上市公司数
    stock_num       REAL,                       -- 上市股票数
    total_shares    REAL,                       -- 总股本(亿股)
    circ_shares     REAL,                       -- 流通股本(亿股)
    mkt_value       REAL,                       -- 总市值(亿元)
    circ_value      REAL,                       -- 流通市值(亿元)
    turnover1       REAL, turnover2 REAL,       -- 换手率两个口径(%)
    effective_date  TEXT,
    source          TEXT,
    fetched_at      TEXT,
    PRIMARY KEY (date, sgn_code)
);

-- 中金所持仓原始表: 逐行原样存, 多空比由 v_if_ratio 现算 (口径可切换)
CREATE TABLE IF NOT EXISTS cffex_rank_raw (
    date                TEXT NOT NULL,
    variety             TEXT NOT NULL,          -- IF / IC / IM / IH
    contract            TEXT NOT NULL,          -- 合约代码; 历史聚合导入存 'ALL'
    rank                INTEGER NOT NULL,         -- 1~20 或 999 合计
    long_open_interest  REAL,                   -- 持买单量
    short_open_interest REAL,                   -- 持卖单量
    fetched_at          TEXT,
    PRIMARY KEY (date, variety, contract, rank)
);

-- ETF 行情表: 实盘成交标的 (512100 中证1000ETF), 与 index_daily 同构
CREATE TABLE IF NOT EXISTS etf_daily (
    code        TEXT    NOT NULL,               -- ETF代码, 如 512100.SH
    date        TEXT    NOT NULL,
    open        REAL, high REAL, low REAL, close REAL,
    volume      REAL, amount REAL,
    status      TEXT    NOT NULL DEFAULT 'final'
                CHECK (status IN ('provisional','final')),
    source      TEXT,
    fetched_at  TEXT,
    finalized_at TEXT,
    PRIMARY KEY (code, date)
);
CREATE INDEX IF NOT EXISTS idx_etf_daily_date ON etf_daily(date);

-- ================================================================ derived 派生层

-- 入模 28 列宽表 (列名与主数据 CSV 一致), build_hash 溯源
CREATE TABLE IF NOT EXISTS features_main (
    date TEXT PRIMARY KEY,
    "开盘价" REAL, "最高价" REAL, "最低价" REAL, "收盘价" REAL,
    "换手率(%)" REAL, "成交量(股)" REAL, "成交额(元)" REAL, "均价(元)" REAL,
    "沪市融资余额(元)" REAL, "深市融资余额(元)" REAL, "全市场融资余额(元)" REAL,
    "全市场融资买入额(元)" REAL, "全市场融资净买入额(元)" REAL, "全市场融券余额(元)" REAL,
    "沪市两融余额(元)" REAL, "深市两融余额(元)" REAL, "全市场两融余额(元)" REAL,
    "板块上市公司数" REAL, "板块上市股票数" REAL, "板块总股本(亿股)" REAL, "板块流通股本(亿股)" REAL,
    "板块总市值(亿元)" REAL, "板块流通市值(亿元)" REAL,
    "板块成交量(亿股)" REAL, "板块成交额(亿元)" REAL,
    "板块换手率(%)" REAL, "板块流通换手率(%)" REAL,
    build_hash TEXT,                            -- 当日 raw 行集合哈希
    built_at   TEXT
);

-- 情绪面板: 33 列宽表 (日期 + 32 字段, 列名与 情绪_面板_日频.csv 一致)
CREATE TABLE IF NOT EXISTS emotion_panel (
    date TEXT PRIMARY KEY,
    "50ETF_PCR成交" REAL, "50ETF_PCR持仓" REAL, "50ETF_成交额万元" REAL,
    "300ETF_PCR成交" REAL, "300ETF_PCR持仓" REAL, "300ETF_成交额万元" REAL,
    "500ETF_PCR成交" REAL, "500ETF_PCR持仓" REAL, "500ETF_成交额万元" REAL,
    "50ETF_IV平值" REAL, "50ETF_IV虚沽" REAL, "50ETF_IV虚购" REAL,
    "50ETF_IV沽购比" REAL, "50ETF_IV偏斜" REAL,
    "300ETF_IV平值" REAL, "300ETF_IV虚沽" REAL, "300ETF_IV虚购" REAL,
    "300ETF_IV沽购比" REAL, "300ETF_IV偏斜" REAL,
    "500ETF_IV平值" REAL, "500ETF_IV虚沽" REAL, "500ETF_IV虚购" REAL,
    "500ETF_IV沽购比" REAL, "500ETF_IV偏斜" REAL,
    "IF_会员多空比" REAL, "IC_会员多空比" REAL, "IM_会员多空比" REAL, "IH_会员多空比" REAL,
    "前20_持买单量" REAL, "前20_持卖单量" REAL,
    "合约条数" REAL, "合约数" REAL
);

-- IF 多空比派生表 (物化, live/signals 直查)
CREATE TABLE IF NOT EXISTS if_sentiment (
    date        TEXT NOT NULL,
    variety     TEXT NOT NULL,
    buy_top20   REAL,                           -- rank=999 合计行买单加总
    sell_top20  REAL,
    ratio       REAL,                           -- buy/sell
    n_contracts INTEGER,                        -- 合约条数 = 20 x 合约数 (轮换监测)
    PRIMARY KEY (date, variety)
);

-- ================================================================ state 状态层

-- 账本: append-only, 错了下一行反向冲销; strategy_hash 记录规则版本
CREATE TABLE IF NOT EXISTS ledger (
    txn_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    exec_date    TEXT NOT NULL,                 -- 成交日 T+1
    signal_date  TEXT,                          -- 信号日 T
    target_pos   REAL,                          -- 成交后目标仓位 [0,1]
    exec_price   REAL,                          -- 成交价(开盘价)
    mark_price   REAL,                          -- 收盘盯市
    position     REAL,                        -- 成交后持仓比例
    shares       REAL, cash REAL, equity REAL,
    nav          REAL, daily_ret REAL, cost REAL,
    strategy_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_ledger_exec ON ledger(exec_date);
CREATE INDEX IF NOT EXISTS idx_ledger_sig  ON ledger(signal_date);

-- 信号冻结: 存证据不存结论; invalidated=1 表示上游数据修正后作废
CREATE TABLE IF NOT EXISTS signal_freeze (
    signal_date TEXT PRIMARY KEY,
    base        REAL, if_pct REAL, a0 REAL, target_pos REAL,
    input_hash  TEXT,                           -- 生成所依赖的数据快照哈希
    model_hash  TEXT,                           -- 模型指纹 (引用 model_files)
    frozen_at   TEXT,
    invalidated INTEGER NOT NULL DEFAULT 0
);

-- 模型注册: ensure 缺年检测与 --retrain 提示的依据; 权重本体留文件系统
CREATE TABLE IF NOT EXISTS model_files (
    view              TEXT NOT NULL,            -- pv / slow
    year              INTEGER NOT NULL,
    seed              TEXT NOT NULL,            -- s0 ~ s5
    file_path         TEXT,                   -- 相对路径
    train_window_hash TEXT,                   -- 训练窗口数据指纹 (数据修正 -> 提示 --retrain)
    train_data_end    TEXT,                   -- 训练数据末日 (判缺年)
    trained_at        TEXT,
    PRIMARY KEY (view, year, seed)
);

-- ================================================================ meta 元数据层

-- 抓取日志: values_changed=1 表示抓了老日期但数值变了 -> 触发重建/重训联动
CREATE TABLE IF NOT EXISTS ingest_log (
    ingest_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name    TEXT, date_from TEXT, date_to TEXT,
    rows_new      INTEGER, rows_updated INTEGER,
    values_changed INTEGER NOT NULL DEFAULT 0,
    check_result  TEXT,                         -- JSON: 本次校验明细
    started_at    TEXT, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ingest_table ON ingest_log(table_name, date_from);

-- 日度自检: dashboard 红绿灯直读
CREATE TABLE IF NOT EXISTS quality_check (
    check_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date  TEXT, check_type TEXT,           -- cross_source / playback / gap / margin_recon
    ref_date  TEXT,
    metric    TEXT, value REAL, threshold REAL,
    pass      INTEGER
);

-- ================================================================ 视图层

-- 铁律: 训练/回测/报表只读此视图 (provisional 在机制上接触不到模型)
CREATE VIEW IF NOT EXISTS v_features_final AS
SELECT f.* FROM features_main f
JOIN index_daily i ON i.code = '000852.SH' AND i.date = f.date AND i.status = 'final';

-- 多空比现算: rank=999 合计行口径 (与 v_if_sentiment 物化表互相印证)
CREATE VIEW IF NOT EXISTS v_if_ratio AS
SELECT date, variety,
       SUM(long_open_interest)  AS buy_top20,
       SUM(short_open_interest) AS sell_top20,
       SUM(long_open_interest) / SUM(short_open_interest) AS ratio
FROM cffex_rank_raw WHERE rank = 999
GROUP BY date, variety;

-- 净值序列 (报表/看板直读)
CREATE VIEW IF NOT EXISTS v_nav AS
SELECT exec_date, nav, equity, position, daily_ret FROM ledger ORDER BY exec_date;
