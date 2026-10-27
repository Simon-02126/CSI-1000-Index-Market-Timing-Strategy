# -*- coding: utf-8 -*-
"""宽基指数择时: LSTM(量价) + 静态慢塔(两融/市场规模) + 注意力池化 + MLP 因子信号
架构(财通证券《深度学习因子选股体系》择时改编版):
  快塔: 窗口内预处理 -> LSTM(8->64) -> 注意力池化(n*64 -> 64)
  慢塔: 两融/市场规模字段压缩成静态特征(t+1披露, 整体滞后1日), 与快塔拼接进MLP
  MLP(74->128->64) -> 64维信号
  loss = 64信号各自MSE均值 + λ * 信号间相关系数平方惩罚
滚动训练: 每年重训, 训练=Y-3之前, 验证=[Y-3,Y-1), 测试=Y全年(只推理)
label: ln(close_{t+10}/close_t)
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 运行时请用 PYTHONIOENCODING=utf-8

# ---------------- 超参数(全局唯一回看窗口 n) ----------------
V2 = os.environ.get("TM_V2", "0") == "1"   # v2: IC-loss+多期限(已归档, 实测无提升)
TM_H = int(os.environ.get("TM_H", "0"))    # >0: 单期限MSE + 16维信号试验
TM_N = int(os.environ.get("TM_N", "0"))     # >0: 自定义回看窗口长度
TM_NOTO = os.environ.get("TM_NOTO", "0") == "1"  # 1: 从量价输入中去掉换手率
N = TM_N if TM_N > 0 else 150               # 回看窗口长度(交易日), 量价预处理/慢塔变化特征/rolling参照共用
HORIZON = 10        # 默认主label 未来收益天数
if V2:
    HORIZONS = (5, 10, 20)
elif TM_H > 0:
    HORIZONS = (TM_H,)
else:
    HORIZONS = (HORIZON,)
K10 = HORIZONS.index(10) if 10 in HORIZONS else 0   # 评估label列(单20日时即20日label)
HID = 64            # LSTM 隐藏维度
SIG_DIM = 16 if (V2 or TM_H > 0) else HID           # 输出信号维度
MLP_HIDDEN = 128
DROPOUT = 0.2
LAMBDA_CORR = 0.1
LR = 3e-3
WEIGHT_DECAY = 1e-4
BATCH = 64
MAX_EPOCHS = int(os.environ.get("TM_MAXEP", "300"))
PATIENCE = int(os.environ.get("TM_PAT", "25"))    # valid IC 早停轮数
MAX_BATCHES_PER_EPOCH = 16  # 每epoch批数上限(吞吐上限, 非统计设计参数)
SEEDS = [0, 1]
FIRST_TEST_YEAR = 2015
MIN_PERIODS = 20    # 慢塔 rolling 参照的最小样本数

BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
OUTDIR = os.path.join(BASE, "output", "signals")
DEVICE = "cpu"

PRICE_COLS = ["开盘价", "最高价", "最低价", "收盘价"]   # 锚: 窗口末日收盘价
VWAP_COL = "均价(元)"                                   # 锚: 窗口末日均价值
SCALE_COLS = ["成交量(股)", "成交额(元)"] + ([] if TM_NOTO else ["换手率(%)"])
FAST_DIM = 5 + len(SCALE_COLS)                          # 快塔输入维度(4价+1均价+量额[率])

# 慢塔字段: 存量类(水平+窗口变化各1特征), 流量类(累计流向+单日强度各1特征)
# 注: 板块总股本仅sgn=11口径有2008年起完整数据(其余口径2020-05起), 换用覆盖完整的板块上市股票数
SLOW_STOCK = ["全市场融资余额(元)", "全市场融券余额(元)", "板块总市值(亿元)", "板块上市股票数"]
SLOW_FLOW = {"全市场融资净买入额(元)": "全市场融资余额(元)"}  # 流量字段: 归一化参照的存量字段


def load_index(fp):
    """读主数据。`fp` 可以是 csv 路径，也可以直接是 DataFrame（DB 数据源走这条）。"""
    if isinstance(fp, pd.DataFrame):
        df = fp.reset_index() if fp.index.name else fp.copy()
        if "日期" not in df.columns:
            df = df.rename(columns={df.columns[0]: "日期"})
        df["日期"] = pd.to_datetime(df["日期"])
        return df.sort_values("日期").reset_index(drop=True)
    df = pd.read_csv(fp, encoding="gbk")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df


def make_slow_features(df: pd.DataFrame, n: int):
    """两融/市场规模 -> 每字段2个静态特征, 共 (2*len(SLOW_STOCK) + 2*len(SLOW_FLOW)) 维。
    t+1披露: 所有原始序列先滞后1日 (s_t = raw_{t-1})。
    存量: log(s_t/rolling_mean(s,n)), log(s_t/s_{t-n+1})
    流量: sum(s_窗口)/存量_t, log(s_t/rolling_mean(|s|,n))"""
    T = len(df)
    slow_cols = []
    for c in SLOW_STOCK:
        raw = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        s = np.concatenate([[np.nan], raw[:-1]])            # 滞后1日
        ref = pd.Series(s).rolling(n, min_periods=MIN_PERIODS).mean().to_numpy()
        lvl = np.log(s / ref)
        chg = np.log(s / np.roll(s, n - 1))                 # s_t / s_{t-n+1}
        chg[:n - 1] = np.nan
        slow_cols += [lvl, chg]
    for c, base_c in SLOW_FLOW.items():
        raw = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        s = np.concatenate([[np.nan], raw[:-1]])
        base_raw = pd.to_numeric(df[base_c], errors="coerce").to_numpy(float)
        base = np.concatenate([[np.nan], base_raw[:-1]])
        cum = pd.Series(s).rolling(n, min_periods=1).sum().to_numpy()
        ratio = cum / base
        abss = np.abs(s)
        ref = pd.Series(abss).rolling(n, min_periods=MIN_PERIODS).mean().to_numpy().copy()
        ref[ref <= 0] = np.nan
        inten = np.log(np.abs(s) / ref) * np.sign(s)        # 带符号的相对强度
        slow_cols += [ratio, inten]
    slow = np.stack(slow_cols, axis=1).astype(np.float32)   # (T, S)
    slow[~np.isfinite(slow)] = np.nan                       # 除零/inf一律视为缺失
    return slow


CACHE_DIR = os.path.join(BASE, "output", "cache")


def ensure_cache(fp):
    """把 make_windows 结果存npz, 并行worker直接加载, 避免每单元重算Python循环.
    v2 用独立缓存文件(多期限labels), 与v1缓存互不干扰.

    缓存里记了源文件的 (mtime, size)：源数据一变（每日追加 / 历史修正）自动重建，
    否则实盘会一直用一份冻结在旧日期上的窗口矩阵。
    """
    is_df = isinstance(fp, pd.DataFrame)
    name = "db_features" if is_df else os.path.basename(str(fp))[:-4]
    os.makedirs(CACHE_DIR, exist_ok=True)
    suffix = "_v2" if V2 else (f"_h{TM_H}" if TM_H > 0 else "")
    if TM_N > 0 or TM_NOTO:
        suffix += f"_n{N}{'_nto' if TM_NOTO else ''}"
    cache_fp = os.path.join(CACHE_DIR, f"{name}{suffix}.npz")
    stamp = _frame_stamp(fp) if is_df else _src_stamp(fp)
    if os.path.exists(cache_fp):
        try:
            z = np.load(cache_fp, allow_pickle=True)
        except Exception:                                     # noqa: BLE001
            z = None
        if z is not None:
            if "src_hash" in z.files:
                same = str(z["src_hash"]) == str(stamp)
            elif (not is_df) and "src_mtime" in z.files and "src_size" in z.files:
                same = (float(z["src_mtime"]) == float(stamp[0])
                        and int(z["src_size"]) == int(stamp[1]))
            else:
                same = False
            if same:
                lab = z["labels"] if "labels" in z.files else z["label"][:, None]
                return (z["dates"], z["X"], z["slow"], lab, z["valid"], z["slow_ok"])
    df = load_index(fp)
    dates, X, slow, labels, valid, slow_ok = make_windows(df, N)
    meta = dict(src_hash=np.str_(str(stamp)))
    if V2:
        np.savez_compressed(cache_fp, dates=dates, X=X, slow=slow, labels=labels,
                            valid=valid, slow_ok=slow_ok, **meta)
    else:
        np.savez_compressed(cache_fp, dates=dates, X=X, slow=slow, label=labels[:, 0],
                            valid=valid, slow_ok=slow_ok, **meta)
    return dates, X, slow, labels, valid, slow_ok


def _src_stamp(fp: str):
    st = os.stat(str(fp))
    return float(st.st_mtime), int(st.st_size)


def _frame_stamp(df: pd.DataFrame) -> str:
    """DataFrame 的内容指纹（数值块 md5 + 末日 + 行数），用于缓存失效判断。"""
    import hashlib as _hl
    d = df.reset_index() if df.index.name else df.copy()
    if "日期" not in d.columns:
        d = d.rename(columns={d.columns[0]: "日期"})
    dts = pd.to_datetime(d["日期"])
    num = d.drop(columns=["日期"]).apply(pd.to_numeric, errors="coerce")
    h = _hl.md5(np.ascontiguousarray(num.to_numpy(np.float64)).tobytes())
    h.update(str(dts.max().date()).encode())
    h.update(str(len(d)).encode())
    return h.hexdigest()


def make_windows(df: pd.DataFrame, n: int):
    """对每个交易日 t 生成快塔窗口特征 X (n×8)、慢塔静态特征 slow、labels (T×len(HORIZONS))。
    价格(开高低收): log(p_i / close_T); 均价: log(v_i / v_T); 量额率: x_i / mean(x)"""
    close = df["收盘价"].to_numpy(float)
    dates = df["日期"].to_numpy()
    T = len(df)
    X = np.full((T, n, FAST_DIM), np.nan, dtype=np.float32)
    labels = np.full((T, len(HORIZONS)), np.nan, dtype=np.float32)
    log_close = np.log(close)
    log_vwap = np.log(df[VWAP_COL].to_numpy(float))
    vols = {c: df[c].to_numpy(float) for c in SCALE_COLS}
    logs = {c: np.log(df[c].to_numpy(float)) for c in PRICE_COLS}
    slow = make_slow_features(df, n)

    for t in range(n - 1, T):
        w = slice(t - n + 1, t + 1)
        anchor = log_close[t]
        cols = [logs[c][w] - anchor for c in PRICE_COLS]
        cols.append(log_vwap[w] - log_vwap[t])
        for c in SCALE_COLS:
            x = vols[c][w]
            m = x.mean()
            cols.append(x / m if m > 0 else np.zeros(n))
        X[t] = np.stack(cols, axis=1)
        for j, h in enumerate(HORIZONS):
            if t + h <= T - 1:
                labels[t, j] = log_close[t + h] - log_close[t]
    valid = ~np.isnan(labels).any(axis=1)
    slow_ok = ~np.isnan(slow).any(axis=1)
    return dates, X, slow, labels, valid, slow_ok


class TimingNet(nn.Module):
    """view='dual': LSTM(8)池化 + 慢塔拼接; view='pv': 仅快塔; view='slow': 仅慢塔静态MLP"""

    def __init__(self, in_dim=8, slow_dim=10, hid=64, mlp_hidden=128, dropout=0.2, out_dim=HID):
        super().__init__()
        self.use_fast = in_dim > 0
        self.use_slow = slow_dim > 0
        if self.use_fast:
            self.lstm = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True)
            self.att_w = nn.Linear(hid, hid, bias=False)
            self.query = nn.Parameter(torch.randn(hid) * 0.02)
        mlp_in = (hid if self.use_fast else 0) + (slow_dim if self.use_slow else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim))

    def forward(self, x, slow):
        parts = []
        if self.use_fast:
            H, _ = self.lstm(x)
            scores = (self.att_w(H) @ self.query) / np.sqrt(H.shape[-1])
            alpha = torch.softmax(scores, dim=1)
            parts.append((alpha.unsqueeze(-1) * H).sum(dim=1))
        if self.use_slow:
            parts.append(slow)
        return self.mlp(torch.cat(parts, dim=1))


def corr_penalty(sig: torch.Tensor) -> torch.Tensor:
    s = sig - sig.mean(dim=0, keepdim=True)
    std = s.std(dim=0, unbiased=False) + 1e-8
    s = s / std
    C = (s.T @ s) / sig.shape[0]
    off = C[~torch.eye(C.shape[0], dtype=torch.bool)]
    return (off ** 2).mean()


def ic_loss(pred: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """v2: 负IC损失. pred (B,D), Y (B,H) 多期限label; 返回各期限各信号维度负IC的均值"""
    p = pred - pred.mean(dim=0, keepdim=True)
    total = 0.0
    for h in range(Y.shape[1]):
        t = Y[:, h:h + 1]
        tc = t - t.mean(dim=0, keepdim=True)
        denom = p.norm(dim=0) * tc.norm(dim=0) + 1e-8
        total = total - ((p * tc).sum(dim=0) / denom).mean()
    return total / Y.shape[1]


def train_one_fold(Xtr, Str, ytr, Xva, Sva, yva, seed, use_fast=True, use_slow=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = TimingNet(in_dim=FAST_DIM if use_fast else 0,
                    slow_dim=Str.shape[1] if use_slow else 0, out_dim=SIG_DIM).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    yva_np = yva[:, K10].ravel()
    best = {"ic": -9.0, "epoch": -1, "state": None, "bad": 0}
    g = torch.Generator().manual_seed(seed)
    for ep in range(MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        nb = 0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 16:
                continue
            xb = torch.tensor(Xtr[idx.numpy()]).float()
            sb = torch.tensor(Str[idx.numpy()]).float()
            if V2:
                yb = torch.tensor(ytr[idx.numpy()]).float()          # (B,H)
                pred = net(xb, sb)
                loss = ic_loss(pred, yb) + LAMBDA_CORR * corr_penalty(pred)
            else:
                yb = torch.tensor(ytr[idx.numpy()][:, 0]).float().unsqueeze(1)  # (B,1) 与(B,D)广播
                pred = net(xb, sb)
                loss = ((pred - yb) ** 2).mean() + LAMBDA_CORR * corr_penalty(pred)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            nb += 1
            if nb >= MAX_BATCHES_PER_EPOCH:
                break
        net.eval()
        with torch.no_grad():
            pv = predict(net, Xva, Sva).mean(axis=1)
        ic = np.corrcoef(pv, yva_np)[0, 1]
        if ic > best["ic"]:
            best = {"ic": ic, "epoch": ep, "state": {k: v.clone() for k, v in net.state_dict().items()}, "bad": 0}
        else:
            best["bad"] += 1
            if best["bad"] >= PATIENCE:
                break
    net.load_state_dict(best["state"])
    return net, best


@torch.no_grad()
def predict(net, X, S):
    net.eval()
    outs = []
    for i in range(0, len(X), 512):
        outs.append(net(torch.tensor(X[i:i + 512]).float(),
                        torch.tensor(S[i:i + 512]).float()).numpy())
    return np.concatenate(outs)


def run(fp: str, year_from: int = None, year_to: int = None,
        view: str = "dual", out_fp: str = None, seeds=None):
    """view: 'pv'=纯量价快塔, 'slow'=纯慢塔静态MLP, 'dual'=双流
    out_fp: 指定则写出单文件(供并行按年写tmp); 否则写 output/signals/{name}_{view}_signals.csv 并合并已有
    返回写入的测试日数(样本不足的年份返回0, 调度器据此标记done)"""
    assert view in ("pv", "slow", "dual")
    use_fast = view in ("pv", "dual")
    use_slow = view in ("slow", "dual")
    name = os.path.basename(fp)[:-4]
    dates, X, slow, labels, valid, slow_ok = ensure_cache(fp)
    d = pd.to_datetime(dates)
    last_year = d[-1].year
    y0 = year_from or FIRST_TEST_YEAR
    y1 = min(year_to or last_year, last_year)
    seeds = seeds or SEEDS
    outp = out_fp or os.path.join(OUTDIR, f"{name}_{view}_signals.csv")
    default_mode = out_fp is None
    existing = None
    if default_mode and os.path.exists(outp):
        existing = pd.read_csv(outp, encoding="utf-8-sig", parse_dates=["日期"])
    elif not default_mode and os.path.exists(outp):
        existing = pd.read_csv(outp, encoding="utf-8-sig", parse_dates=["日期"])

    total_new = 0
    rows = []
    for year in range(y0, y1 + 1):
        t_start = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
        t_valid = np.datetime64(pd.Timestamp(year=year - 3, month=1, day=1))
        t_valid_end = np.datetime64(pd.Timestamp(year=year - 1, month=1, day=1))
        ok = valid & slow_ok if use_slow else valid   # 纯量价视图不依赖慢塔数据完整性
        tr_idx = np.where(ok & (d < t_valid))[0][:-(max(HORIZONS) + 1)]   # embargo(按最长horizon)
        va_mask = ok & (d >= t_valid) & (d < t_valid_end)   # 验证集=[Y-3, Y-1) 两年
        te_mask = ok & (d >= t_start) & (d < np.datetime64(pd.Timestamp(year=year + 1, month=1, day=1)))
        if tr_idx.size < 100 or va_mask.sum() < 100:
            print(f"  {name} {view} {year}: 训练/验证样本不足, 跳过")
            continue
        # 快塔/慢塔/labels 均按训练集统计 z-score
        mu, sigma = labels[tr_idx].mean(0), labels[tr_idx].std(0) + 1e-8
        if use_fast:
            fmu = X[tr_idx].reshape(-1, FAST_DIM).mean(0); fsd = X[tr_idx].reshape(-1, FAST_DIM).std(0) + 1e-8
            Xtr, Xva, Xte = (X[tr_idx] - fmu) / fsd, (X[va_mask] - fmu) / fsd, (X[te_mask] - fmu) / fsd
        else:
            Xtr = np.zeros((len(tr_idx), N, FAST_DIM), np.float32)
            Xva = np.zeros((va_mask.sum(), N, FAST_DIM), np.float32)
            Xte = np.zeros((te_mask.sum(), N, FAST_DIM), np.float32)
        if use_slow:
            smu = slow[tr_idx].mean(0); ssd = slow[tr_idx].std(0) + 1e-8
            Str, Sva, Ste = (slow[tr_idx] - smu) / ssd, (slow[va_mask] - smu) / ssd, (slow[te_mask] - smu) / ssd
        else:
            Str = np.zeros((len(tr_idx), 0), np.float32)
            Sva = np.zeros((va_mask.sum(), 0), np.float32)
            Ste = np.zeros((te_mask.sum(), 0), np.float32)
        ytr_z, yva_z = (labels[tr_idx] - mu) / sigma, (labels[va_mask] - mu) / sigma
        label10 = labels[:, K10]          # 输出与评估统一用10日label
        preds = []
        t0 = time.time()
        for seed in seeds:
            net, best = train_one_fold(Xtr, Str, ytr_z, Xva, Sva, yva_z, seed,
                                       use_fast=use_fast, use_slow=use_slow)
            preds.append(predict(net, Xte, Ste))
            print(f"  {name} {view} {year} seed{seed}: 训练{len(tr_idx)} 验证{va_mask.sum()} "
                  f"最优epoch {best['epoch']} validIC {best['ic']:+.4f}")
        sig = np.mean(preds, axis=0)
        te_idx = np.where(te_mask)[0]
        for i, ti in enumerate(te_idx):
            rows.append([d[ti]] + list(sig[i]) + [label10[ti]])
        total_new += len(te_idx)
        cols = ["日期"] + [f"信号{i:02d}" for i in range(SIG_DIM)] + ["label"]
        out_new = pd.DataFrame(rows, columns=cols)
        if default_mode:
            out = pd.concat([existing, out_new], ignore_index=True) if existing is not None else out_new
            out = out.drop_duplicates(subset="日期", keep="last").sort_values("日期").reset_index(drop=True)
            os.makedirs(OUTDIR, exist_ok=True)
            out.to_csv(outp, index=False, encoding="utf-8-sig")
            existing = out
            rows = []
        print(f"  {name} {view} {year}: 测试 {te_mask.sum()} 日, 耗时 {time.time() - t0:.0f}s")

    if not default_mode:
        if rows:
            cols = ["日期"] + [f"信号{i:02d}" for i in range(SIG_DIM)] + ["label"]
            out_new = pd.DataFrame(rows, columns=cols)
            if existing is not None:
                out_new = pd.concat([existing, out_new], ignore_index=True)
            out_new = out_new.drop_duplicates(subset="日期", keep="last").sort_values("日期")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            out_new.to_csv(outp, index=False, encoding="utf-8-sig")
        return total_new

    # 默认模式: 汇总打印逐年IC
    out = existing
    if out is None or len(out) == 0:
        return 0
    out["信号均值"] = out[[f"信号{i:02d}" for i in range(SIG_DIM)]].mean(axis=1)
    ok_df = out.dropna(subset=["label"])
    print(f"\n[{name} {view}] 逐年 IC (信号均值 vs 未来10日对数收益):")
    for y, g in ok_df.groupby(ok_df["日期"].dt.year):
        ic = g["信号均值"].corr(g["label"])
        print(f"  {y}: IC={ic:+.4f}")
    return total_new


if __name__ == "__main__":
    yf = int(sys.argv[1]) if len(sys.argv) > 1 else None
    yt = int(sys.argv[2]) if len(sys.argv) > 2 else None
    view = sys.argv[3] if len(sys.argv) > 3 else "dual"
    if len(sys.argv) > 4:
        SEEDS = [int(s) for s in sys.argv[4].split(",")]
    run(os.path.join(INPUT, "沪深300_000300.SH.csv"), yf, yt, view=view)
