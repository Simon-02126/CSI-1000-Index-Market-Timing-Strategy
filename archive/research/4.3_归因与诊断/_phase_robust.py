import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = _P.ROOT
TRAIN = _P.TRAIN
ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
def load_price():
    fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
    d=[c for c in df.columns if "日期" in c][0]; c=[c for c in df.columns if "收盘" in c][0]
    s=pd.Series(pd.to_numeric(df[c],errors="coerce").values,index=pd.to_datetime(df[d])); return s[~s.index.duplicated(keep="last")].sort_index()
def load_train(tag):
    parts=[]
    for fp in sorted(glob.glob(os.path.join(TRAIN,f"{tag}_*.csv"))):
        df=pd.read_csv(fp,encoding="utf-8-sig",parse_dates=["日期"])
        cols=[c for c in df.columns if c.startswith("信号") and c!="信号均值"]
        s=pd.Series(df[cols].mean(axis=1).values,index=df["日期"]); parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index()
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float); o=np.full(len(v),np.nan)
    for i in range(len(v)):
        if i<minp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo=.30,hi=.80): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_mode(t,k):
    v=t.values; o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1]; w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True); o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[d.index>=pd.Timestamp("2019-01-01")]
    if len(d)<150: return None
    r=np.log(d["c"]).diff(); turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna()
    eq=st.cumsum(); mdd=float(np.expm1((eq-eq.cummax()).min()))
    ann=float(st.mean()*ANN); vol=float(st.std()*np.sqrt(ANN))
    return dict(sharpe=ann/vol if vol else np.nan,ann=ann,mdd=mdd,turn=float(turn.sum())/(len(d)/ANN))
close=load_price()
print("=== 相位稳健性: 每 N 日调仓, 起点错开 off=0..N-1 (2019-2024, label=10日) ===")
for tag in ("h10","h20"):
    s=load_train(tag); base=pos_mode(tgt3(roll_pct(s)),5)
    print(f"\n--- {tag} ---")
    for N in (5,10,15,20,25,40):
        sh, an = [], []
        for off in range(N):
            v=base.values.copy(); cur=v[0]
            for i in range(len(v)):
                if (i-off)%N==0: cur=v[i]
                else: v[i]=cur
            r=bt(pd.Series(v,index=base.index))
            if r: sh.append(r["sharpe"]); an.append(r["ann"])
        sh=np.array(sh)
        ngt=(sh>0).mean()
        print(f"  N={N:<3} 夏普 均值{sh.mean():>5.2f} 中位{np.median(sh):>5.2f} 最好{sh.max():>5.2f} 最差{sh.min():>5.2f} "
              f"| 年化 均值{np.mean(an):>6.1%} | 正相位比例{ngt:>4.0%} (n={len(sh)})")
