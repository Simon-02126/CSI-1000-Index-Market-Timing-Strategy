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
BASE = _P.ROOT; TRAIN = _P.TRAIN
ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
dc=[c for c in df.columns if "日期" in c][0]; cc=[c for c in df.columns if "收盘" in c][0]
close=pd.Series(pd.to_numeric(df[cc],errors="coerce").values,index=pd.to_datetime(df[dc]))
close=close[~close.index.duplicated(keep="last")].sort_index()
def load_tag(tag):
    parts=[]
    for p in sorted(glob.glob(os.path.join(TRAIN,f"{tag}_*.csv"))):
        x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"])
        cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
        s=pd.Series(x[cols].mean(axis=1).values,index=x["日期"]); parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None
def load_dir(v):
    p=glob.glob(os.path.join(BASE,"output","signals",f"*中证1000*_{v}_signals.csv"))[0]
    x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"]); cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
    x=x.drop_duplicates("日期").set_index("日期").sort_index(); return x[cols].mean(axis=1)
def roll_pct(s,win,minp=250):
    v=s.values.astype(float);o=np.full(len(v),np.nan);mp=min(minp,max(60,win//3))
    for i in range(len(v)):
        if i<mp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def pos3(p,lo,hi): 
    return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_lin(p,lo,span): return ((p-lo)/span).clip(0,1)
def pos_mode(t,k=5):
    v=t.values;o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1];w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True);o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,start="2019-01-01",end="2099-12-31",cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[(d.index>=pd.Timestamp(start))&(d.index<=pd.Timestamp(end))]
    if len(d)<60: return None
    r=np.log(d["c"]).diff();turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna(); ann=float(st.mean()*ANN);vol=float(st.std()*np.sqrt(ANN))
    eq=st.cumsum();mdd=float(np.expm1((eq-eq.cummax()).min()))
    return dict(sharpe=ann/vol if vol else np.nan,ann=ann,mdd=mdd,avg=float(d["p"].mean()),byyear=st.groupby(st.index.year).sum())
sig={sd:load_tag(f"slow_s{sd}") for sd in range(6)}
sig={k:v for k,v in sig.items() if v is not None}
avg=pd.concat(sig.values(),axis=1).mean(axis=1).dropna(); prod=load_dir("slow")
print("=== 满仓阈值 hi 扫描（三档 lo=0.30, k=5）===")
print(f"  {'信号':<10}{'W':>5}"+"".join(f"{'hi='+str(hi):>16}" for hi in (0.60,0.70,0.75,0.80,0.85,0.95)))
for lbl,s in (("6seed平均",avg),("生产版",prod)):
    for W in (150,300,750):
        row=[]
        for hi in (0.60,0.70,0.75,0.80,0.85,0.95):
            p=roll_pct(s,W); pos=pos_mode(pos3(p,0.30,hi),5)
            f=bt(pos); y26=bt(pos,"2026-01-01")
            row.append(f"{f['sharpe']:>7.2f}/{y26['sharpe']:>+5.2f}")
        print(f"  {lbl:<10}{W:>5}"+"".join(f"{x:>16}" for x in row))
print("  (格式: 全期夏普/2026夏普)")
print("\n=== 线性映射（pos=clip((pct-lo)/span)）===")
print(f"  {'信号':<10}{'W':>5}"+"".join(f"{str(lo)+'/'+str(sp):>14}" for lo,sp in ((0.20,0.40),(0.30,0.30),(0.30,0.50),(0.40,0.40))))
for lbl,s in (("6seed平均",avg),("生产版",prod)):
    for W in (150,300,750):
        row=[]
        for lo,sp in ((0.20,0.40),(0.30,0.30),(0.30,0.50),(0.40,0.40)):
            pos=pos_mode(pos_lin(roll_pct(s,W),lo,sp),5)
            f=bt(pos); y26=bt(pos,"2026-01-01")
            row.append(f"{f['sharpe']:>6.2f}/{y26['sharpe']:>+5.2f}")
        print(f"  {lbl:<10}{W:>5}"+"".join(f"{x:>14}" for x in row))
print("\n=== 各阈值下的仓位分布（6seed, W=300）===")
s=avg; p=roll_pct(s,300)
for hi in (0.60,0.70,0.80,0.95):
    pos=pos_mode(pos3(p,0.30,hi),5).dropna()
    print(f"  hi={hi}: 满仓{(pos>=.99).mean():>4.0%} 半仓{((pos>.01)&(pos<.99)).mean():>4.0%} 空仓{(pos<=.01).mean():>4.0%}")
