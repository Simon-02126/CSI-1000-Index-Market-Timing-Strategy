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
ANN=252; START="2019-01-01"; END="2024-12-31"
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
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float); o=np.full(len(v),np.nan)
    for i in range(len(v)):
        if i<minp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo,hi): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_mode(t,k):
    v=t.values; o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1]; w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True); o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[(d.index>=pd.Timestamp(START))&(d.index<=pd.Timestamp(END))]
    r=np.log(d["c"]).diff(); turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna()
    eq=st.cumsum(); mdd=float(np.expm1((eq-eq.cummax()).min()))
    ann=float(st.mean()*ANN); vol=float(st.std()*np.sqrt(ANN))
    return (ann/vol if vol else np.nan), ann, mdd, float(d["p"].mean())
S={"单10日":load_tag("h10"),"双期5+10 MSE":load_tag("d5_10_mse"),"双期5+10 IC":load_tag("d5_10_ic")}
for k in (1,5):
    print(f"\n=== 阈值网格 × 众数{k}（夏普 / 满仓占比）===")
    hdr="  "+f"{'信号':<14}"+"".join(f"{str(lo)+'/'+str(hi):>17}" for lo in (.20,.25,.30) for hi in (.60,.70,.80))
    print(hdr)
    for nm,s in S.items():
        p=roll_pct(s); row=[]
        for lo in (.20,.25,.30):
            for hi in (.60,.70,.80):
                pos=pos_mode(tgt3(p,lo,hi),k)
                sh,ann,mdd,avg=bt(pos)
                pv=p.dropna(); pv=pv[(pv.index>=START)&(pv.index<=END)]
                full=((tgt3(p,lo,hi)).dropna()>=1).mean()
                row.append(f"{sh:>10.2f}/{full:>4.0%}")
        print(f"  {nm:<14}"+"".join(f"{x:>17}" for x in row))
