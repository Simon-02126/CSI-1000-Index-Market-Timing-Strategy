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
TRAIN=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_train_out"); OUT=os.path.join(BASE,"output")
ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
dc=[c for c in df.columns if "日期" in c][0]; cc=[c for c in df.columns if "收盘" in c][0]
close=pd.Series(pd.to_numeric(df[cc],errors="coerce").values,index=pd.to_datetime(df[dc]))
close=close[~close.index.duplicated(keep="last")].sort_index()
def load_dir(d,view):
    p=glob.glob(os.path.join(OUT,d,f"*中证1000*_{view}_signals.csv"))[0]
    x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"])
    cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
    x=x.drop_duplicates("日期").set_index("日期").sort_index(); return x[cols].mean(axis=1)
def load_train(tag):
    parts=[]
    for p in sorted(glob.glob(os.path.join(TRAIN,f"{tag}_*.csv"))):
        x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"])
        cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
        s=pd.Series(x[cols].mean(axis=1).values,index=x["日期"]); parts.append(s[~s.index.duplicated(keep="last")])
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
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[(d.index>=pd.Timestamp("2019-01-01"))&(d.index<=pd.Timestamp("2024-12-31"))]
    r=np.log(d["c"]).diff(); turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna(); return st.groupby(st.index.year).sum()
base=load_dir("signals","pv"); h10=load_train("h10")
print("=== 逐年策略收益 (三档+众数10, 单因子 pv) ===")
print(f"  {'年份':<6}{'v4.0 pv':>10}{'h10 pv':>10}{'差异':>10}{'信号相关性':>10}")
r1=bt(pos_mode(tgt3(roll_pct(base)),10)); r2=bt(pos_mode(tgt3(roll_pct(h10)),10))
for y in sorted(set(r1.index)&set(r2.index)):
    a,b=r1.get(y,0),r2.get(y,0)
    m=pd.DataFrame({"a":base,"b":h10}).dropna(); m=m[m.index.year==y]
    c=m["a"].corr(m["b"])
    print(f"  {y:<6}{a*100:>9.1f}%{b*100:>9.1f}%{(b-a)*100:>9.1f}%{c:>10.3f}")
print(f"  {'累计':<6}{r1.sum()*100:>9.1f}%{r2.sum()*100:>9.1f}%{(r2.sum()-r1.sum())*100:>9.1f}%")
print("\n=== 同年份 IC(H=10) 对比 ===")
fwd=np.log(close.shift(-10)/close)
for y in range(2019,2025):
    row=[]
    for nm,s in (("v4.0",base),("h10",h10)):
        m=pd.DataFrame({"s":s,"f":fwd.reindex(s.index)}).dropna(); m=m[m.index.year==y]
        row.append(f"{nm}={m['s'].corr(m['f']):+.3f}")
    print(f"  {y}  "+"  ".join(row))
