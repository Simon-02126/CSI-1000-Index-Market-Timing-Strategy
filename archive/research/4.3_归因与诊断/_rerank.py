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
BASE = _P.ROOT; ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
dc=[c for c in df.columns if "日期" in c][0]; cc=[c for c in df.columns if "收盘" in c][0]
close=pd.Series(pd.to_numeric(df[cc],errors="coerce").values,index=pd.to_datetime(df[dc]))
close=close[~close.index.duplicated(keep="last")].sort_index()
def load_dir(v):
    p=glob.glob(os.path.join(BASE,"output","signals",f"*中证1000*_{v}_signals.csv"))[0]
    x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"]); cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
    x=x.drop_duplicates("日期").set_index("日期").sort_index(); return x[cols].mean(axis=1)
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float);o=np.full(len(v),np.nan)
    for i in range(len(v)):
        if i<minp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo=.30,hi=.80): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_mode(t,k=5):
    v=t.values;o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1];w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True);o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,start="2019-01-01",cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[d.index>=pd.Timestamp(start)]
    if len(d)<100: return np.nan
    r=np.log(d["c"]).diff();turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna(); ann=float(st.mean()*ANN);vol=float(st.std()*np.sqrt(ANN))
    return ann/vol if vol else np.nan
pv=load_dir("pv"); slow=load_dir("slow")
idx=pv.index.intersection(slow.index)
a=roll_pct(pv.reindex(idx)); b=roll_pct(slow.reindex(idx))
fwd10=np.log(close.shift(-10)/close); f=fwd10.reindex(idx).values
ok=np.isfinite(f); zf=np.zeros_like(f); zf[ok]=(f[ok]-f[ok].mean())/f[ok].std()
def make(ic,ar=.90,seed=0):
    r=np.random.default_rng(seed); n=len(f); e=r.normal(size=n); x=np.zeros(n)
    for i in range(1,n): x[i]=ar*x[i-1]+e[i]*np.sqrt(1-ar**2)
    x=(x-x.mean())/x.std(); s=ic*zf+np.sqrt(max(1-ic**2,0))*x; s[~ok]=np.nan
    return pd.Series(s,index=idx)
print("=== 关键对照：合成后是否重新做分位(re-rank) ===")
print(f"  {'方案':<26}{'直接阈值':>10}{'合成后re-rank':>14}{'满仓占比(直)':>13}{'满仓占比(re)':>13}")
def full_ratio(p): 
    p=p.dropna(); return (p>=.80).mean()
def show(nm,comp):
    d1=bt(pos_mode(tgt3(comp))); d2=bt(pos_mode(tgt3(roll_pct(comp))))
    print(f"  {nm:<26}{d1:>10.3f}{d2:>14.3f}{full_ratio(comp):>13.0%}{full_ratio(roll_pct(comp)):>13.0%}")
print(f"  {'slow 单因子':<26}{bt(pos_mode(tgt3(b))):>10.3f}{bt(pos_mode(tgt3(roll_pct(b)))):>14.3f}"
      f"{full_ratio(b):>13.0%}{full_ratio(roll_pct(b)):>13.0%}")
show("双因子 0.5/0.5", 0.5*a+0.5*b)
for ic in (0.10,0.15,0.20,0.25):
    vals1=[];vals2=[]
    for sd in range(10):
        c=roll_pct(make(ic,seed=sd))
        m=pd.DataFrame({"a":a,"b":b,"c":c}).dropna()
        comp3=(m["a"]+m["b"]+m["c"])/3
        vals1.append(bt(pos_mode(tgt3(comp3))))
        vals2.append(bt(pos_mode(tgt3(roll_pct(comp3)))))
    print(f"  三维(各1/3) 新因子IC={ic:<6}"+f"{np.nanmean(vals1):>10.3f}{np.nanmean(vals2):>14.3f}"
          f"{'':>13}{'':>13}"+f"   (re-rank: 最小{np.nanmin(vals2):.2f} 最大{np.nanmax(vals2):.2f})")
print("\n=== 双因子自身用 re-rank 的效果（对 v4.0 的直接影响）===")
for w in (0.3,0.4,0.5,0.6,0.7):
    comp=w*a+(1-w)*b
    print(f"  w_pv={w:.1f}  直接阈值{bt(pos_mode(tgt3(comp))):>7.3f}   re-rank{bt(pos_mode(tgt3(roll_pct(comp)))):>7.3f}   满仓占比{full_ratio(roll_pct(comp)):>5.0%}")
