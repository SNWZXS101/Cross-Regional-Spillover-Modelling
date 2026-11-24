
from pathlib import Path
import math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = None
RESULTS_ROOT = "Model4GDGXcq2x_v6_optimized"
DATA_FILE = "Data_all_sm.xlsx"
GD_COLUMN, GX_COLUMN = "GD", "GX"
SUBSTEPS_PER_DAY = 20   # h = 0.05 day

DEFAULT_N1, DEFAULT_N2 = 12700e4, 5037e4
DEFAULT_IC = dict(Sq1=0.0,Sq2=4.0,I1=0.0,I2=1.0,Iq1=0.0,Iq2=0.0,R1=0.0,R2=0.0)
RATE_NAMES = ["c1","c2","q1","q2","m1","m2","d1","d2"]
STATE_NAMES = ["S1","S2","Sq1","Sq2","I1","I2","Iq1","Iq2","R1","R2","C1","C2"]

def newest_results_dir(root):
    c=[p for p in Path(root).glob("Training-results-*") if p.is_dir()]
    if not c: raise FileNotFoundError(f"No Training-results-* under {Path(root).resolve()}")
    return max(c,key=lambda p:p.stat().st_mtime)

def vec(p): return np.asarray(np.loadtxt(p,dtype=float)).reshape(-1)

def load_cfg(d):
    p=Path(d)/"config.json"
    return json.load(open(p,"r",encoding="utf-8")) if p.exists() else {}

def cfgnum(cfg,key,default): return float(cfg.get(key,default))

def metrics(y,p):
    y=np.asarray(y,float).reshape(-1); p=np.asarray(p,float).reshape(-1)
    rmse=float(np.sqrt(np.mean((y-p)**2)))
    mae=float(np.mean(np.abs(y-p)))
    den=float(np.sum((y-y.mean())**2))
    r2=np.nan if den==0 else 1-float(np.sum((y-p)**2))/den
    return dict(RMSE=rmse,MAE=mae,R2=r2,NRMSE=rmse/max(float(np.std(y)),1e-12))

def interp(arr,t):
    if t<=0: return float(arr[0])
    if t>=len(arr)-1: return float(arr[-1])
    j=int(math.floor(t)); a=t-j
    return float((1-a)*arr[j]+a*arr[j+1])

def rhs(t,x,r,beta,gamma,mu,N1,N2):
    S1,S2,Sq1,Sq2,I1,I2,Iq1,Iq2,R1,R2,C1,C2=x
    c1,c2,q1,q2,m1,m2,d1,d2=[interp(r[k],t) for k in RATE_NAMES]
    lam1=c1*S1*I1/N1; lam2=c2*S2*I2/N2
    dS1=-lam1+(1-q1)*(1-beta)*(1-m1)*lam1+mu*Sq1+(1-q2)*(1-beta)*m2*lam2
    dS2=-lam2+(1-q2)*(1-beta)*(1-m2)*lam2+mu*Sq2+(1-q1)*(1-beta)*m1*lam1
    dSq1=q1*(1-beta)*lam1-mu*Sq1; dSq2=q2*(1-beta)*lam2-mu*Sq2
    dI1=(1-q1)*beta*(1-m1)*lam1-gamma*I1+(1-q2)*beta*m2*lam2-d1*I1+d2*I2
    dI2=(1-q2)*beta*(1-m2)*lam2-gamma*I2+(1-q1)*beta*m1*lam1-d2*I2+d1*I1
    dIq1=q1*beta*lam1-gamma*Iq1; dIq2=q2*beta*lam2-gamma*Iq2
    dR1=gamma*(I1+Iq1); dR2=gamma*(I2+Iq2)
    dC1=q1*beta*lam1; dC2=q2*beta*lam2
    dI1-=gamma*I1; dI2-=gamma*I2
    dIq1+=gamma*I1; dIq2+=gamma*I2
    dC1+=gamma*I1; dC2+=gamma*I2
    return np.array([dS1,dS2,dSq1,dSq2,dI1,dI2,dIq1,dIq2,dR1,dR2,dC1,dC2],float)

def rk4(x0,r,beta,gamma,mu,N1,N2,n,substeps):
    h=1.0/substeps; x=np.asarray(x0,float).copy(); states=[x.copy()]; warns=[]
    for day in range(n):
        for s in range(substeps):
            t=day+s*h
            k1=rhs(t,x,r,beta,gamma,mu,N1,N2)
            k2=rhs(t+h/2,x+h*k1/2,r,beta,gamma,mu,N1,N2)
            k3=rhs(t+h/2,x+h*k2/2,r,beta,gamma,mu,N1,N2)
            k4=rhs(t+h,x+h*k3,r,beta,gamma,mu,N1,N2)
            x=x+h*(k1+2*k2+2*k3+k4)/6
            if not np.all(np.isfinite(x)): raise RuntimeError(f"Non-finite state at t={t}")
            if np.min(x[:10]) < -1e-5: warns.append(f"Negative state at t={t:.4f}: {np.min(x[:10]):.6g}")
        states.append(x.copy())
    return np.asarray(states),warns

def main():
    rd=Path(RESULTS_DIR) if RESULTS_DIR else newest_results_dir(RESULTS_ROOT)
    print("Using:",rd.resolve())
    out=rd/"Standalone-ODE-validation"; out.mkdir(exist_ok=True)

    df=pd.read_excel(DATA_FILE)
    gd=df[GD_COLUMN].to_numpy(float); gx=df[GX_COLUMN].to_numpy(float); n=len(gd)
    cfg=load_cfg(rd)
    N1=cfgnum(cfg,"N1_raw",DEFAULT_N1); N2=cfgnum(cfg,"N2_raw",DEFAULT_N2)
    Sq1=cfgnum(cfg,"Sq10_raw",DEFAULT_IC["Sq1"]); Sq2=cfgnum(cfg,"Sq20_raw",DEFAULT_IC["Sq2"])
    I1=cfgnum(cfg,"I10_raw",DEFAULT_IC["I1"]); I2=cfgnum(cfg,"I20_raw",DEFAULT_IC["I2"])
    Iq1=cfgnum(cfg,"Iq10_raw",DEFAULT_IC["Iq1"]); Iq2=cfgnum(cfg,"Iq20_raw",DEFAULT_IC["Iq2"])
    R1=cfgnum(cfg,"R10_raw",DEFAULT_IC["R1"]); R2=cfgnum(cfg,"R20_raw",DEFAULT_IC["R2"])
    x0=np.array([N1-Sq1-I1-Iq1-R1,N2-Sq2-I2-Iq2-R2,Sq1,Sq2,I1,I2,Iq1,Iq2,R1,R2,0,0],float)

    rates={k:vec(rd/f"{k}.txt") for k in RATE_NAMES}
    for k,v in rates.items():
        if len(v)!=n: raise ValueError(f"{k}.txt length {len(v)} != data length {n}")
    beta=float(vec(rd/"beta.txt")[0]); gamma=float(vec(rd/"gamma.txt")[0]); mu=float(vec(rd/"mu.txt")[0])
    print(f"beta={beta:.8f}, gamma={gamma:.8f}, mu={mu:.8f}, h={1/SUBSTEPS_PER_DAY:.3f}")

    states,warns=rk4(x0,rates,beta,gamma,mu,N1,N2,n,SUBSTEPS_PER_DAY)
    pgd=states[1:,10]-states[:-1,10]; pgx=states[1:,11]-states[:-1,11]

    saved_gd=vec(rd/"I1_new.txt") if (rd/"I1_new.txt").exists() else None
    saved_gx=vec(rd/"I2_new.txt") if (rd/"I2_new.txt").exists() else None

    rows=[]
    if saved_gd is not None: rows.append({"method":"V6 saved differentiable ODE fit","region":"GD",**metrics(gd,saved_gd)})
    if saved_gx is not None: rows.append({"method":"V6 saved differentiable ODE fit","region":"GX",**metrics(gx,saved_gx)})
    rows += [
        {"method":f"Standalone RK4 h={1/SUBSTEPS_PER_DAY:.3f}","region":"GD",**metrics(gd,pgd)},
        {"method":f"Standalone RK4 h={1/SUBSTEPS_PER_DAY:.3f}","region":"GX",**metrics(gx,pgx)},
    ]
    mdf=pd.DataFrame(rows); mdf.to_csv(out/"standalone_rollout_metrics.csv",index=False)
    print("\n",mdf.to_string(index=False))

    t=np.arange(n)
    fig,ax=plt.subplots(2,1,figsize=(13,8),sharex=True)
    ax[0].plot(t,gd,"k.",ms=4,label="GD observed")
    if saved_gd is not None: ax[0].plot(t,saved_gd,lw=1.5,label="V6 saved differentiable ODE fit")
    ax[0].plot(t,pgd,"--",lw=2,label=f"Standalone RK4 h={1/SUBSTEPS_PER_DAY:.2f}")
    ax[0].set_ylabel("Daily reported cases"); ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(t,gx,"g.",ms=4,label="GX observed")
    if saved_gx is not None: ax[1].plot(t,saved_gx,lw=1.5,label="V6 saved differentiable ODE fit")
    ax[1].plot(t,pgx,"--",lw=2,label=f"Standalone RK4 h={1/SUBSTEPS_PER_DAY:.2f}")
    ax[1].set_ylabel("Daily reported cases"); ax[1].set_xlabel("Days"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(out/"standalone_ODE_vs_data.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(2,1,figsize=(13,8),sharex=True)
    ax[0].plot(t,np.cumsum(gd),"k.",ms=4,label="Observed cumulative GD")
    ax[0].plot(t,states[1:,10],"--",lw=2,label="Standalone ODE cumulative")
    ax[0].grid(alpha=.3); ax[0].legend()
    ax[1].plot(t,np.cumsum(gx),"g.",ms=4,label="Observed cumulative GX")
    ax[1].plot(t,states[1:,11],"--",lw=2,label="Standalone ODE cumulative")
    ax[1].grid(alpha=.3); ax[1].legend(); ax[1].set_xlabel("Days")
    fig.tight_layout(); fig.savefig(out/"standalone_ODE_cumulative.png",dpi=300); plt.close(fig)

    for j,k in enumerate(STATE_NAMES):
        np.savetxt(out/f"Standalone_{k}.txt",states[:,j].reshape(-1,1))
    np.savetxt(out/"Standalone_I1_new.txt",pgd.reshape(-1,1))
    np.savetxt(out/"Standalone_I2_new.txt",pgx.reshape(-1,1))
    (out/"warnings.txt").write_text("\n".join(warns) if warns else "No warnings.",encoding="utf-8")

    gd_m=metrics(gd,pgd); gx_m=metrics(gx,pgx)
    print("\nSaved to:",out.resolve())
    if gd_m["R2"]>0.95 and gx_m["R2"]>0.95:
        print("STANDALONE MECHANISTIC VALIDATION: STRONG PASS")
    elif gd_m["R2"]>0.90 and gx_m["R2"]>0.90:
        print("STANDALONE MECHANISTIC VALIDATION: PASS")
    else:
        print("STANDALONE MECHANISTIC VALIDATION: NOT YET PASSING")

if __name__=="__main__":
    main()
