"""Current 12-state model, in raw persons and days. No state clipping."""
from pathlib import Path
import json
import numpy as np
from numba import njit

from scripts.common import Context
CTX=Context()
ROOT=CTX.root;PROJECT=CTX.project;BASE=CTX.base
CFG=json.loads((BASE/'config.json').read_text())
NAMES=['c1','c2','q1','q2','m1','m2','d1','d2']
STATES=['S1','S2','Sq1','Sq2','I1','I2','Iq1','Iq2','R1','R2','C1','C2']
LOW=np.array([CFG[k+'_bounds'][0] for k in NAMES],float)
HIGH=np.array([CFG[k+'_bounds'][1] for k in NAMES],float)
BETA_BOUNDS=CFG['beta_bounds']
N=np.array([CFG['N1_raw'],CFG['N2_raw']])
X0=np.array([0.,0.]+[CFG[k+'0_raw'] for k in ['Sq1','Sq2','I1','I2','Iq1','Iq2','R1','R2']]+[0.,0.])
X0[:2]=N-X0[2:10].reshape(4,2).sum(axis=0)
G=float(np.loadtxt(BASE/'gamma.txt')); MU=float(np.loadtxt(BASE/'mu.txt'))
NU=np.array([CFG['nu1_external'],CFG['nu2_external']],float)

@njit(cache=True)
def interp(a,t,dt):
    z=max(0.,min(t/dt,len(a)-1.))
    i=min(int(z),len(a)-2)
    return a[i]*(1-(z-i))+a[i+1]*(z-i)

@njit(cache=True)
def rhs(x,r,b):
    S=x[:2]; Q=x[2:4]; I=x[4:6]; J=x[6:8]
    c=r[:2]; q=r[2:4]; m=r[4:6]; d=r[6:8]
    lam=c*S*I/N
    external=NU*S
    y=q*(b*lam+external)
    A=(1-q)*b*lam
    B=(1-q)*(1-b)*lam
    p=m*A+d*I
    u=m*B
    out=np.empty(12)
    out[:2]=-b*lam-external-q*(1-b)*lam+MU*Q-u+u[::-1]
    out[2:4]=q*(1-b)*lam-MU*Q
    out[4:6]=A+(1-q)*external-G*I-p+p[::-1]
    out[6:8]=y-G*J
    out[8:10]=G*(I+J)
    out[10:12]=y
    return out

@njit(cache=True)
def rollout(rates,rate_dt,beta,days,substeps=20,start=-1.,duration=14.,strength=.5):
    h=1./substeps
    states=np.empty((days*substeps+1,12)); states[0]=X0
    x=X0.copy()
    for j in range(days*substeps):
        t=j*h
        r1=interp(rates,t,rate_dt); rm=interp(rates,t+h/2,rate_dt); r4=interp(rates,t+h,rate_dt)
        # Step midpoint defines interval membership, avoiding RK endpoint leakage.
        if start>=0 and start<=t+h/2<start+duration:
            r1[4:8]*=1-strength; rm[4:8]*=1-strength; r4[4:8]*=1-strength
        k1=rhs(x,r1,beta); k2=rhs(x+h*k1/2,rm,beta)
        k3=rhs(x+h*k2/2,rm,beta); k4=rhs(x+h*k3,r4,beta)
        x=x+h*(k1+2*k2+2*k3+k4)/6
        states[j+1]=x
    return states

@njit(cache=True)
def forcing_rhs(x,y,qmd,b):
    q=qmd[:2]; m=qmd[2:4]; d=qmd[4:6]
    external=NU*x[:2]
    if np.any(q<=0) or np.any(y-q*external<0):return np.full(12,np.nan)
    local=(y-q*external)/q
    A=(1-q)*(local+external)
    B=(1-q)*(1-b)*local/b
    p=m*(1-q)*local+d*x[4:6]; u=m*B
    out=np.empty(12)
    out[:2]=-(local+external)-q*(1-b)*local/b+MU*x[2:4]-u+u[::-1]
    out[2:4]=q*(1-b)*local/b-MU*x[2:4]
    out[4:6]=A-G*x[4:6]-p+p[::-1]
    out[6:8]=y-G*x[6:8]
    out[8:10]=G*(x[4:6]+x[6:8])
    out[10:12]=y
    return out

def read_baseline():
    cfg=json.loads((BASE/'config.json').read_text())
    if not (cfg['gamma_fixed']==G and cfg['mu_fixed']==MU):
        raise AssertionError("model.py: scientific/integrity check failed: cfg['gamma_fixed']==G and cfg['mu_fixed']==MU")
    if not (cfg['N1_raw']==N[0] and cfg['N2_raw']==N[1]):
        raise AssertionError("model.py: scientific/integrity check failed: cfg['N1_raw']==N[0] and cfg['N2_raw']==N[1]")
    if not (all(cfg[k]==v for k,v in [('Sq10_raw',0),('Sq20_raw',4),('I10_raw',0),('I20_raw',1),('Iq10_raw',0),('Iq20_raw',0),('R10_raw',0),('R20_raw',0)])):
        raise AssertionError("model.py: scientific/integrity check failed: all(cfg[k]==v for k,v in [('Sq10_raw',0),('Sq20_raw',4),('I10_raw',0),('I20_raw',1),('Iq10_raw',0),('Iq20_raw',0),('R10_raw',0),('R20_raw',0)])")
    for i,name in enumerate(NAMES):
        if not (cfg[name+'_bounds']==[LOW[i],HIGH[i]]):
            raise AssertionError("model.py: scientific/integrity check failed: cfg[name+'_bounds']==[LOW[i],HIGH[i]]")
    rates=np.column_stack([np.loadtxt(BASE/(k+'.txt')) for k in NAMES])
    return rates,float(np.loadtxt(BASE/'beta.txt')),cfg

def metrics(y,p):
    e=p-y
    return dict(R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))))

def fluxes(states,rates,b):
    lam=rates[:,:2]*states[:,:2]*states[:,4:6]/N
    y=rates[:,2:4]*(b*lam+NU*states[:,:2])
    A=(1-rates[:,2:4])*b*lam
    contact=rates[:,4:6]*A
    migration=rates[:,6:8]*states[:,4:6]
    return y,A,contact,migration,contact+migration
