import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from scripts.core.model import *
import pandas as pd
from scipy.integrate import solve_ivp
ROOT=CTX.root;PROJECT=CTX.project
PROTOCOL=json.loads(CTX.protocol.read_text())
T=int(PROTOCOL['horizon'])
STRATEGIES=[('N','N'),('H','N'),('L','N'),('N','H'),('N','L'),('H','H'),('H','L'),('L','H'),('L','L')]
EVENTS=np.array(sorted(set([0.,float(T)]+[v for reg in ['GD','GX'] for pair in PROTOCOL['regions'][reg]['high_intervals'] for v in pair])))

def active(t,j,s):
    h=any(a<=t<b for a,b in PROTOCOL['regions'][['GD','GX'][j]]['high_intervals'])
    return s=='H' and h or s=='L' and not h

def grid(dt):return np.unique(np.r_[np.arange(round(T/dt)+1)*dt,EVENTS])

def factors(tt,s):
    mid=(tt[1:]+tt[:-1])/2;f=np.ones((len(mid),8))
    for j in range(2):
        ac=np.array([active(t,j,s[j]) for t in mid]);f[:,4+j]=f[:,6+j]=np.where(ac,.2,1)
    return f

@njit(cache=True)
def aug_rhs(x,r,b):
    out=np.empty(14);out[:12]=rhs(x[:12],r,b);out[12:]=b*r[:2]*x[:2]*x[4:6]/N+NU*x[:2];return out

@njit(cache=True)
def simulate(r,dt,b,tt,fac,hmax,history=True):
    x=np.zeros(14);x[:12]=X0
    xx=np.empty((len(tt) if history else 1,14));xx[0]=x;minstate=0.;masserr=0.
    for j in range(len(tt)-1):
        start,end=tt[j],tt[j+1];ns=max(1,int(np.ceil((end-start)/hmax-1e-9)));h=(end-start)/ns
        for k in range(ns):
            t=start+k*h;r1=interp(r,t,dt)*fac[j];rm=interp(r,t+h/2,dt)*fac[j];r4=interp(r,t+h,dt)*fac[j]
            a=aug_rhs(x,r1,b);bb=aug_rhs(x+h*a/2,rm,b);c=aug_rhs(x+h*bb/2,rm,b);d=aug_rhs(x+h*c,r4,b)
            x=x+h*(a+2*bb+2*c+d)/6
            minstate=min(minstate,x[:10].min());masserr=max(masserr,abs(x[:10].sum()-N.sum()))
        if history:xx[j+1]=x
    if not history:xx[0]=x
    return xx,minstate,masserr

def classify(a,s,tol=.01):
    pos=a>tol;neg=a< -tol
    if pos.all():return 'win-win'
    if neg.all():return 'lose-lose'
    if pos.any() and neg.any():
        if (s[0]=='N') != (s[1]=='N'):
            j=0 if s[0]!='N' else 1
            if neg[j] and pos[1-j]:return 'self-harm / neighbor-benefit'
        return 'burden shifting'
    if (abs(a)<=tol).all():return 'neutral'
    return 'mixed with neutral'

def make_independent_rhs():
    from scripts.core.independent_rhs import rhs as independent
    return independent


def dop853(r,dt,b,s):
    independent=make_independent_rhs();x=np.zeros(14);x[:12]=X0;daily=np.empty((T+1,14));daily[0]=x
    for a,z in zip(EVENTS[:-1],EVENTS[1:]):
        f=np.ones(8)
        for j in range(2):
            if active((a+z)/2,j,s[j]):f[4+j]=f[6+j]=.2
        rd=np.ascontiguousarray((r*f).T)
        def fun(t,x):
            out=np.empty(14);out[:12]=independent(t/dt,x[:12],rd,b,G,MU,N[0],N[1])
            # Diagnostic all-infection incidence, c unchanged by the operator.
            c=interp(r,t,dt)[:2]
            out[12:]=b*c*x[:2]*x[4:6]/N+NU*x[:2];return out
        sol=solve_ivp(fun,(a,z),x,method='DOP853',rtol=2e-12,atol=1e-11,max_step=.05,dense_output=True)
        if not (sol.success and sol.y[:10].min()>-1e-8):
            raise AssertionError('solvers.py: scientific/integrity check failed: sol.success and sol.y[:10].min()>-1e-8')
        ii=np.arange(int(np.ceil(a)),int(np.floor(z))+1)
        if len(ii):daily[ii]=sol.sol(ii).T
        x=sol.y[:,-1]
    return daily

def dop853_strategy(r,dt,b,s):
    independent=make_independent_rhs();tt=grid(dt);fac=factors(tt,s);x=np.zeros(14);x[:12]=X0
    daily=np.empty((T+1,14));daily[0]=x;arrays={}
    for f in fac:
        key=(float(f[4]),float(f[5]))
        if key not in arrays:arrays[key]=np.ascontiguousarray((r*f).T)
    for k,(a,z) in enumerate(zip(tt[:-1],tt[1:])):
        rd=arrays[(float(fac[k,4]),float(fac[k,5]))]
        def fun(t,x):
            out=np.empty(14);out[:12]=independent(t/dt,x[:12],rd,b,G,MU,N[0],N[1])
            c=interp(r,t,dt)[:2];out[12:]=b*c*x[:2]*x[4:6]/N+NU*x[:2]
            return out
        sol=solve_ivp(fun,(a,z),x,method='DOP853',rtol=1e-13,atol=1e-12,max_step=z-a)
        if not (sol.success and sol.y.min()>=0):
            raise AssertionError('solvers.py: scientific/integrity check failed: sol.success and sol.y.min()>=0')
        x=sol.y[:,-1]
        if abs(z-round(z))<1e-10:daily[round(z)]=x
    return daily
