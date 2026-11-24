import math
import numpy as np
from numba import njit
@njit(cache=True)
def interp(arr,t):
    if t<=0: return float(arr[0])
    if t>=len(arr)-1: return float(arr[-1])
    j=int(math.floor(t)); a=t-j
    return float((1-a)*arr[j]+a*arr[j+1])

@njit(cache=True)
def rhs(t,x,r,beta,gamma,mu,N1,N2):
    S1,S2,Sq1,Sq2,I1,I2,Iq1,Iq2,R1,R2,C1,C2=x
    c1,c2,q1,q2,m1,m2,d1,d2=[interp(r[j],t) for j in range(8)]
    lam1=c1*S1*I1/N1; lam2=c2*S2*I2/N2
    dS1=-lam1+(1-q1)*(1-beta)*(1-m1)*lam1+mu*Sq1+(1-q2)*(1-beta)*m2*lam2
    dS2=-lam2+(1-q2)*(1-beta)*(1-m2)*lam2+mu*Sq2+(1-q1)*(1-beta)*m1*lam1
    dSq1=q1*(1-beta)*lam1-mu*Sq1; dSq2=q2*(1-beta)*lam2-mu*Sq2
    dI1=(1-q1)*beta*(1-m1)*lam1-gamma*I1+(1-q2)*beta*m2*lam2-d1*I1+d2*I2
    dI2=(1-q2)*beta*(1-m2)*lam2-gamma*I2+(1-q1)*beta*m1*lam1-d2*I2+d1*I1
    dIq1=q1*beta*lam1-gamma*Iq1; dIq2=q2*beta*lam2-gamma*Iq2
    dR1=gamma*(I1+Iq1); dR2=gamma*(I2+Iq2)
    dC1=q1*beta*lam1; dC2=q2*beta*lam2
    return np.array([dS1,dS2,dSq1,dSq2,dI1,dI2,dIq1,dIq2,dR1,dR2,dC1,dC2],np.float64)
