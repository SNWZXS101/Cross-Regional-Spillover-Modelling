"""M2 equations and inverse identity in raw persons, no clipping."""
import numpy as np

def rhs14(x,r,beta,gamma,mu,N,nu):
 x=np.asarray(x,float);r=np.asarray(r,float);N=np.asarray(N,float);nu=np.asarray(nu,float)
 S=x[:2];Q=x[2:4];I=x[4:6];Iq=x[6:8];c=r[:2];q=r[2:4];m=r[4:6];d=r[6:8]
 f=beta*c*S*I/N;u=nu*S;observed=q*(f+u);A=(1-q)*f; B=(1-q)*(1-beta)*f/beta;P=m*A+d*I;Z=m*B
 out=np.empty(14);out[:2]=-(f+u)-q*(1-beta)*f/beta+mu*Q-Z+Z[::-1];out[2:4]=q*(1-beta)*f/beta-mu*Q
 out[4:6]=(1-q)*(f+u)-gamma*I-P+P[::-1];out[6:8]=observed-gamma*Iq;out[8:10]=gamma*(I+Iq);out[10:12]=observed;out[12:14]=f+u
 return out

def inverse_rhs12(x,y,qmd,beta,gamma,mu,nu):
 x=np.asarray(x,float);y=np.asarray(y,float);q=np.asarray(qmd[:2],float);m=np.asarray(qmd[2:4],float);d=np.asarray(qmd[4:6],float);u=np.asarray(nu,float)*x[:2]
 if np.any(q<=0) or np.any(y-q*u<0):raise ValueError('Negative y-q*u or nonpositive q: infeasible')
 f=(y-q*u)/q;A=(1-q)*(f+u);B=(1-q)*(1-beta)*f/beta;P=m*(1-q)*f+d*x[4:6];Z=m*B
 out=np.empty(12);out[:2]=-(f+u)-q*(1-beta)*f/beta+mu*x[2:4]-Z+Z[::-1];out[2:4]=q*(1-beta)*f/beta-mu*x[2:4]
 out[4:6]=A-gamma*x[4:6]-P+P[::-1];out[6:8]=y-gamma*x[6:8];out[8:10]=gamma*(x[4:6]+x[6:8]);out[10:12]=y
 return out
