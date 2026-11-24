from scripts.core.model import *
@njit(cache=True)
def inverse_profile(qmd,y,dt,b,r0c):
    nn=(len(y)-1)//2;x=X0.copy();states=np.empty((nn+1,12));states[0]=x
    for k in range(nn):
        j=2*k
        k1=forcing_rhs(x,y[j],qmd[j],b)
        k2=forcing_rhs(x+dt*k1/2,y[j+1],qmd[j+1],b)
        k3=forcing_rhs(x+dt*k2/2,y[j+1],qmd[j+1],b)
        k4=forcing_rhs(x+dt*k3,y[j+2],qmd[j+2],b)
        x=x+dt*(k1+2*k2+2*k3+k4)/6;states[k+1]=x
    r=np.empty((nn+1,8));r[:,2:]=qmd[::2]
    for k in range(nn+1):
        for i in range(2):
            den=b*r[k,2+i]*states[k,i]*states[k,4+i]
            # Preserve operation order matching y=q*(local+nu*S).
            residual=y[2*k,i]-r[k,2+i]*(NU[i]*states[k,i])
            if den>0 and residual>=0:
                r[k,i]=residual*N[i]/den
            elif k==0 and den==0 and residual==0:
                # Initial protected point only: contact is unidentifiable when residual is exactly zero.
                r[k,i]=r0c[i]
            else:
                r[k,i]=np.nan
    return r,states
