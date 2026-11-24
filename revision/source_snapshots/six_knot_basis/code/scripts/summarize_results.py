"""Common N-solution summary contract; all computation uses saved artifacts."""
from scripts.common import *
import itertools,platform,importlib.metadata

RATE_NAMES=['c1','c2','q1','q2','m1','m2','d1','d2']

def summarize_policy(df,n):
    rows=[]
    for (measure,strategy),d in df.groupby(['measure','strategy']):
        row=dict(measure=measure,strategy=strategy,n=n)
        for label in ['win-win','self-harm / neighbor-benefit','burden shifting','lose-lose','neutral','mixed with neutral']:
            key=label.replace(' / ','_').replace('-','_').replace(' ','_');row[key+'_count']=int(d.classification.eq(label).sum());row[key+'_fraction']=float(d.classification.eq(label).mean())
        for col in ['A1','A2','Atotal']:
            row[col+'_positive_count']=int((d[col]>.01).sum());row[col+'_negative_count']=int((d[col]<-.01).sum())
            for stat in ['min','median','max']:row[col+'_'+stat]=float(getattr(d[col],stat)())
        rows.append(row)
    return pd.DataFrame(rows)

def boundary_rows(r,lo,hi,name):
    z=(r-lo)/(hi-lo);rows=[]
    def fraction(x,a):
        left=x[:-1];right=x[1:];out=np.zeros(len(left));out[(left<=a)&(right<=a)]=1
        cross=(left<a)&(right>a);out[cross]=(a-left[cross])/(right[cross]-left[cross])
        cross=(left>a)&(right<a);out[cross]=(a-right[cross])/(left[cross]-right[cross])
        return float(out.mean())
    for j,k in enumerate(RATE_NAMES):
        lower=fraction(z[:,j],.02);upper=fraction(1-z[:,j],.02)
        rows.append(dict(solution=name,parameter=k,lower_bound=lo[j],upper_bound=hi[j],near_lower_fraction=lower,near_upper_fraction=upper,total_boundary_fraction=lower+upper,threshold_fraction=.02,raw_min=float(r[:,j].min()),raw_max=float(r[:,j].max())))
    return rows

def summarize_functions(ctx):
    cfg=read_json(ctx.base/'config.json');lo=np.array([cfg[k+'_bounds'][0] for k in RATE_NAMES]);hi=np.array([cfg[k+'_bounds'][1] for k in RATE_NAMES]);pop=np.array([cfg['N1_raw'],cfg['N2_raw']]);rates=[];flux=[];occ=[];smooth=[];totals=[]
    for name in ids(ctx.n):
        a=np.load(ctx.ensemble_root/'solutions'/(name+'.npz'));r=a['rates'];x=a['states'];dt=float(a['dt']);b=float(a['beta']);t=np.arange(len(r))*dt;T=t[-1]
        if r.shape!=(len(x),8):raise ValueError('Stored rate/state grids differ')
        phi=r[:,4:6]*(1-r[:,2:4])*b*r[:,:2]*x[:,:2]*x[:,4:6]/pop+r[:,6:8]*x[:,4:6]
        rates.append(r);flux.append(phi);occ.extend(boundary_rows(r,lo,hi,name))
        for j,k in enumerate(RATE_NAMES):
            dr=np.diff(r[:,j]/(hi[j]-lo[j]));smooth.append(dict(solution=name,parameter=k,normalized_total_variation=float(abs(dr).sum()),normalized_slope_RMS=float(np.sqrt(np.mean((dr/dt)**2)))))
        for j,label in enumerate(['GD_to_GX','GX_to_GD']):totals.append(dict(solution=name,direction=label,cumulative_spillover=float(np.trapezoid(phi[:,j],t)),peak_flux=float(phi[:,j].max())))
    raw=[];fr=[]
    for i,j in itertools.combinations(range(ctx.n),2):
        z=(rates[i]-rates[j])/(hi-lo);per=np.sqrt(np.sum((z[:-1]**2+z[:-1]*z[1:]+z[1:]**2)*dt/3,axis=0)/T)
        for k,p in enumerate(RATE_NAMES):raw.append(dict(solution_a=ids(ctx.n)[i],solution_b=ids(ctx.n)[j],parameter=p,normalized_RMS=float(per[k])))
        for k,label in enumerate(['GD_to_GX','GX_to_GD']):
            num=np.sqrt(np.trapezoid((flux[i][:,k]-flux[j][:,k])**2,t)/T);den=.5*(np.sqrt(np.trapezoid(flux[i][:,k]**2,t)/T)+np.sqrt(np.trapezoid(flux[j][:,k]**2,t)/T))
            fr.append(dict(solution_a=ids(ctx.n)[i],solution_b=ids(ctx.n)[j],direction=label,symmetric_normalized_RMS=float(num/den) if den else 0.))
    for name,rows in [('raw_parameter_pairwise_rms',raw),('flux_pairwise_rms',fr),('boundary_occupancy',occ),('smoothness_diagnostics',smooth),('flux_summary',totals)]:pd.DataFrame(rows).to_csv(ctx.results/(name+'.csv'),index=False)
    np.savez_compressed(ctx.results/'function_arrays.npz',time=t,names=ids(ctx.n),rates=rates,fluxes=flux)

def main():
    ctx=Context();df=validate_scenarios(pd.read_csv(ctx.results/'complete_scenarios.csv'),ctx.n);summary=summarize_policy(df,ctx.n)
    summary.to_csv(ctx.results/'policy_summary.csv',index=False)
    if ctx.mode=='reuse_frozen':
        old=pd.read_csv(ctx.results/'archived_policy_summary.csv').sort_values(['measure','strategy']).reset_index(drop=True)
        pd.testing.assert_frame_equal(summary[old.columns].sort_values(['measure','strategy']).reset_index(drop=True),old,check_dtype=False,rtol=1e-12,atol=1e-8)
        atomic_json(ctx.results/'archived_summary_equivalence.json',dict(passed=True,numerical_tolerance=1e-8,original_results_modified=False))
    summarize_functions(ctx)
    atomic_json(ctx.root/'methods/runtime.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','numba','matplotlib','openpyxl']}))
    print('Saved common summaries for',ctx.n,'members; calculations use stored arrays only')

if __name__=='__main__':main()
