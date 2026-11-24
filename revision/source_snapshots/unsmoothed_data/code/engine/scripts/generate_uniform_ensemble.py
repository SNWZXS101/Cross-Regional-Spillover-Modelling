"""Policy-blind, uniformly applied constrained maximin construction only."""
from scripts.common import *
if Context().mode=='fresh':
    from scripts.core.solvers import *
if Context().mode=='fresh':
    from scripts.core.inverse import inverse_profile
if Context().mode=='fresh':
    from scipy.interpolate import CubicSpline
    from scipy.special import expit,logit
    from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import time,sys
from scripts.core.o2_gates import candidate_gates
from scripts.adaptive_search import StartCache,clean,digest,round_sizes,run_adaptive

BASELINE_AUDIT=json.loads(Path(json.loads((ROOT/'results/baseline_audit.json').read_text())['source']).read_text())
if not BASELINE_AUDIT['accepted']:raise RuntimeError('O2 baseline not accepted')
DT=.025
ITERATIONS=Context().config['max_iterations']
NSTARTS=Context().config['n_starts']
NSOLUTIONS=Context().n
SEED=Context().config['seed']
MIN_NOVELTY=.025
SCALES=[1.,.99,.97,.94,.9,.8,.65,.5,.3,.1,0.]

def distance(a,b,dt):
    z=(a-b)/(HIGH-LOW)
    return float(np.sqrt(np.sum((z[:-1]**2+z[:-1]*z[1:]+z[1:]**2)*dt/3,axis=0)/T).mean())

class UniformSearch:
    def __init__(self,bank):
        self.r0,self.beta,_=read_baseline();self.bank=bank;self.nfev=0;self.best=None;self.history=[];self.cache={};self.started=time.time()
        self.tb=np.arange(T*80+1)/80;rb=np.array([interp(self.r0,t,1.) for t in self.tb]);xb=rollout(self.r0,1.,self.beta,T,80)
        self.yb=fluxes(xb,rb,self.beta)[0];self.obs=pd.read_excel(PROJECT/CFG['data_file'])[[CFG['gd_column'],CFG['gx_column']]].values
        self.reference=rb[::2]
        self.geometry={}
    def construct(self,z,dt=.25):
        if dt not in self.geometry:
            t=np.arange(round(T*2/dt)+1)*dt/2;w=np.clip((t-7)/14,0,1);w=w*w*(3-2*w)
            basis=CubicSpline(np.linspace(0,T,4),np.eye(4),axis=0,bc_type='natural')(t)*w[:,None]
            yy=np.column_stack([np.interp(t,self.tb,self.yb[:,j]) for j in range(2)])
            self.geometry[dt]=(t,w,basis,yy)
        t,w,B,yy=self.geometry[dt];qm=np.empty((len(t),6))
        for j in range(6):
            warped=np.interp(t+w*z[24+j]*np.sin(np.pi*t/T),np.arange(len(self.r0)),self.r0[:,j+2])
            f=(warped-LOW[j+2])/(HIGH[j+2]-LOW[j+2])
            eps=1e-6
            f=eps+(1-2*eps)*f
            qm[:,j]=LOW[j+2]+(HIGH[j+2]-LOW[j+2])*expit(logit(f)+B@z[:24].reshape(4,6)[:,j])
        return inverse_profile(qm,yy,dt,self.beta,self.r0[0,:2])
    def evaluate(self,z):
        key=z.tobytes()
        if key in self.cache:return self.cache[key]
        self.nfev+=1;r,x=self.construct(z);cv=(r[:,:2]-LOW[:2])/(HIGH[:2]-LOW[:2])
        con=np.r_[cv.min(0)-2e-5,1-cv.max(0)-2e-5,x[:,4:6].min(0)/1000]
        if not np.isfinite(r).all() or not np.isfinite(x).all() or abs(r).max()>1000 or x[:,:10].min()<-1:
            result=(1e5+float(np.minimum(np.nan_to_num(con,nan=-1e4),0)@np.minimum(np.nan_to_num(con,nan=-1e4),0)),np.r_[np.nan_to_num(con,nan=-1e4),np.full(10,-1.)],None)
        else:
            xx=rollout(r,.25,self.beta,T,4);y=np.diff(xx[::4,10:12],axis=0)
            fits=np.array([metrics(self.obs[:,j],y[:,j])['R2'] for j in range(2)])
            try:og,_=candidate_gates(self.obs,y,CFG,BASELINE_AUDIT)
            except ValueError:og=np.full(10,-1.)
            cons=np.r_[con,og];value=min(distance(r,a,.25) for a in self.bank)
            result=(-value,cons,dict(distance=value,R2_GD=fits[0],R2_GX=fits[1]))
            if cons.min()>=-1e-7 and (self.best is None or value>self.best['distance']):
                self.best=dict(z=z.copy(),**result[2]);self.history.append(dict(nfev=self.nfev,seconds=time.time()-self.started,**result[2]))
        self.cache.clear();self.cache[key]=result;return result
    def optimize(self,k,trial):
        z=initial_coordinates(k,trial,SEED)
        proposed=z.copy()
        for scale in [1.,.5,.25,.1,0.]:
            if self.evaluate(z*scale)[1].min()>=-1e-7:z=z*scale;break
        initial=z.copy();v=self.evaluate(z)[0]
        opt=minimize(lambda z:self.evaluate(z)[0],z,method='SLSQP',bounds=[(-12,12)]*24+[(-21,21)]*6,
            constraints=[{'type':'ineq','fun':lambda z:self.evaluate(z)[1]}],options={'maxiter':ITERATIONS,'ftol':2e-6,'eps':1e-4})
        self.evaluate(opt.x)
        info=dict(k=k,trial=trial,seed=SEED,source='UDE baseline only',proposed_initial_coordinates=proposed.tolist(),actual_initial_coordinates=initial.tolist(),initial_objective=float(v),stop_reason=str(opt.message),optimizer_success=bool(opt.success),iterations=int(opt.nit),nfev=self.nfev,seconds=time.time()-self.started,feasible_incumbent=self.best is not None,progress=self.history)
        if self.best:info['best_coordinates']=self.best['z'].tolist();info['best_search_distance']=self.best['distance']
        return self.best,info

def initial_coordinates(k,trial,seed):
    # Default reproduces the original deterministic v2 starts exactly.
    phase=0. if seed==31415 else float(np.random.default_rng(np.random.SeedSequence([seed,k,trial])).uniform(0,2*np.pi))
    z=np.zeros(30)
    z[:24]=.25*np.sin(np.arange(24)*1.17+k+trial*2.3+phase)
    z[24:]=1.5*np.cos(np.arange(6)*.9+k*.7+trial+phase)
    return z

def proposal(args):
    k,trial,bank,context,number=args
    obj=UniformSearch(bank)
    try:best,info=obj.optimize(k,trial)
    except Exception as e:
        atomic_json(ROOT/'logs'/f'U{k:02d}_start{trial}_interrupted.json',clean(dict(
            context=context,k=k,trial=trial,seed=SEED,search_round=number,
            proposed_initial_coordinates=initial_coordinates(k,trial,SEED),
            status='interrupted_not_completed',stop_reason=repr(e),
            best_candidate=getattr(obj,'best',None),nfev=getattr(obj,'nfev',0),progress=getattr(obj,'history',[]))))
        raise
    phase=0. if SEED==31415 else float(np.random.default_rng(np.random.SeedSequence([SEED,k,trial])).uniform(0,2*np.pi))
    info.update(search_round=number,_max_iterations=ITERATIONS,
                start_identity=dict(seed_sequence_entropy=[SEED,k,trial],phase=phase,
                                    sine_phase=k+trial*2.3+phase,cosine_phase=k*.7+trial+phase,
                                    proposed_coordinates_sha256=digest(info['proposed_initial_coordinates'])),
                best_candidate=clean(best),optimization_completed=True)
    cache=StartCache(ROOT,k,context);cache.save(trial,info)
    return best,cache.load(trial)

def high_precision_check(r,b,obs):
    cv=(r-LOW)/(HIGH-LOW)
    if not np.isfinite(r).all() or cv.min()<0 or cv.max()>1 or not BETA_BOUNDS[0]<=b<=BETA_BOUNDS[1]:return None
    x=rollout(r,DT,b,T,80);coarse=rollout(r,DT,b,T,40);daily=x[::80];y=np.diff(daily[:,10:12],axis=0)
    fits=[metrics(obs[:,j],y[:,j])['R2'] for j in range(2)];err=float(abs(daily[:,10:12]-coarse[::40,10:12]).max())
    mass=float(abs(x[:,:10].sum(axis=1)-N.sum()).max())
    if not np.isfinite(x).all() or not np.isfinite(coarse).all():return None
    try:gates,rkdiag=candidate_gates(obs,y,CFG,BASELINE_AUDIT)
    except ValueError:return None
    if np.min(gates)<0 or x.min()<0 or err>.01 or mass>.01:return None
    di=dop853(r,DT,b,('N','N'));dy=np.diff(di[:,10:12],axis=0);dfits=[metrics(obs[:,j],dy[:,j])['R2'] for j in range(2)]
    derr=float(abs(di[:,:12]-daily).max());caseerr=float(abs(di[:,10:12]-daily[:,10:12]).max())
    try:dgates,dopdiag=candidate_gates(obs,dy,CFG,BASELINE_AUDIT)
    except ValueError:return None
    if np.min(dgates)<0 or not np.isfinite(di).all() or di.min()<0 or np.max(np.abs(di[:,:10].sum(axis=1)-N.sum()))>.01 or caseerr>.01:return None
    smooth=[]
    for j in range(8):
        z=r[:,j]/(HIGH[j]-LOW[j]);smooth.append(dict(parameter=NAMES[j],normalized_total_variation=float(abs(np.diff(z)).sum()),normalized_slope_RMS=float(np.sqrt(np.mean((np.diff(z)/DT)**2)))))
    return dict(raw_fit_RK4=rkdiag,raw_fit_DOP853=dopdiag,R2_GD_RK4=fits[0],R2_GX_RK4=fits[1],R2_GD_DOP853=dfits[0],R2_GX_DOP853=dfits[1],min_state=float(x.min()),mass_error=mass,bound_min=float(cv.min()),bound_max=float(cv.max()),beta=b,RK4_step_error=err,DOP853_cumulative_error=caseerr,DOP853_state_max_error=derr,states=x[::2],DOP853_daily_states=di,smoothness=smooth)

def check_with_diagnostics(r,b,obs):
    """Observe the original checker without changing its code, criteria or solvers."""
    calls={};original_rollout=globals()['rollout'];original_dop=globals()['dop853']
    def capture_rollout(*args,**kwargs):
        value=original_rollout(*args,**kwargs);calls[args[4]]=value;return value
    def capture_dop(*args,**kwargs):
        value=original_dop(*args,**kwargs);calls['dop']=value;return value
    globals()['rollout']=capture_rollout;globals()['dop853']=capture_dop
    try:result=high_precision_check(r,b,obs)
    finally:globals()['rollout']=original_rollout;globals()['dop853']=original_dop
    cv=(r-LOW)/(HIGH-LOW);reasons=[]
    details=dict(bound_min=float(cv.min()),bound_max=float(cv.max()),beta=float(b))
    if not np.isfinite(r).all():reasons.append('nonfinite_parameters')
    if cv.min()<0 or cv.max()>1:reasons.append('parameter_bounds')
    if not BETA_BOUNDS[0]<=b<=BETA_BOUNDS[1]:reasons.append('beta_bounds')
    if 80 in calls:
        x=calls[80];daily=x[::80];y=np.diff(daily[:,10:12],axis=0)
        fits=[metrics(obs[:,j],y[:,j])['R2'] for j in range(2)]
        err=float(abs(daily[:,10:12]-calls[40][::40,10:12]).max())
        mass=float(abs(x[:,:10].sum(axis=1)-N.sum()).max())
        details.update(R2_RK4=fits,RK4_min_state=float(x.min()),RK4_step_error=err,mass_error=mass)
        try:
            gates,diagnostics=candidate_gates(obs,y,CFG,BASELINE_AUDIT);details['raw_fit_RK4']=diagnostics
            if min(gates)<0:reasons.append('RK4_raw_observation_gate_failed')
        except ValueError:reasons.append('RK4_invalid_observation')
        if x.min()<0:reasons.append('RK4_negative_state')
        if err>.01:reasons.append('RK4_step_error_above_0.01')
        if mass>.01:reasons.append('population_error_above_0.01')
    if 'dop' in calls:
        di=calls['dop'];dy=np.diff(di[:,10:12],axis=0)
        fits=[metrics(obs[:,j],dy[:,j])['R2'] for j in range(2)]
        err=float(abs(di[:,10:12]-daily[:,10:12]).max())
        details.update(R2_DOP853=fits,DOP853_min_state=float(di[:,:10].min()),DOP853_cumulative_error=err)
        try:
            gates,diagnostics=candidate_gates(obs,dy,CFG,BASELINE_AUDIT);details['raw_fit_DOP853']=diagnostics
            if min(gates)<0:reasons.append('DOP853_raw_observation_gate_failed')
        except ValueError:reasons.append('DOP853_invalid_observation')
        if di[:,:10].min()<0:reasons.append('DOP853_negative_state')
        if err>.01:reasons.append('DOP853_case_error_above_0.01')
    return result,dict(status='accepted' if result is not None else 'rejected',reasons=reasons,metrics=details)


def checked_validation(k,trial,scale_index,r,b,obs,context):
    stem=f'U{k:02d}_validation_start{trial}_scale{scale_index}'
    marker=ROOT/'checkpoints'/(stem+'.json');data=ROOT/'checkpoints'/(stem+'.npz')
    key=dict(context=context,rates_sha256=hashlib.sha256(r.tobytes()).hexdigest(),beta=float(b))
    if marker.exists():
        saved=read_json(marker)
        if saved['key']!=key or saved['payload_sha256']!=digest({x:v for x,v in saved.items() if x!='payload_sha256'}):raise RuntimeError('Validation checkpoint mismatch: '+stem)
        v=saved['verification']
        if v is not None:
            if sha(data)!=saved['data_sha256']:raise RuntimeError('Validation state checksum mismatch: '+stem)
            with np.load(data) as arrays:v.update(states=arrays['states'],DOP853_daily_states=arrays['DOP853_daily_states'])
        return v,saved['diagnostic']
    try:v,diagnostic=check_with_diagnostics(r,b,obs)
    except Exception as e:
        atomic_json(ROOT/'logs'/(stem+'_error.json'),dict(key=key,status='interrupted',reason=repr(e)))
        raise
    saved=dict(key=key,diagnostic=clean(diagnostic),verification=None)
    if v is not None:
        temp=data.with_name(data.name+'.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,states=v['states'],DOP853_daily_states=v['DOP853_daily_states'])
        os.replace(temp,data)
        saved.update(verification={x:y for x,y in v.items() if x not in ['states','DOP853_daily_states']},data_sha256=sha(data))
    saved=clean(saved);saved['payload_sha256']=digest(saved);atomic_json(marker,saved)
    return v,diagnostic


def screening_reasons(row):
    if not row['fine_feasible']:return ['fine_grid_parameter_or_hidden_state_infeasible']
    reasons=[]
    if row['distance']<=MIN_NOVELTY:reasons.append('distance_to_reference_not_above_0.025')
    if row['baseline_distance']<=MIN_NOVELTY:reasons.append('distance_to_baseline_not_above_0.025')
    return reasons


def search_member(k,bank,baseline,obj,pool):
    cfg=Context().config;sizes=round_sizes(cfg);ref=bank if bank else [baseline]
    context=dict(version='adaptive-v1',k=k,seed=SEED,max_iterations=ITERATIONS,
                 round_sizes=sizes,inputs=digest(read_json(ROOT/'methods/source_hashes.json')['inputs']),
                 reference_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in ref])
    upgrade=ROOT/'methods/adaptive_upgrade.json'
    legacy=read_json(upgrade)['legacy_files'] if upgrade.exists() else {}
    cache=StartCache(ROOT,k,context,legacy)
    all_screen=[]

    def fetch(number,trials):
        found={};missing=[]
        for trial in trials:
            info=cache.load(trial)
            if info is None:missing.append(trial);continue
            np.testing.assert_array_equal(info['proposed_initial_coordinates'],initial_coordinates(k,trial,SEED),err_msg='Cached start identity changed')
            best={'z':np.array(info['best_coordinates'])} if info['feasible_incumbent'] else None
            found[trial]=(best,dict(info,_max_iterations=ITERATIONS))
        print(f'U{k:02d} round {number}: {len(found)} verified starts reused; {len(missing)} new starts {missing}',flush=True)
        jobs=[(k,j,[a[::10] for a in ref],context,number) for j in missing]
        # Each worker commits its result before returning. A stopped pool cannot
        # erase completed work, and worker scheduling cannot change trial order.
        for best,info in pool.map(proposal,jobs):found[info['trial']]=(best,info)
        return [found[j] for j in trials]

    def assess(number,proposals):
        stem=f'U{k:02d}_round{number}';marker=ROOT/'checkpoints'/(stem+'_failed.json')
        report=ROOT/'logs'/(stem+'_selection.json')
        if marker.exists():
            saved=read_json(marker)
            if saved['context']!=context:raise RuntimeError('Round checkpoint context mismatch')
            for rel,h in saved['files'].items():
                if sha(ROOT/rel)!=h:raise RuntimeError('Round checkpoint checksum mismatch: '+rel)
            all_screen.extend(read_json(report));return None
        oldscreen=ROOT/'logs'/f'U{k:02d}_selection_screen.json'
        legacy_screen=None
        if number==1 and str(oldscreen.relative_to(ROOT)) in legacy:
            if sha(oldscreen)!=legacy[str(oldscreen.relative_to(ROOT))]:raise RuntimeError('Legacy selection screen changed')
            rows=read_json(oldscreen)
            expected=[(info['trial'],scale) for best,info in proposals if best is not None for scale in SCALES]
            if [(r['trial'],r['scale']) for r in rows]!=expected:raise RuntimeError('Incomplete legacy round screen')
            if all(not row['eligible'] for row in rows):legacy_screen=rows
        candidates=[];screen=[]
        if legacy_screen is not None:
            screen=[dict(row,search_round=number,rejection_reasons=screening_reasons(row),final_status='rejected_at_screen',legacy_screen_reused=True) for row in legacy_screen]
        else:
            for best,info in proposals:
                if best is None:continue
                for scale_index,scale in enumerate(SCALES):
                    z=best['z']*scale;r,x=obj.construct(z,DT);cv=(r-LOW)/(HIGH-LOW)
                    feasible=bool(np.isfinite(r).all() and cv.min()>=0 and cv.max()<=1 and x[:,:10].min()>=0)
                    d=min(distance(r,a,DT) for a in ref) if feasible else None;novel=distance(r,baseline,DT) if feasible else None
                    eligible=feasible and d>MIN_NOVELTY and novel>MIN_NOVELTY
                    row=dict(trial=info['trial'],scale=scale,search_round=number,bound_min=float(cv.min()),bound_max=float(cv.max()),fine_hidden_min_state=float(x[:,:10].min()),fine_feasible=feasible,distance=d,baseline_distance=novel,eligible=bool(eligible))
                    row.update(rejection_reasons=screening_reasons(row),final_status='pending_validation' if eligible else 'rejected_at_screen')
                    screen.append(row)
                    if eligible:candidates.append((d,r,z,info,scale,novel,scale_index,row))
        candidates.sort(key=lambda v:-v[0]);accepted=None
        for d,r,z,info,scale,novel,scale_index,row in candidates:
            verification,diagnostic=checked_validation(k,info['trial'],scale_index,r,obj.beta,obj.obs,context)
            row.update(final_status=diagnostic['status'],validation=diagnostic,rejection_reasons=diagnostic['reasons'])
            if verification is not None:accepted=(d,r,z,info,scale,novel,verification);break
        for row in screen:
            if row['final_status']=='pending_validation':row['final_status']='not_evaluated_after_higher_distance_acceptance'
        atomic_json(report,clean(screen));all_screen.extend(screen)
        files=[report]
        for best,info in proposals:
            trial=info['trial'];rows=[row for row in screen if row['trial']==trial]
            outcome='selected' if accepted is not None and trial==accepted[3]['trial'] else 'no_coarse_feasible_incumbent' if best is None else 'not_selected'
            p=ROOT/'logs'/f'U{k:02d}_start{trial}_acceptance.json'
            atomic_json(p,clean(dict(search_round=number,trial=trial,outcome=outcome,
                         reason='no coarse feasible incumbent' if best is None else None,candidates=rows)))
            files.extend([p,cache.paths(trial)[0],cache.paths(trial)[1]])
        if accepted is None:
            atomic_json(marker,dict(context=context,status='failed_final_acceptance',files={str(p.relative_to(ROOT)):sha(p) for p in files}))
        # Keep legacy selection logs immutable; expanded audit has its own name.
        atomic_json(ROOT/'logs'/f'U{k:02d}_adaptive_selection_screen.json',clean(all_screen))
        return accepted

    def record(stats):atomic_json(ROOT/'logs'/f'U{k:02d}_search_summary.json',stats)
    return run_adaptive(k,sizes,fetch,assess,record)


def generate_fresh():
    if not (json.loads((ROOT/'results/baseline_audit.json').read_text())['passed']):
        raise AssertionError("generate_uniform_ensemble.py: scientific/integrity check failed: json.loads((ROOT/'results/baseline_audit.json').read_text())['passed']")
    bank=[];manifest=[]
    # Commit markers are written only after the member files are complete.
    # Uncommitted files are never members and may be overwritten on resume.
    for name in ids(NSOLUTIONS):
        marker=ROOT/'checkpoints'/(name+'_accepted.json')
        if not marker.exists():break
        for rel,h in read_json(marker)['files'].items():
            if sha(ROOT/rel)!=h:raise RuntimeError('Accepted member checksum mismatch: '+rel)
        p=ROOT/'solutions'/(name+'.npz')
        bank.append(np.load(p)['rates']);meta=read_json(p.with_suffix('.json'))
        # Old committed member JSONs stay byte-identical. Derive budget columns
        # only for the newly written manifest, preserving original evaluation totals.
        if 'search_rounds_used' not in meta:
            meta.update(search_rounds_used=1,search_starts_used=NSTARTS,adaptive_start_rounds=round_sizes(Context().config),max_iterations_per_start=ITERATIONS,budget_metadata_source='legacy first-round budget; committed JSON unchanged')
        manifest.append(meta)
    if any((ROOT/'checkpoints'/(name+'_accepted.json')).exists() for name in ids(NSOLUTIONS)[len(bank)+1:]):raise RuntimeError('Noncontiguous generation checkpoints')
    obj=UniformSearch([np.zeros((T*4+1,8))]);baseline=obj.reference
    with ProcessPoolExecutor(max_workers=min(NSTARTS,Context().config['workers'])) as pool:
        for k in range(len(bank)+1,NSOLUTIONS+1):
            accepted,search_stats=search_member(k,bank,baseline,obj,pool)
            d,r,z,info,scale,novel,v=accepted;name=f'U{k:02d}'
            meta={kk:vv for kk,vv in v.items() if kk not in ['states','DOP853_daily_states','smoothness']}
            meta.update(solution=name,generation_index=k,generation_rule='uniform policy-blind constrained maximin',source='new optimization from current UDE baseline',initialization_reference='baseline only for k=1' if k==1 else 'previous new uniform solutions only',selected_start=info['trial'],selection_distance=d,min_distance_to_prior_new=None if k==1 else d,baseline_distance=novel,optimizer_success=info['optimizer_success'],stop_reason=info['stop_reason'],iterations=info['iterations'],total_search_evaluations=search_stats['total_search_evaluations'],restoration_scale=scale,accepted=True)
            meta.update(search_stats)
            np.savez_compressed(ROOT/'solutions'/(name+'.npz'),rates=r,dt=DT,beta=obj.beta,states=v['states'],DOP853_daily_states=v['DOP853_daily_states'],coordinates=z)
            (ROOT/'solutions'/(name+'.json')).write_text(json.dumps(meta,indent=2));pd.DataFrame(v['smoothness']).to_csv(ROOT/'solutions'/(name+'_smoothness.csv'),index=False)
            atomic_json(ROOT/'checkpoints'/(name+'_accepted.json'),dict(files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'solutions'/(name+'.npz'),ROOT/'solutions'/(name+'.json'),ROOT/'solutions'/(name+'_smoothness.csv')]}))
            bank.append(r);manifest.append(meta);pd.DataFrame(manifest).to_csv(ROOT/'results/ensemble_manifest.csv',index=False)
            print('ACCEPTED',name,'distance',d,'R2 DOP853',v['R2_GD_DOP853'],v['R2_GX_DOP853'],flush=True)
    mat=np.zeros((NSOLUTIONS,NSOLUTIONS))
    for i in range(NSOLUTIONS):
        for j in range(i):mat[i,j]=mat[j,i]=distance(bank[i],bank[j],DT)
    labels=[m['solution'] for m in manifest];pd.DataFrame(mat,index=labels,columns=labels).to_csv(ROOT/'results/ensemble_distances.csv')
    for i,row in enumerate(manifest):row['min_distance_to_final_others']=float(np.min(np.delete(mat[i],i)))
    pd.DataFrame(manifest).to_csv(ROOT/'results/ensemble_manifest.csv',index=False)
    print('UNIFORM FAMILY COMPLETE',len(manifest),flush=True)


def main():
    ctx=Context()
    if ctx.mode=='fresh':
        if (ctx.root/'methods/ensemble_freeze.json').exists():raise RuntimeError('Refuse generation after ensemble freeze')
        generate_fresh();return
    check_frozen_source(ctx)
    provenance=[copy_existing(ctx,'results/uniform_30_manifest_v2.csv','results/ensemble_manifest.csv'),copy_existing(ctx,'results/uniform_30_pairwise_distances_v2.csv','results/ensemble_distances.csv')]
    manifest=pd.read_csv(ctx.results/'ensemble_manifest.csv').rename(columns={'min_distance_to_final_other_29':'min_distance_to_final_others'})
    manifest.to_csv(ctx.results/'ensemble_manifest.csv',index=False)
    atomic_json(ctx.root/'methods/ensemble_reuse.json',dict(mode='references to frozen members; no generation',ensemble_root=str(ctx.source),files=provenance))
    print('Reused existing 30 members; no search or ODE calls')

if __name__=='__main__':main()




