"""Separate mandatory freeze barrier before any policy integration."""
from scripts.common import *
import datetime
from scripts.core.model import CFG,PROJECT
from scripts.core.o2_gates import candidate_gates

def main():
    ctx=Context();path=ctx.root/'methods/ensemble_freeze.json'
    if path.exists():verify_ensemble(ctx);print('Existing freeze verified');return
    m=pd.read_csv(ctx.results/'ensemble_manifest.csv')
    if m.solution.tolist()!=ids(ctx.n) or not m.accepted.all():raise ValueError('Incomplete admissible ensemble')
    baseline=read_json(Path(read_json(ctx.results/'baseline_audit.json')['source']))
    obs=pd.read_excel(PROJECT/CFG['data_file'])[[CFG['gd_column'],CFG['gx_column']]].values
    for name in ids(ctx.n):
        with np.load(ctx.root/'solutions'/(name+'.npz')) as z:
            for data in [z['states'][::40],z['DOP853_daily_states']]:
                y=np.diff(data[:,10:12],axis=0);gates,_=candidate_gates(obs,y,CFG,baseline)
                if min(gates)<0:raise ValueError('O2 observation gate failed '+name)
    if ctx.mode=='reuse_frozen':files=check_frozen_source(ctx);base=ctx.source
    else:
        base=ctx.root;files={}
        for name in ids(ctx.n):
            marker=read_json(ctx.root/'checkpoints'/(name+'_accepted.json'))
            for rel,h in marker['files'].items():
                if sha(ctx.root/rel)!=h:raise ValueError('Incomplete member transaction: '+name)
                files[rel]=h
        files['results/ensemble_manifest.csv']=sha(ctx.results/'ensemble_manifest.csv')
        files['results/ensemble_distances.csv']=sha(ctx.results/'ensemble_distances.csv')
    atomic_json(path,dict(base_path=str(base),member_ids=ids(ctx.n),n_solutions=ctx.n,mode=ctx.mode,files=files,frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy_independent=True,source_code_hashes=read_json(ctx.root/'methods/source_hashes.json')['code']))
    print('Ensemble frozen:',ctx.n,'members')

if __name__=='__main__':main()


