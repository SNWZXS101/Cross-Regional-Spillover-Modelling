"""File contracts and validation shared by stages; never imports ODE solvers."""
from pathlib import Path
import os,json,hashlib,tempfile,shutil
import numpy as np
import pandas as pd

STRATEGIES=['NN','HN','LN','NH','NL','HH','HL','LH','LL']
STAGES=['audit_baseline','generate_uniform_ensemble','freeze_ensemble','run_strategy_matrix','analyze_pairwise_rankings','summarize_results','make_figures','build_report','verify_integrity']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(temp,path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ids(n):return [f'U{i:02d}' for i in range(1,n+1)]

class Context:
    def __init__(self,path=None):
        self.path=Path(path or os.environ['UDE_RUN_CONTEXT']).resolve();self.config=read_json(self.path)
        self.project=Path(self.config['project_root']);self.root=Path(self.config['output_dir']);self.n=int(self.config['n_solutions']);self.mode=self.config['mode']
        self.source=self.project/self.config['source_analysis'];self.base=self.project/self.config['baseline_dir'];self.protocol=self.project/self.config['risk_protocol']
        work=(self.project/'Revise_work').resolve()
        if not self.path.is_relative_to(work) or not self.root.resolve().is_relative_to(work):raise RuntimeError('Refusing context or output outside Revise_work')
        if not self.base.resolve().is_relative_to(work) and not self.config.get('nonmember_pilot',False):raise RuntimeError('Formal T2 baseline must reside in Revise_work')
        if self.mode!='fresh':raise RuntimeError('T2 copy refuses reuse_frozen mode')
        self.results=self.root/'results';self.figures=self.root/'figures'
    @property
    def ensemble_root(self):return self.source if self.mode=='reuse_frozen' else self.root
    @property
    def trajectory_root(self):return self.source/'results/matrices' if self.mode=='reuse_frozen' else self.results/'scenarios'

def copy_existing(ctx,source_name,target_name):
    src=ctx.source/source_name;dest=ctx.root/target_name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    return {'source':str(src),'source_sha256':sha(src),'target':str(dest.relative_to(ctx.root))}

def check_frozen_source(ctx):
    frozen=read_json(ctx.source/'methods/frozen_ensemble_sha256.json')
    for p,h in frozen.items():
        if sha(ctx.source/p)!=h:raise RuntimeError('Frozen v2 artifact changed: '+p)
    m=pd.read_csv(ctx.source/'results/uniform_30_manifest_v2.csv')
    if ctx.n!=len(m):raise ValueError('reuse_frozen must use the original ensemble size')
    if m.solution.tolist()!=ids(ctx.n):raise ValueError('Unexpected frozen member order')
    return frozen

def validate_scenarios(df,n):
    need={'solution','strategy','measure','C1','C2','A1','A2','Atotal'}
    if not need<=set(df):raise ValueError('Missing scenario columns: '+str(need-set(df)))
    if len(df)!=n*9*2 or set(df.solution)!=set(ids(n)):raise ValueError('Incorrect ensemble or row count')
    if df.duplicated(['solution','strategy','measure']).any():raise ValueError('Duplicate scenario rows')
    if not np.isfinite(df[['C1','C2','A1','A2','Atotal']].to_numpy()).all():raise ValueError('Nonfinite scenario values')
    if (df[['C1','C2']]<0).any().any():raise ValueError('Negative cumulative cases')
    for (name,measure),d in df.groupby(['solution','measure']):
        if len(d)!=9 or set(d.strategy)!=set(STRATEGIES) or measure not in ['reported','all_infections']:raise ValueError('Incomplete 3x3 block')
        nn=d[d.strategy=='NN'].iloc[0]
        if not np.allclose(d.A1,nn.C1-d.C1,rtol=0,atol=1e-7) or not np.allclose(d.A2,nn.C2-d.C2,rtol=0,atol=1e-7) or not np.allclose(d.Atotal,d.A1+d.A2,rtol=0,atol=1e-7):raise ValueError('Inconsistent avoided-case equation')
    return df

def verify_ensemble(ctx):
    f=read_json(ctx.root/'methods/ensemble_freeze.json')
    for p,h in f['files'].items():
        if sha(Path(f['base_path'])/p)!=h:raise RuntimeError('Ensemble changed since freeze: '+p)
    return f

def md_table(frame):
    def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))
