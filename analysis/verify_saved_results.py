"""Recompute key paper statistics from saved results; no ODE integration."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
order=['NN','HN','LN','NH','NL','HH','HL','LH','LL']
summary={};ranking=[]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,help='new output directory for summary and complete pairwise counts');args=ap.parse_args()
for run,n in [('main_30',30),('sensitivity_50',50),('sensitivity_100',100)]:
 d=ROOT/'results'/run
 s=pd.read_csv(d/'complete_scenarios.csv');m=pd.read_csv(d/'ensemble_manifest.csv');dist=pd.read_csv(d/'ensemble_distances.csv',index_col=0).to_numpy(dtype=float)
 if len(m)!=n or len(s)!=n*18 or len(set(s.solution))!=n:raise ValueError(f'{run}: unexpected solution or scenario count')
 if s.duplicated(['solution','strategy','measure']).any():raise ValueError('duplicate scenario')
 if dist.shape!=(n,n) or not np.allclose(dist,dist.T,atol=1e-12) or not np.allclose(np.diag(dist),0,atol=1e-12):raise ValueError(f'{run}: invalid distance matrix')
 nn=s[s.strategy=='NN'].set_index(['solution','measure'])
 for x in s.itertuples(index=False):
  ref=nn.loc[(x.solution,x.measure)]
  if not np.allclose([x.A1,x.A2,x.Atotal],[ref.C1-x.C1,ref.C2-x.C2,ref.C1+ref.C2-x.C1-x.C2],atol=1e-6,rtol=1e-10):raise ValueError('avoided effect mismatch')
 groups={}
 for measure in ('reported','all_infections'):
  z=s[s.measure==measure].copy();z['total']=z.C1+z.C2
  p=z.pivot(index='solution',columns='strategy',values='total')[order]
  for i,a in enumerate(order):
   for b in order[i+1:]:
    left=int((p[a]<p[b]).sum());right=int((p[b]<p[a]).sum());ties=n-left-right
    ranking.append(dict(run=run,measure=measure,strategy_a=a,strategy_b=b,a_lower=left,b_lower=right,ties=ties,n=n))
  unanimous=sum(int(((p[a]<p[b]).all() or (p[b]<p[a]).all())) for i,a in enumerate(order) for b in order[i+1:])
  hhll=int((p.HH<p.LL).sum());hhvsnn=int((p.HH<p.NN).sum());llharm=int((p.LL>p.NN).sum())
  q=(p.LL-p.HH);groups[measure]={'unanimous_pairs':unanimous,'HH_less_LL':hhll,'HH_less_NN':hhvsnn,'LL_more_NN':llharm,'LL_minus_HH_min_median_max':[float(q.min()),float(q.median()),float(q.max())]}
 srep=s[s.measure=='reported'];hn=srep[srep.strategy=='HN'];nh=srep[srep.strategy=='NH']
 pairs=dist[np.triu_indices(n,1)]
 summary[run]={'n':n,'solution_pairs':len(pairs),'distance_min_median_max':[float(np.min(pairs)),float(np.median(pairs)),float(np.max(pairs))],'HN_GD_harm':int((hn.A1<0).sum()),'NH_GX_benefit':int((nh.A2>0).sum()),'outcomes':groups}
 if groups['reported']['unanimous_pairs']!={30:6,50:6,100:4}[n] or groups['reported']['HH_less_LL']!=n:raise ValueError('Paper ranking counts do not match')
 saved=pd.read_csv(d/'pairwise_rankings.csv')
 for row in [x for x in ranking if x['run']==run and x['measure']=='reported']:
  match=saved[(saved.measure=='total_reported')&(saved.strategy_A==row['strategy_a'])&(saved.strategy_B==row['strategy_b'])]
  if len(match)!=1 or int(match.iloc[0].A_fewer_count)!=row['a_lower'] or int(match.iloc[0].B_fewer_count)!=row['b_lower'] or int(match.iloc[0].equal_count)!=row['ties']:raise ValueError('saved pairwise ranking mismatch')
print(json.dumps(summary,ensure_ascii=False,indent=2))

if args.output:
 target=args.output.resolve()
 if (ROOT/'results').resolve() in target.parents or target==(ROOT/'results').resolve():raise SystemExit('Refuse writing into archived results')
 target.mkdir(parents=True,exist_ok=True)
 (target/'recomputed_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
 pd.DataFrame(ranking).to_csv(target/'pairwise_rankings.csv',index=False)