import json, numpy as np
from pathlib import Path
from scripts.core.fit_diagnostics import diagnostics
_CONFIG=Path(__file__).resolve().parents[4]/'configs/admissibility_prepolicy.json'
_SPEC=json.loads(_CONFIG.read_text(encoding='utf-8-sig'))
_B=_SPEC['baseline_gate'];_C=_SPEC['candidate_gate']
if not _SPEC['locked_before_any_policy_computation'] or _C['weekly_WAPE_per_region_max']!='min(0.20, 1.05 * baseline_weekly_WAPE)' or _C['absolute_cumulative_relative_error_max']!='min(0.10, max(0.01, 1.05 * baseline_absolute_relative_error))':raise RuntimeError('Unsupported or unlocked O2 gate configuration')

def candidate_gates(obs,pred,cfg,baseline):
 scale=max(float(obs[:,0].sum()),float(obs[:,1].sum()),1.)
 ds=[diagnostics(obs[:,j],pred[:,j],cfg,scale) for j in range(2)];vals=[]
 for j,d in enumerate(ds):
  b=baseline['raw_data_diagnostics']['DOP853_'+['GD','GX'][j]]
  vals += [_C['observation_loss_per_region_max_baseline_multiplier']-d['observation_loss']/b['observation_loss'],_B['weekly_WAPE_max_each_region']-d['weekly_WAPE'],min(.20,1.05*b['weekly_WAPE'])-d['weekly_WAPE'],min(.10,max(.01,1.05*abs(b['cumulative_relative_difference'])))-abs(d['cumulative_relative_difference']),d['daily_R2']-_C['daily_R2_min_each_region']]
 v=np.array(vals)
 if not np.isfinite(v).all():raise ValueError('nonfinite observation gate')
 return v,ds


