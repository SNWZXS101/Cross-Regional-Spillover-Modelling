"""Deterministic budget orchestration and atomic start caches; no numerical model."""
import hashlib
import json
import math
from pathlib import Path
from scripts.common import atomic_json, read_json, sha


def clean(value):
    """Serialize nonfinite diagnostics explicitly, never modify numerical candidates."""
    if isinstance(value, dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [clean(v) for v in value]
    if hasattr(value, 'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def round_sizes(config):
    n=config['n_starts'];sizes=config.get('adaptive_start_rounds',[n,n,2*n])
    if not isinstance(sizes,list) or len(sizes)!=3 or any(type(v) is not int or v<1 for v in sizes) or sizes[0]!=n:
        raise ValueError('adaptive_start_rounds requires three positive counts; first must equal n_starts')
    return sizes


class StartCache:
    def __init__(self,root,k,context,legacy=None):
        self.root=Path(root);self.k=k;self.context=context;self.legacy=legacy or {}

    def paths(self,trial):
        stem=f'U{self.k:02d}_start{trial}'
        return self.root/'logs'/(stem+'.json'),self.root/'checkpoints'/(stem+'_completed.json')

    def load(self,trial):
        log,marker=self.paths(trial)
        if marker.exists():
            m=read_json(marker)
            if m.get('context')!=self.context or m.get('status')!='completed' or not log.is_file() or sha(log)!=m['log_sha256']:
                raise RuntimeError('Start checkpoint checksum/context mismatch: '+str(log))
            return read_json(log)
        if not log.exists():return None
        info=read_json(log);rel=str(log.relative_to(self.root))
        legacy_ok=rel in self.legacy and sha(log)==self.legacy[rel]
        unsigned={key:v for key,v in info.items() if key!='_payload_sha256'}
        current_ok=info.get('_cache_context')==self.context and info.get('_payload_sha256')==digest(unsigned)
        if not (legacy_ok or current_ok):raise RuntimeError('Unverified existing start log: '+str(log))
        self.validate(info,trial)
        self.commit(trial,legacy=legacy_ok)
        return info

    def validate(self,info,trial):
        required=['k','trial','seed','proposed_initial_coordinates','actual_initial_coordinates','stop_reason','optimizer_success','iterations','nfev','feasible_incumbent','progress']
        if any(x not in info for x in required) or info['k']!=self.k or info['trial']!=trial or info['seed']!=self.context['seed']:
            raise RuntimeError('Incomplete or incompatible start log')
        for key in ['proposed_initial_coordinates','actual_initial_coordinates']+(['best_coordinates'] if info['feasible_incumbent'] else []):
            if len(info[key])!=30 or not all(math.isfinite(x) for x in info[key]):raise RuntimeError('Invalid start coordinates')
        if not 0<=info['iterations']<=self.context['max_iterations'] or info['nfev']<1:raise RuntimeError('Invalid start accounting')

    def commit(self,trial,legacy=False):
        log,marker=self.paths(trial)
        atomic_json(marker,dict(status='completed',context=self.context,log_sha256=sha(log),legacy_adopted=legacy))

    def save(self,trial,info):
        self.validate(info,trial)
        log,marker=self.paths(trial)
        if marker.exists():raise RuntimeError('Refuse to overwrite a completed start')
        info=clean(dict(info,_cache_context=self.context))
        info['_payload_sha256']=digest(info)
        atomic_json(log,info);self.commit(trial)


def run_adaptive(k,sizes,fetch,assess,record):
    """Finish one whole round, select by unchanged maximin, expand only on failure.

    fetch reuses or computes starts in stable trial order. assess may reuse a
    hash-verified failed round. No policy information enters this controller.
    """
    used=[];offset=0
    for number,count in enumerate(sizes,1):
        trials=list(range(offset,offset+count));offset+=count
        proposals=fetch(number,trials);used.extend(proposals)
        candidate=assess(number,proposals)
        stats=dict(search_rounds_used=number,search_starts_used=len(used),
                   total_search_evaluations=sum(info['nfev'] for _,info in used),
                   adaptive_start_rounds=list(sizes),max_iterations_per_start=used[0][1].get('_max_iterations'))
        record(dict(stats,status='accepted' if candidate is not None else 'round_failed',last_round=number))
        if candidate is not None:return candidate,stats
    record(dict(stats,status='exhausted',last_round=len(sizes)))
    raise RuntimeError(f'No candidate passed the unchanged uniform fine-ODE standard at k={k} after {len(sizes)} rounds / {len(used)} starts; completed members and start checkpoints retained. No substitute old solution accepted.')
