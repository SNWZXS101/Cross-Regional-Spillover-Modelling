"""Hash-pinned upgrade restricted to finished analyses and verification-only execution."""
import copy
from pathlib import Path
from scripts.common import atomic_json,read_json,sha,Context,verify_ensemble
from scripts.adaptive_search import digest
from scripts.research_integrity import research_entries,is_nonresearch,require_hashes,require_completed_analysis,verify_member_commits


def migrate_verify_upgrade(project,root,cfg,state,new_fingerprint,code,inputs):
    project=Path(project);root=Path(root)
    release_path=project/'configs/verify_only_upgrade_v1.json';release=read_json(release_path)
    if code!=release['new_code']:raise RuntimeError('Not the audited verify-only code version')
    event_path=root/'methods/verify_only_upgrade.json';backup=root/'methods/verify_only_upgrade_original.json'
    if event_path.exists():
        event=read_json(event_path)
        if event['new_fingerprint']!=new_fingerprint or sha(backup)!=event['backup_sha256'] or sha(release_path)!=event['release_sha256']:
            raise RuntimeError('Verify-only migration journal mismatch')
        prior=read_json(backup)
    else:
        oldcfg=read_json(root/'config_resolved.json');source=read_json(root/'methods/source_hashes.json')
        if source['code']!=release['old_code']:raise RuntimeError('Unsupported prior code version for verify-only upgrade')
        if oldcfg!=cfg or research_entries(source['inputs'])!=inputs:raise RuntimeError('Config, scientific inputs or runtime changed; verify-only upgrade refused')
        if state['fingerprint']!=digest(dict(config=oldcfg,**source)):raise RuntimeError('Old run fingerprint inconsistent')
        require_completed_analysis(root,state)
        require_hashes(project,read_json(root/'methods/protected_before.json'),'Protected research changed before verify-only upgrade')
        verify_ensemble(Context(root/'config_resolved.json'))
        if cfg['mode']=='fresh':verify_member_commits(root,cfg['n_solutions'])
        preserved={}
        for p in root.rglob('*'):
            if not p.is_file():continue
            rel=str(p.relative_to(root));norm=rel.replace('\\','/')
            if is_nonresearch(rel) or norm in ['pipeline_state.json','methods/source_hashes.json','results/integrity_check.json','logs/verify_integrity.log']:continue
            preserved[rel]=sha(p)
        # Legacy protected_before excluded maintained source/config directories.
        # Add their strict hashes rather than widening that old exclusion policy.
        static={str(p.relative_to(project)):sha(p) for p in (project/'configs').rglob('*') if p.is_file() and not is_nonresearch(p.relative_to(project))}
        static.update(code)
        prior=dict(config=oldcfg,source=source,state=state,preserved_files=preserved,static_research_files=static)
        atomic_json(backup,prior)
        event=dict(version=1,old_fingerprint=state['fingerprint'],new_fingerprint=new_fingerprint,
                   backup_sha256=sha(backup),release_sha256=sha(release_path),
                   scope='Only audited integrity filtering and verify-only orchestration; all eight earlier stages remain completed',
                   scientific_recalculation=False,static_research_files=static)
        atomic_json(event_path,event)
    if prior['config']!=cfg or research_entries(prior['source']['inputs'])!=inputs:raise RuntimeError('Config/inputs changed during verify-only migration')
    require_hashes(root,prior['preserved_files'],'Previously completed artifact changed during verify-only migration')
    require_hashes(project,prior['static_research_files'],'Source/config changed during verify-only migration')
    require_completed_analysis(root,prior['state'],skip_source_metadata=True)
    if cfg['mode']=='fresh':verify_member_commits(root,cfg['n_solutions'])
    atomic_json(root/'methods/source_hashes.json',dict(code=code,inputs=inputs))
    updated=copy.deepcopy(prior['state'])
    for entry in updated['stages'].values():
        for rel in entry.get('outputs',{}):
            if rel.replace('\\','/')=='methods/source_hashes.json':entry['outputs'][rel]=sha(root/rel)
    updated['fingerprint']=new_fingerprint
    updated['verify_only_upgrade']=dict(event=str(event_path.relative_to(root)),event_sha256=sha(event_path))
    atomic_json(root/'pipeline_state.json',updated)
    return updated
