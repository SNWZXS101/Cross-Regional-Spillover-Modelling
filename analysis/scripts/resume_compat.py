"""Narrow, hash-pinned migration of the pre-adaptive runner; never relax inputs."""
import copy
from pathlib import Path
from scripts.common import atomic_json,read_json,sha
from scripts.adaptive_search import digest,round_sizes


def inspect_upgrade(project,root,cfg,state,code,inputs):
    project=Path(project);root=Path(root)
    release=read_json(project/'configs/adaptive_upgrade_v1.json')
    oldcfg=read_json(root/'config_resolved.json');source=read_json(root/'methods/source_hashes.json')
    if source['code']!=release['old_code'] or code!=release['new_code']:
        raise RuntimeError('Code changes are not the hash-pinned adaptive-budget upgrade')
    if source['inputs']!=inputs:raise RuntimeError('Inputs or runtime changed; adaptive upgrade refused')
    if state['fingerprint']!=digest(dict(config=oldcfg,code=source['code'],inputs=source['inputs'])):
        raise RuntimeError('Legacy fingerprint is inconsistent')
    expected=copy.deepcopy(oldcfg)
    if expected['mode']=='fresh':expected['adaptive_start_rounds']=round_sizes(expected)
    if expected!=cfg:raise RuntimeError('Config changes exceed the adaptive-budget upgrade')
    for stage,entry in state['stages'].items():
        if entry.get('status')=='completed':
            for rel,h in entry['outputs'].items():
                if not (root/rel).is_file() or sha(root/rel)!=h:raise RuntimeError('Completed stage changed before upgrade: '+stage+'/'+rel)
    accepted=sorted((root/'checkpoints').glob('*_accepted.json'))
    for marker in accepted:
        for rel,h in read_json(marker)['files'].items():
            if not (root/rel).is_file() or sha(root/rel)!=h:raise RuntimeError('Accepted member changed before upgrade: '+rel)
    pinned=release.get('legacy_artifacts',{})
    legacy={}
    for log in (root/'logs').glob('U*.json'):
        rel=str(log.relative_to(project))
        if rel not in pinned or sha(log)!=pinned[rel]:raise RuntimeError('Legacy search log not covered by upgrade snapshot: '+rel)
        legacy[str(log.relative_to(root))]=pinned[rel]
    # Accepted files are additionally anchored to the maintenance snapshot when present.
    for rel,h in pinned.items():
        p=project/rel
        if root in p.parents and (not p.is_file() or sha(p)!=h):raise RuntimeError('Pinned legacy artifact changed: '+rel)
    return dict(old_config=oldcfg,old_source_hashes=source,old_state=state,legacy_files=legacy,
                accepted_members=len(accepted),release_manifest_sha256=sha(project/'configs/adaptive_upgrade_v1.json'))


def migrate_upgrade(project,root,cfg,state,new_fingerprint,code,inputs):
    root=Path(root);journal=root/'methods/adaptive_upgrade.json'
    # Journal first: an interruption between metadata writes can finish on resume.
    if journal.exists():
        event=read_json(journal)
        if event['new_fingerprint']!=new_fingerprint:raise RuntimeError('Incomplete adaptive migration targets different code/config')
        backup=root/'methods/adaptive_upgrade_original.json'
        if sha(backup)!=event['original_sha256']:raise RuntimeError('Adaptive migration backup changed')
        prior=read_json(backup)
        # Temporarily writing nothing: inspect using an isolated metadata view is
        # unnecessary here; pinned artifact checks below cover the unfinished run.
        release=read_json(Path(project)/'configs/adaptive_upgrade_v1.json')
        if code!=release['new_code'] or inputs!=prior['old_source_hashes']['inputs']:raise RuntimeError('Migration inputs/code changed')
        for rel,h in prior['legacy_files'].items():
            if sha(root/rel)!=h:raise RuntimeError('Legacy log changed during migration')
        for entry in prior['old_state']['stages'].values():
            for rel,h in entry.get('outputs',{}).items():
                if Path(rel).as_posix()=='methods/source_hashes.json':continue
                if sha(root/rel)!=h:raise RuntimeError('Artifact changed during migration: '+rel)
        for marker in (root/'checkpoints').glob('*_accepted.json'):
            for rel,h in read_json(marker)['files'].items():
                if sha(root/rel)!=h:raise RuntimeError('Accepted member changed during migration')
    else:
        prior=inspect_upgrade(project,root,cfg,state,code,inputs)
        backup=root/'methods/adaptive_upgrade_original.json';atomic_json(backup,prior)
        event=dict(version=1,new_fingerprint=new_fingerprint,old_fingerprint=state['fingerprint'],
                   original_sha256=sha(backup),legacy_files=prior['legacy_files'],
                   accepted_members_preserved=prior['accepted_members'],
                   scope='Only adaptive search effort; all science, completed members and original start logs unchanged')
        atomic_json(journal,event)
    updated=copy.deepcopy(prior['old_state'])
    atomic_json(root/'config_resolved.json',cfg)
    atomic_json(root/'methods/source_hashes.json',dict(code=code,inputs=inputs))
    for entry in updated['stages'].values():
        for rel in entry.get('outputs',{}):
            if Path(rel).as_posix()=='methods/source_hashes.json':entry['outputs'][rel]=sha(root/rel)
    updated['fingerprint']=new_fingerprint
    updated['adaptive_upgrade']=dict(event_sha256=sha(journal),event=str(journal.relative_to(root)))
    atomic_json(root/'pipeline_state.json',updated)
    return updated
