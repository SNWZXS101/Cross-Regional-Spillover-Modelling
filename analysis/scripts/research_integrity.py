"""Shared research-file policy. Only explicitly named IDE/runtime artifacts escape hashing."""
from pathlib import Path,PurePosixPath
from scripts.common import sha,read_json,STAGES,ids

NONRESEARCH_DIRECTORIES=frozenset({'.idea','.vscode','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'})
NONRESEARCH_FILENAMES=frozenset({'.ds_store','thumbs.db','.pipeline.lock'})
NONRESEARCH_SUFFIXES=frozenset({'.pyc','.pyo'})


def is_nonresearch(path):
    parts=PurePosixPath(str(path).replace('\\','/')).parts
    lower=[part.casefold() for part in parts]
    return bool(lower and (any(part in NONRESEARCH_DIRECTORIES for part in lower) or
                           lower[-1] in NONRESEARCH_FILENAMES or
                           PurePosixPath(lower[-1]).suffix in NONRESEARCH_SUFFIXES))


def research_entries(entries):
    # Applies to historical manifests too; never rewrite them just to drop IDE files.
    return {rel:h for rel,h in entries.items() if not is_nonresearch(rel)}


def changed_files(base,entries):
    base=Path(base)
    return [rel for rel,h in research_entries(entries).items()
            if not (base/rel).is_file() or sha(base/rel)!=h]


def require_hashes(base,entries,label):
    changed=changed_files(base,entries)
    if changed:raise RuntimeError(label+': '+str(changed[:20]))


def research_snapshot(project,output_root=None):
    project=Path(project).resolve();active=Path(output_root).resolve() if output_root else None
    files={}
    for p in project.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(project)
        if is_nonresearch(rel) or rel.parts[0]=='maintenance' or active is not None and (p==active or active in p.parents):continue
        files[str(rel)]=sha(p)
    # Current output is covered by stage/member/freeze commits, rather than by a
    # pre-run snapshot of files which this run is itself about to create.
    return files


def require_completed_analysis(root,state,skip_source_metadata=False):
    root=Path(root)
    for stage in STAGES[:-1]:
        entry=state.get('stages',{}).get(stage,{})
        if entry.get('status')!='completed' or not entry.get('outputs'):
            raise RuntimeError('verify-only requires every earlier stage completed: '+stage)
        outputs=entry['outputs']
        if skip_source_metadata:outputs={p:h for p,h in outputs.items() if str(p).replace('\\','/')!='methods/source_hashes.json'}
        require_hashes(root,outputs,'Completed stage changed: '+stage)
    if 'FINAL_REPORT.md' not in state['stages']['build_report']['outputs']:
        raise RuntimeError('Final report is missing from its committed stage')


def verify_member_commits(root,n):
    root=Path(root)
    for suffix in ['accepted','strategies']:
        expected={name+'_'+suffix+'.json' for name in ids(n)}
        actual={p.name for p in (root/'checkpoints').glob('*_'+suffix+'.json')}
        if expected!=actual:raise RuntimeError('Unexpected '+suffix+' member set')
        for name in sorted(expected):require_hashes(root,read_json(root/'checkpoints'/name)['files'],name)
