"""Maintained UDE analysis pipeline. Archived analyses remain read-only."""
import os,sys,importlib.abc

class _NoODEImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='scripts.core' or fullname.startswith('scripts.core.') or fullname=='scipy.integrate' or fullname.startswith('scipy.integrate.'):
            raise RuntimeError('ODE imports are forbidden in reuse_frozen/downstream mode: '+fullname)
        return None

if os.environ.get('UDE_FORBID_ODE')=='1':sys.meta_path.insert(0,_NoODEImports())
