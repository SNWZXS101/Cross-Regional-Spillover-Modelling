"""O2 UDE training on unsmoothed daily counts using the original initial conditions.

Run from PyCharm or command line; output is a unique directory under T3 only.
No previous checkpoint is loaded. Use --smoke solely for isolation checks.
"""
from pathlib import Path
import os,sys,json,hashlib,uuid,datetime,dataclasses,argparse,platform
HERE=Path(__file__).resolve().parent;GROUP=HERE.parent;PROJECT=HERE.parents[3]
CACHE=GROUP/'logs/runtime';CACHE.mkdir(parents=True,exist_ok=True)
for n in ['temp','matplotlib','numba']:(CACHE/n).mkdir(exist_ok=True)
os.environ.update(TEMP=str(CACHE/'temp'),TMP=str(CACHE/'temp'),MPLCONFIGDIR=str(CACHE/'matplotlib'),NUMBA_CACHE_DIR=str(CACHE/'numba'),PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import ude_training as model
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 orig=PROJECT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1/config.json';data=PROJECT/'Data/input/Data_all.xlsx';source=json.loads(orig.read_text(encoding='utf-8'))
 out=GROUP/'runs'/('smoke' if args.smoke else 'baseline_training')/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
 if not out.resolve().is_relative_to((PROJECT/'Revise_work/T4_observation_data').resolve()):raise RuntimeError('Unsafe output')
 fields={f.name for f in dataclasses.fields(model.Config)};kw={k:v for k,v in source.items() if k in fields};selected=source['autotune_candidates'][2];kw.update(seed=31415,autotune_candidates=(tuple(selected),),data_file=str(data),output_root=str(out),run_autotune=True)
 if args.smoke:kw.update(screen_adam_iterations=1,adam_iterations=1,use_lbfgs=False)
 model.CONFIG=model.Config(**kw)
 meta={'experiment':'O2','seed':31415,'selected_candidate':3,'smoke_only':args.smoke,'output':str(out),'data':str(data),'data_sha256':sha(data),'baseline_config_sha256':sha(orig),'copied_trainer_sha256':sha(HERE/'ude_training.py'),'wrapper_sha256':sha(Path(__file__)),'python':platform.python_version(),'torch':str(model.torch.__version__),'state':'running'}
 target=out/'run_metadata.json';target.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
 print('O2 training output:',out,flush=True)
 with (out/'console.log').open('w',encoding='utf-8',buffering=1) as file:
  class Tee:
   def __init__(self,a,b):self.a=a;self.b=b
   def write(self,s):self.a.write(s);self.b.write(s);self.b.flush()
   def flush(self):self.a.flush();self.b.flush()
  oldout,olderr=sys.stdout,sys.stderr;sys.stdout,sys.stderr=Tee(oldout,file),Tee(olderr,file)
  try:model.main()
  except BaseException as exc:
   meta.update(state='failed',error=repr(exc));target.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8');raise
  finally:sys.stdout,sys.stderr=oldout,olderr
 meta['state']='training_finished_independent_validation_required';target.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
 print('O2 training complete; independent validation required.',flush=True)
if __name__=='__main__':main()

