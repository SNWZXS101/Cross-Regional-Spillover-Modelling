"""O1: restart L-BFGS at saved 27182 network weights, without Adam or policy data."""
from pathlib import Path
import os,sys,json,hashlib,datetime,uuid,dataclasses,time
HERE=Path(__file__).resolve().parent;G=HERE.parent;PROJECT=HERE.parents[3];cache=G/'logs/runtime'
os.environ.update(TEMP=str(cache/'temp'),TMP=str(cache/'temp'),MPLCONFIGDIR=str(cache/'matplotlib'),NUMBA_CACHE_DIR=str(cache/'numba'),PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
import ude_training as m
import torch,numpy as np
from copy import deepcopy

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 prev=G/'runs/baseline_retry2/20261120_111217_e83c02ff/Training-results-26-11-20_112342-set1';old=G/'runs/baseline_retry2/20261120_111217_e83c02ff/validation/baseline_independent_validation.json'
 if json.loads(old.read_text())['accepted']:raise RuntimeError('Continuation only authorized for failed baseline')
 ckpt=prev/'parameter_net_checkpoint.pt';saved=torch.load(ckpt,map_location='cpu',weights_only=False);fields={f.name for f in dataclasses.fields(m.Config)};kw={k:v for k,v in saved['config'].items() if k in fields};kw.update(data_file=str(PROJECT/'Data/input/Data_all_sm.xlsx'),lbfgs_max_iter=150)
 registration=json.loads((G/'configs/continuation_150_preregister.json').read_text(encoding='utf-8'))
 for name,path in [('checkpoint_sha256',ckpt),('previous_validation_sha256',old),('trainer_sha256',HERE/'ude_training.py'),('continuation_sha256',Path(__file__)),('data_sha256',Path(kw['data_file']))]:
  if sha(path)!=registration[name]:raise RuntimeError('Preregistered source changed: '+name)
 out=G/'runs/baseline_continuation'/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False);cfg=m.Config(**kw);cfg.output_root=str(out);data=m.prepare_data(cfg)
 if cfg.seed!=27182 or data['count_scale']!=saved['count_scale']:raise RuntimeError('Checkpoint/data mismatch')
 metadata={'experiment':'O1 continuation','seed':27182,'state':'running','smoke_only':False,'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),'previous_validation_sha256':sha(old),'data_sha256':sha(Path(cfg.data_file)),'baseline_config_sha256':sha(PROJECT/'baseline/Model4GDGXcq2x_v6_optimized01/Training-results-26-09-12_003513-set1/config.json'),'copied_trainer_sha256':sha(HERE/'ude_training.py'),'wrapper_sha256':sha(Path(__file__)),'lbfgs_restart':True,'optimizer_history_restored':False,'max_iterations':150,'max_evaluations':187,'policy_blind':True}
 (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
 pnet=m.ParameterNet(cfg,float(data['n_days']),data['dtype']).to(device=data['device'],dtype=data['dtype']);pnet.load_state_dict(saved['parameter_net_state_dict']);solver=m.DifferentiableTwoRegionODE(cfg,pnet,data['N1_scaled'],data['N2_scaled'],data['x0_scaled'],data['dtype'],data['device']);loss=m.ODEFitLoss(cfg,solver,data)
 opt=torch.optim.LBFGS(pnet.parameters(),lr=cfg.lbfgs_lr,max_iter=150,max_eval=187,history_size=cfg.lbfgs_history_size,tolerance_grad=cfg.lbfgs_tolerance_grad,tolerance_change=cfg.lbfgs_tolerance_change,line_search_fn='strong_wolfe'); count=0;history=[];t0=time.time()
 def closure():
  nonlocal count
  opt.zero_grad(set_to_none=True);total,parts,_,_,_=loss.compute(data['n_days'],cfg.ode_substeps_per_day)
  if not torch.isfinite(total):raise RuntimeError('Nonfinite loss')
  total.backward();count+=1;row={'evaluation':count,'total':float(total.detach().cpu())};history.append(row)
  if count==1 or count%25==0:print('LBFGS evaluation',count,'loss',row['total'],flush=True)
  return total
 opt.step(closure)
 with torch.no_grad():total,parts,states,daily,rates=loss.compute(data['n_days'],cfg.ode_substeps_per_day)
 result={'pnet':pnet,'pnet_state_dict':deepcopy(pnet.state_dict()),'states_scaled':states.detach().cpu().numpy(),'daily_scaled':daily.detach().cpu().numpy(),'rates_daily':rates.detach().cpu().numpy(),'beta':float(pnet.beta().detach().cpu()),'training_seconds':time.time()-t0,'final_loss':float(total.detach().cpu()),'loss_parts':{k:float(v.detach().cpu()) for k,v in parts.items()},'substeps':cfg.ode_substeps_per_day}
 dest=out/('Training-results-'+datetime.datetime.now().strftime('%y-%m-%d_%H%M%S')+'-set1');dest.mkdir();val=m.evaluate_candidate(cfg,data,result);cfgdict=dataclasses.asdict(cfg);cfgdict['COUNT_SCALE_used']=data['count_scale'];(dest/'config.json').write_text(json.dumps(cfgdict,indent=2),encoding='utf-8');torch.save({'parameter_net_state_dict':result['pnet_state_dict'],'config':cfgdict,'count_scale':data['count_scale']},dest/'parameter_net_checkpoint.pt');m.save_outputs(cfg,data,result,val,dest);(out/'lbfgs_history.json').write_text(json.dumps(history,indent=2));(out/'optimizer_status.json').write_text(json.dumps({'evaluations':count,'n_iter':int(next(iter(opt.state.values())).get('n_iter',0)),'func_evals':int(next(iter(opt.state.values())).get('func_evals',0)),'max_iter':150,'max_eval':187,'termination_reason':'not exposed by torch LBFGS','final_loss':result['final_loss']},indent=2));metadata['state']='training_finished_independent_validation_required';(out/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print('outputs',dest,flush=True)
if __name__=='__main__':main()
