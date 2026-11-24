"""O2 per-region observational data loss matching the training objective."""
import numpy as np

def robust(x):
 x=np.asarray(x,float);return max(float(np.std(x)),.05*float(np.max(np.abs(x))),1e-6)

def diagnostics(observed,predicted,cfg,scale):
 y=np.asarray(observed,float);p=np.asarray(predicted,float)
 if y.shape!=(236,) or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y<0) or np.any(p<0) or np.sum(y)<=0 or np.sum((y-y.mean())**2)<=0:raise ValueError('invalid data, degenerate observation or negative predicted daily counts')
 cy=np.cumsum(y)/scale;cp=np.cumsum(p)/scale
 raw=np.mean(((p/scale-y/scale)/robust(y/scale))**2)
 sq=np.mean(((np.sqrt(p+1)-np.sqrt(y+1))/robust(np.sqrt(y+1)))**2)
 lg=np.mean(((np.log1p(p)-np.log1p(y))/robust(np.log1p(y)))**2)
 cum=np.mean(((cp-cy)/robust(cy))**2)
 loss=cfg['w_daily_raw']*raw+cfg['w_daily_sqrt']*sq+cfg['w_daily_log']*lg+cfg['w_cumulative']*cum
 wr=np.array([np.sum(p[i:i+7])-np.sum(y[i:i+7]) for i in range(0,236,7)])
 ss=np.sum((y-y.mean())**2)
 return dict(observation_loss=float(loss),daily_R2=float(1-np.sum((p-y)**2)/ss),daily_RMSE=float(np.sqrt(np.mean((p-y)**2))),weekly_WAPE=float(np.sum(np.abs(wr))/np.sum(y)),cumulative_relative_difference=float(np.sum(p-y)/np.sum(y)), weekly_last_days=5)

