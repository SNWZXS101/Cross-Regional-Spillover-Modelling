from pathlib import Path
import numpy as np,pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[4];g=Path(__file__).resolve().parent.parent
y=pd.read_excel(p/'Data/input/Data_all.xlsx')[['GD','GX']].to_numpy(float)
a=np.load(p/'analysis_runs/sensitivity_100/results/baseline_rollout.npz');pred=np.diff(a['DOP853'][:,10:12],axis=0)
fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
for j,(ax,n,c) in enumerate(zip(axes,['GD','GX'],['#235a85','#168374'])):
 ax.bar(np.arange(236),y[:,j],color='#b6c4c5',width=1,label='Raw daily observations');ax.plot(pred[:,j],color=c,lw=1.2,label='Original baseline independent ODE');ax.set_ylabel(n+' cases/day');ax.legend(frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
axes[-1].set_xlabel('Day');fig.savefig(g/'figures/original_baseline_vs_raw.png',dpi=220);plt.close(fig)
