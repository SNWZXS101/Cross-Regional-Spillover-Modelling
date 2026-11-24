"""All maintained figures; reads CSV/NPZ only, never calls an ODE solver."""
from scripts.common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

def save(ctx,fig,name):
    fig.tight_layout(rect=(0,0,1,.92) if fig._suptitle is not None else (0,0,1,1));fig.savefig(ctx.figures/name,dpi=180,bbox_inches='tight');plt.close(fig)

def ranking_figures(ctx):
    measures=[('total_reported','Total reported'),('gd_reported','GD reported'),('gx_reported','GX reported')]
    fig,axes=plt.subplots(1,3,figsize=(18,7))
    for ax,(name,title) in zip(axes,measures):
        mat=pd.read_csv(ctx.results/f'pairwise_ranking_{name}.csv',index_col=0).loc[STRATEGIES,STRATEGIES].values
        im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1)
        for i in range(9):
            for j in range(9):ax.text(j,i,f'{round(mat[i,j]*ctx.n)}/{ctx.n}',ha='center',va='center',fontsize=7.5,color='white' if mat[i,j]>.55 else 'black')
        ax.set_xticks(range(9),STRATEGIES,rotation=45);ax.set_yticks(range(9),STRATEGIES);ax.set_xlabel('Column strategy B');ax.set_ylabel('Row strategy A');ax.set_title(title)
    fig.suptitle('Row A has fewer cases than column B: count / ensemble size\nEnsemble fraction, not probability; strict comparison; diagonal = 0',fontsize=12)
    save(ctx,fig,'pairwise_ranking_stability.png')
    effects=pd.read_csv(ctx.results/'policy_effect_sizes.csv')
    titles={'G_HH_minus_LL':'G(HH - LL): total HH advantage over LL','A_total_HH':'A_total(HH): total HH benefit vs NN','A_GD_HN':'A_GD(HN): GD controller benefit vs NN','A_GX_NH':'A_GX(NH): GX controller benefit vs NN'}
    for measure in ['reported','all_infections']:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(name,title) in zip(axes.flat,titles.items()):
            v=np.sort(effects[(effects.measure==measure)&(effects.functional==name)].effect_cases.to_numpy())
            ax.step(np.r_[v[0],v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color='tab:blue')
            ax.scatter(v,np.arange(1,len(v)+1)/len(v),s=12,color='tab:blue');ax.axvline(0,color='black',ls='--',lw=1);ax.axvline(np.median(v),color='tab:orange',ls=':',label=f'Median = {np.median(v):,.1f}')
            ax.set_xscale('symlog',linthresh=10);ax.set_xlim(min(-10.,float(v.min())*1.1),max(10.,float(v.max())*1.1));ax.set_ylim(-.02,1.04);ax.grid(alpha=.2)
            ax.set_title(title+'\n'+f'Positive {int((v>0).sum())}/{len(v)}; negative {int((v<0).sum())}/{len(v)}',fontsize=10)
            ax.set_xlabel('Effect in cases (symmetric-log axis; linear within +/-10)');ax.set_ylabel('Fraction of ensemble at or below effect');ax.legend(fontsize=8,loc='lower right')
        fig.suptitle(measure+': empirical effect-size distributions (no probability interpretation)',fontsize=12)
        save(ctx,fig,'policy_effect_ecdf_'+measure+'.png')

def primary_figures(ctx):
    cfg=read_json(ctx.base/'config.json');names=['c1','c2','q1','q2','m1','m2','d1','d2'];a=np.load(ctx.results/'function_arrays.npz');t=a['time'];colors=plt.cm.viridis(np.linspace(0,1,ctx.n))
    fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True)
    for j,ax in enumerate(axes.flat):
        for i,r in enumerate(a['rates']):ax.plot(t,r[:,j],c=colors[i],alpha=.65,lw=.6)
        ax.set_ylim(*cfg[names[j]+'_bounds']);ax.set_ylabel(names[j]);ax.grid(alpha=.15)
    for ax in axes[-1]:ax.set_xlabel('Day')
    fig.suptitle(f'{ctx.n} diverse admissible solutions: parameter functions');save(ctx,fig,'parameter_functions.png')
    dist=pd.read_csv(ctx.results/'ensemble_distances.csv',index_col=0).values;v=dist[np.triu_indices(ctx.n,1)]
    fig,axes=plt.subplots(1,2,figsize=(12,5));im=axes[0].imshow(dist,cmap='viridis');fig.colorbar(im,ax=axes[0]);axes[0].set_title('Normalized function distances');axes[1].hist(v,bins=20);axes[1].set_title(f'{len(v)} unordered pairs');axes[1].set_xlabel('Distance');axes[1].set_ylabel('Pair count');save(ctx,fig,'ensemble_distances.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for j,ax in enumerate(axes):
        for i,f in enumerate(a['fluxes']):ax.plot(t,f[:,j],c=colors[i],alpha=.65,lw=.6)
        ax.set_title(['GD to GX','GX to GD'][j]);ax.set_xlabel('Day');ax.set_ylabel('Infectious spillover / day')
    save(ctx,fig,'spillover_functions.png')
    df=pd.read_csv(ctx.results/'complete_scenarios.csv');fig,axes=plt.subplots(2,3,figsize=(13,8))
    for i,measure in enumerate(['reported','all_infections']):
        d=df[df.measure==measure]
        for j,col in enumerate(['A1','A2','Atotal']):
            med=d.groupby('strategy')[col].median();ar=np.array([[med[x+y]/1000 for y in 'NHL'] for x in 'NHL']);lim=max(abs(ar).max(),1);ax=axes[i,j]
            ax.imshow(ar,cmap='RdBu',norm=SymLogNorm(linthresh=.2,vmin=-lim,vmax=lim))
            for x in range(3):
                for y in range(3):ax.text(y,x,f'{ar[x,y]:+.2f}',ha='center',va='center',color='white' if abs(ar[x,y])>lim*.3 else 'black')
            ax.set_xticks(range(3),list('NHL'));ax.set_yticks(range(3),list('NHL'));ax.set_xlabel('GX strategy');ax.set_ylabel('GD strategy');ax.set_title(measure+': '+col)
    fig.suptitle('Median avoided cases (thousands): positive = benefit');save(ctx,fig,'strategy_heatmaps.png')
    fig,axes=plt.subplots(1,2,figsize=(15,7));keys=[(s,col) for s in STRATEGIES[1:] for col in ['A1','A2']]
    for ax,measure in zip(axes,['reported','all_infections']):
        d=df[df.measure==measure].set_index(['solution','strategy']);ar=np.array([[np.sign(d.loc[(name,s),col]) for name in ids(ctx.n)] for s,col in keys])
        ax.imshow(ar,cmap='RdBu',vmin=-1,vmax=1,aspect='auto');ax.set_yticks(range(len(keys)),[s+' '+col for s,col in keys]);ticks=np.unique(np.linspace(0,ctx.n-1,min(10,ctx.n)).astype(int));ax.set_xticks(ticks,[ids(ctx.n)[k] for k in ticks],rotation=45);ax.set_title(measure+' (blue positive / red negative)')
    save(ctx,fig,'policy_consistency.png')
    representatives=[ids(ctx.n)[k] for k in sorted(set([0,max(0,ctx.n//2-1),ctx.n-1]))]
    fig,axes=plt.subplots(2,len(representatives),figsize=(5*len(representatives),7),squeeze=False)
    for i,name in enumerate(representatives):
        tr=np.load(ctx.trajectory_root/(name+'_daily_trajectories.npz'))
        for j in range(2):
            ax=axes[j,i]
            for strategy,color in [('NN','black'),('HH','tab:blue'),('LL','tab:orange')]:ax.plot(tr['time'][:-1]+.5,np.diff(tr[strategy][:,10+j]),label=strategy,color=color)
            ax.set_title(name+' '+['GD','GX'][j]);ax.set_yscale('symlog',linthresh=1);ax.set_ylabel('Daily cases (symmetric-log)');ax.set_xlabel('Day');ax.legend()
    save(ctx,fig,'representative_trajectories.png')

def main():
    ctx=Context();ranking_figures(ctx);primary_figures(ctx);print('All figures generated from saved artifacts; ODE calls=0')

if __name__=='__main__':main()
