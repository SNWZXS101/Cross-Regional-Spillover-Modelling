"""Render S4–S7 solely from audited saved arrays, in clean/marked form."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT
registry=json.loads((OUT/'scripts/figure_source_registry.json').read_text(encoding='utf8'))
paths={k:ROOT/v for k,v in registry['input_paths'].items()}
names=['NN − LL: total','LL − HH: total','NN − HH: total','NN − HN: Guangdong','NN − NH: Guangxi']
blue='#0072B2';green='#009E73';orange='#D55E00';grey='#777777'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def effect(df,group,measure,typ):
 block=df[(df.group==group)&(df.measure==measure)]
 def get(s,r):
  d=block[(block.strategy==s)&(block.region==r)].set_index('solution').avoided.sort_index();assert not d.index.duplicated().any() and np.isfinite(d.to_numpy()).all();return d
 if typ==0:return get('LL','total')
 if typ==1:
  hh=get('HH','total');ll=get('LL','total');assert hh.index.equals(ll.index);return hh-ll
 if typ==2:return get('HH','total')
 if typ==3:return get('HN','GD')
 return get('NH','GX')
def boxes(ax,seq,labels,colors,readable_ticks=False):
 allvalues=[]
 for i,(series,color) in enumerate(zip(seq,colors),1):
  arr=series.to_numpy(float);allvalues.extend(arr.tolist());ax.boxplot([arr],positions=[i],vert=False,widths=.52,showfliers=False,patch_artist=True,boxprops={'facecolor':'white','edgecolor':color,'linewidth':.8},medianprops={'color':color,'linewidth':1.1},whiskerprops={'color':color,'linewidth':.7},capprops={'color':color,'linewidth':.7})
  jitter=np.array([((int(k[1:])*37)%31-15)/90 for k in series.index]);ax.scatter(arr,i+jitter,s=2.3,c=color,alpha=.45,linewidths=0,rasterized=False,zorder=3)
 ax.set_yticks(range(1,len(labels)+1),labels);ax.invert_yaxis();ax.axvline(0,color='0.5',lw=.6);ax.set_xscale('symlog',base=10,linthresh=100,linscale=1);ax.margins(x=.05)
 lo,hi=ax.get_xlim();ticks=([-1e8,-1e6,-1e4,0,1e4,1e6,1e8] if readable_ticks else [-1e8,-1e6,-1e4,-100,0,100,1e4,1e6,1e8]);ax.set_xticks([x for x in ticks if lo<=x<=hi]);ax.set_xlim(lo,hi)
 assert len(allvalues)>0 and all(np.isfinite(allvalues))
def panel(ax,i,size=9):ax.text(-.13,1.03,chr(ord('a')+i),transform=ax.transAxes,fontweight='bold',fontsize=size,va='bottom')
def save(fig,num,mode):
 if mode=='marked':
  for text in fig.findobj(match=matplotlib.text.Text):text.set_fontweight('bold')
 d=OUT/'figures'/mode;d.mkdir(parents=True,exist_ok=True);fig.savefig(d/f'FigS{num}.pdf',bbox_inches='tight');fig.savefig(d/f'FigS{num}.svg',bbox_inches='tight');fig.savefig(d/f'FigS{num}.png',dpi=300,bbox_inches='tight');plt.close(fig)
def f4(df,mode):
 fig,axs=plt.subplots(3,2,figsize=(7.5,7.9));raw=pd.read_excel(paths['raw']);a=np.load(paths['observation_fit']);pred=np.diff(a['dop853'][:,10:12],axis=0);assert len(raw)==236 and a['observed'].shape==(236,2) and np.allclose(raw[['GD','GX']],a['observed'])
 for j in range(2):
  ax=axs[0,j];ax.scatter(np.arange(236),raw[['GD','GX']].iloc[:,j],s=2,c='black',label='Unsmoothed observations');ax.plot(np.arange(236),pred[:,j],color=blue,lw=.8,label='Unsmoothed-data fit');ax.set_xlabel('Reporting day');ax.set_ylabel('Reported cases / day');ax.text(.97,.75 if j==0 else .93,['R² = 0.95836','R² = 0.88620'][j],ha='right',va='top',transform=ax.transAxes);panel(ax,j)
 axs[0,0].legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=1)
 groups=registry['mechanism_groups'];labels=['Primary (n=100)','Alt. baseline (seed 27182)','Alt. baseline (seed 16180)','Member-specific constant β','Six-knot representation','Modified initial infections','External case introductions','Unsmoothed observations']
 for k,(typ,title) in enumerate([(1,'LL − HH: total cases'),(2,'NN − HH: total cases'),(3,'NN − HN: GD cases'),(4,'NN − NH: GX cases')]):
  ax=axs[1+k//2,k%2];seq=[effect(df,g,'reported',typ) for g in groups];assert [len(x) for x in seq]==[100]+[30]*7;boxes(ax,seq,labels,[blue]+[green]*7,readable_ticks=True);ax.set_xlabel(title);panel(ax,k+2,11)
 fig.tight_layout();save(fig,4,mode)
def f5(df,mode):
 fig=plt.figure(figsize=(7.5,7.4));gs=gridspec.GridSpec(3,2,figure=fig);axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[2,:])];groups=registry['policy_groups'];labels=['Primary','59-day periods','50% restriction','95% restriction','Preceding-days score']
 for k,ax in enumerate(axs):
  seq=[effect(df,g,'reported',k) for g in groups];assert all(len(x)==100 for x in seq);boxes(ax,seq,labels,[blue,green,green,green,green],readable_ticks=True);ax.set_xlabel(names[k]+' cases');panel(ax,k,11)
 neg=effect(df,registry['policy_groups'][-1],'reported',1);assert set(neg[neg<0].index)==set(registry['reversal_members'])
 fig.tight_layout();save(fig,5,mode)
def f6(df,mode):
 fig=plt.figure(figsize=(6.69,5.51));gs=gridspec.GridSpec(2,2,figure=fig);axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])];f=np.array(json.loads(paths['detection'].read_text(encoding='utf8'))['baseline']['fractions_GD_GX'])*100;ax=axs[0];ax.bar([0,1],100-f,color=blue,label='Reporting at infection');ax.bar([0,1],f,bottom=100-f,color=orange,label='Subsequent detection');ax.set_xticks([0,1],['Guangdong','Guangxi']);ax.set_ylim(0,100);ax.set_ylabel('Share of baseline reported cases (%)');ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02));
 for j in [0,1]:ax.text(j,100-f[j]/2,f'{f[j]:.2f}%',ha='center',va='center',color='white');ax.text(j,(100-f[j])/2,f'{100-f[j]:.2f}%',ha='center',va='center',color='white')
 panel(ax,0)
 for j,measure in enumerate(['reported','all_infections']):
  ax=axs[j+1];seq=[effect(df,registry['detection_group'],measure,k) for k in range(5)];assert all(len(x)==30 for x in seq);boxes(ax,seq,names,[blue,green,blue,green,blue]);ax.set_xlabel('Difference in '+('reported cases' if j==0 else 'all-infections diagnostic'));panel(ax,j+1)
 g=effect(df,registry['detection_group'],'reported',1);assert (g>0).sum()==10 and (g<0).sum()==20
 fig.tight_layout();save(fig,6,mode)
def f7(mode):
 fig,axs=plt.subplots(4,2,figsize=(6.69,7.48));peaks=pd.read_csv(paths['peaks']);states=[pd.read_csv(paths['maximum']),pd.read_csv(paths['median'])]
 for col,(ident,df) in enumerate(zip(registry['representatives'],states)):
  ax=axs[0,col]
  for s,c in [('NN',grey),('HH',blue),('LL',orange)]:
   d=df[df.strategy==s];ax.plot(d.day,d.C1+d.C2,label=s,color=c,lw=.8)
  ax.set_yscale('symlog',linthresh=100);ax.set_ylabel('Cumulative reported cases');ax.legend(frameon=False,ncol=3,fontsize=7.5)
  for s,ls in [('HH','-'),('LL','--')]:
   d=df[df.strategy==s];axs[1,col].plot(d.day,d.I1,color=blue,ls=ls,lw=.8,label='GD '+s);axs[1,col].plot(d.day,d.I2,color=green,ls=ls,lw=.8,label='GX '+s)
   axs[3,col].plot(d.day,d.S1/127000000,color=blue,ls=ls,lw=.8);axs[3,col].plot(d.day,d.S2/50370000,color=green,ls=ls,lw=.8)
  axs[1,col].set_yscale('symlog',linthresh=100);axs[1,col].set_ylabel('Unisolated infectious stock');axs[1,col].legend(frameon=False,fontsize=7.5,ncol=2,loc='upper left')
  for reg,color in [('GD',blue),('GX',green)]:
   x=peaks[(peaks.solution==ident)&(peaks.strategy=='LL')&(peaks.region==reg)&(peaks.metric=='unisolated_infectious_stock')].iloc[0];axs[1,col].text(.99,1.24 if reg=='GD' else 1.14,f'{reg} LL peak: {x.value:.0f}; day {x.day:.3f}',transform=axs[1,col].transAxes,fontsize=7.5,color=color,ha='right',va='top')
  for s,c in [('HH',blue),('LL',orange)]:
   d=df[df.strategy==s];axs[2,col].plot(d.day,d.GD_net_inflow_right,color=c,label=s,lw=.8)
  axs[2,col].set_yscale('symlog',linthresh=1);axs[2,col].set_ylabel('GD net inflow (individuals / day)');axs[2,col].legend(frameon=False,fontsize=7.5)
  axs[3,col].set_ylabel('S / N');
  for row in range(4):
   ax=axs[row,col];ax.axvspan(177,236,color='0.94',zorder=-1);ax.set_xlim(0,236);ax.set_xticks([0,59,118,177,236]);ax.set_xlabel('Day');panel(ax,row*2+col)
  axs[0,col].text(.5,1.04,['Maximum contrast','Near-median contrast'][col],ha='center',va='bottom',transform=axs[0,col].transAxes)
 for row in range(4):
  low=min(axs[row,0].get_ylim()[0],axs[row,1].get_ylim()[0]);high=max(axs[row,0].get_ylim()[1],axs[row,1].get_ylim()[1]);axs[row,0].set_ylim(low,high);axs[row,1].set_ylim(low,high)
 for ax in axs[2]:
  lo,hi=ax.get_ylim();ax.set_yticks([x for x in [-1e6,-1e4,-100,0,100,1e4,1e6] if lo<=x<=hi]);ax.set_ylim(lo,hi)
 fig.tight_layout(h_pad=3.0);save(fig,7,mode)
def main():
 mechan=pd.read_csv(paths['mechanisms']);pol=pd.read_csv(paths['policies'])
 for mode in ['clean','marked']:
  plt.rcParams.update({'font.family':'Arial','font.size':7.8,'font.weight':'bold' if mode=='marked' else 'normal','axes.labelweight':'bold' if mode=='marked' else 'normal','axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
  with plt.rc_context({'font.size':10.2,'xtick.labelsize':9.5,'ytick.labelsize':9.5}):
   f4(mechan,mode);f5(pol,mode)
  f6(mechan,mode);f7(mode)
 manifest={'inputs':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()},'script_sha256':sha(Path(__file__)),'outputs':{str(p.relative_to(OUT)):sha(p) for d in ['clean','marked'] for p in (OUT/'figures'/d).glob('FigS[4567].*')},'contract':'saved data only; matched members; no ODE; marked changes text weight only'}
 (OUT/'figures/supplementary_figure_sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
if __name__=='__main__':main()

