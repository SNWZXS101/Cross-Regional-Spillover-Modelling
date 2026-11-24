"""CSV-only ranking stability and effect-size analysis. No ODE dependencies."""
from scripts.common import *

MEASURES={'total_reported':lambda d:d.C1+d.C2,'gd_reported':lambda d:d.C1,'gx_reported':lambda d:d.C2}
FUNCTIONALS={'G_HH_minus_LL':('HH','LL','Atotal'),'A_total_HH':('HH',None,'Atotal'),'A_GD_HN':('HN',None,'A1'),'A_GX_NH':('NH',None,'A2')}

def ranking_tables(df,n):
    validate_scenarios(df,n);reported=df[df.measure=='reported'];long=[];ranks=[];matrices={}
    for measure,get_values in MEASURES.items():
        d=reported.copy();d['cases']=get_values(d)
        p=d.pivot(index='solution',columns='strategy',values='cases').reindex(index=ids(n),columns=STRATEGIES)
        matrices[measure]=np.zeros((9,9))
        for i,a in enumerate(STRATEGIES):
            for j,b in enumerate(STRATEGIES):
                gain=p[b]-p[a] # positive means row strategy A has fewer cases
                wins=int((gain>0).sum());ties=int((gain==0).sum());losses=int((gain<0).sum())
                matrices[measure][i,j]=wins/n
                long.append(dict(measure=measure,strategy_A=a,strategy_B=b,n=n,A_fewer_count=wins,B_fewer_count=losses,equal_count=ties,ensemble_fraction=wins/n,strict_comparison=True,A_fewer_beyond_001=int((gain>.01).sum()),within_001_count=int((abs(gain)<=.01).sum()),cases_B_minus_A_min=float(gain.min()),cases_B_minus_A_median=float(gain.median()),cases_B_minus_A_max=float(gain.max())))
        rr=p.rank(axis=1,method='average',ascending=True)
        for strategy in STRATEGIES:
            ranks.append(dict(measure=measure,strategy=strategy,best_rank=float(rr[strategy].min()),median_rank=float(rr[strategy].median()),worst_rank=float(rr[strategy].max()),minimum_cases_count=int(p[strategy].eq(p.min(axis=1)).sum()),unique_best_count=int((p[strategy].eq(p.min(axis=1))&(p.eq(p.min(axis=1),axis=0).sum(axis=1)==1)).sum())))
    return pd.DataFrame(long),pd.DataFrame(ranks),matrices

def main():
    ctx=Context();p=ctx.results/'complete_scenarios.csv';df=validate_scenarios(pd.read_csv(p),ctx.n)
    long,ranks,matrices=ranking_tables(df,ctx.n);long.to_csv(ctx.results/'pairwise_rankings.csv',index=False);ranks.to_csv(ctx.results/'strategy_rank_summary.csv',index=False)
    for measure,arr in matrices.items():pd.DataFrame(arr,index=STRATEGIES,columns=STRATEGIES).rename_axis('strategy_A').to_csv(ctx.results/f'pairwise_ranking_{measure}.csv')
    long[(long.strategy_A!=long.strategy_B)&(long.A_fewer_count==ctx.n)].to_csv(ctx.results/'unanimous_strategy_rankings.csv',index=False)
    effects=[]
    for (solution,measure),block in df.groupby(['solution','measure']):
        d=block.set_index('strategy')
        for name,(a,b,column) in FUNCTIONALS.items():
            value=float(d.loc[a,column]-(d.loc[b,column] if b else 0))
            effects.append(dict(solution=solution,measure=measure,functional=name,effect_cases=value))
    effects=pd.DataFrame(effects);effects.to_csv(ctx.results/'policy_effect_sizes.csv',index=False);summary=[]
    for (measure,functional),d in effects.groupby(['measure','functional']):
        v=d.effect_cases
        summary.append(dict(measure=measure,functional=functional,n=ctx.n,positive_count=int((v>0).sum()),negative_count=int((v<0).sum()),zero_count=int((v==0).sum()),neutral_001_count=int((abs(v)<=.01).sum()),min=float(v.min()),q05=float(v.quantile(.05)),q25=float(v.quantile(.25)),median=float(v.median()),q75=float(v.quantile(.75)),q95=float(v.quantile(.95)),max=float(v.max()),mean=float(v.mean()),std=float(v.std(ddof=1))))
    pd.DataFrame(summary).to_csv(ctx.results/'policy_effect_size_summary.csv',index=False)
    atomic_json(ctx.root/'methods/ranking_definition.json',dict(definition='row A cases < column B cases; count / n_solutions',denominator=ctx.n,comparison='strict <; exact ties contribute to neither direction; diagonal=0',fraction_interpretation='ensemble fraction, NOT probability or posterior probability',cases='Final C1+C2, C1, C2; reported measure only',source_sha256=sha(p),ODE_calls=0,functions='G_HH_minus_LL=Atotal(HH)-Atotal(LL); A_total_HH; A_GD_HN; A_GX_NH',effect_plot_scales='raw-case ECDF; symlog x axis for wide ranges; zero reference'))
    print('CSV-only pairwise ranking complete:',len(long),'ordered comparisons; no ODE calls')

if __name__=='__main__':main()
