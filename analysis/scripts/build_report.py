"""Run-specific report; archived scientific report is never rewritten."""
from scripts.common import *

def main():
    ctx=Context()
    if (ctx.root/'FINAL_REPORT.md').exists() and (ctx.root/'pipeline_state.json').exists():
        state=read_json(ctx.root/'pipeline_state.json')
        if state.get('stages',{}).get('build_report',{}).get('status')=='completed':
            raise RuntimeError('Committed historical report is read-only; use MainResults/current/MANUSCRIPT_SUMMARY.md for current manuscript framing.')
    r=ctx.results;rank=pd.read_csv(r/'pairwise_rankings.csv');stats=pd.read_csv(r/'policy_effect_size_summary.csv');policy=pd.read_csv(r/'policy_summary.csv');rs=pd.read_csv(r/'strategy_rank_summary.csv')
    parts=[]
    def add(x):parts.append(x)
    add(f'# UNIFIED UDE ANALYSIS REPORT — N={ctx.n}\n\n运行模式：`{ctx.mode}`。'+('本运行仅复用冻结v2的既有baseline审计、30解和270场景，未重新求解ODE、未重新搜索。原FINAL_UNIFORM_30_REPORT_V2.md及主结论保持原样；本报告为独立下游补充。' if ctx.mode=='reuse_frozen' else '本运行在独立目录按原科学模型与统一maximin规则生成新ensemble；不得与旧30解合并解释。'))
    add('## 1. Pairwise strategy-ranking stability\n\n矩阵行是策略A、列是策略B。单元格为本ensemble中A病例严格少于B的解数/N。三套指标分别为最终total reported C1+C2、GD reported C1、GX reported C2。它是ensemble fraction，不是概率、后验概率或对所有可行解的保证。严格相等不计入任一方向；对角线为0。CSV另列0.01例容差诊断，但不改主矩阵定义。')
    add('![Pairwise ranking stability](figures/pairwise_ranking_stability.png)')
    overview=[]
    for measure,d in rank.groupby('measure'):
        off=d[d.strategy_A!=d.strategy_B]
        overview.append(dict(measure=measure,unordered_pairs=36,unanimously_ordered_pairs=int((off.A_fewer_count==ctx.n).sum()),switching_pairs=int(((off.A_fewer_count>0)&(off.B_fewer_count>0)).sum()//2)))
    add(md_table(pd.DataFrame(overview)))
    for measure in ['total_reported','gd_reported','gx_reported']:
        d=rank[(rank.measure==measure)&(rank.strategy_A!=rank.strategy_B)&(rank.A_fewer_count==ctx.n)]
        add('### '+measure+'：跨ensemble一致的成对排序\n\n'+md_table(d[['strategy_A','strategy_B','A_fewer_count','n','cases_B_minus_A_median']]))
    add('### 各策略在9种策略中的名次范围\n\n'+md_table(rs)+'\n\n最低病例计数允许并列；unique_best_count仅计唯一最优。中位名次不代表跨所有解一致的完整排序。')
    add('## 2. 关键policy functionals：方向与幅度\n\nG_HH_minus_LL=Atotal(HH)−Atotal(LL)，正值表示HH相对LL更好；其余A均以每条解自身NN为参照，正值表示避免病例。A_GD_HN和A_GX_NH是各自控制者效应，负值表示控制者受损。分位数仅描述该有限ensemble的经验分布，不是置信区间。')
    add(md_table(stats[['measure','functional','positive_count','negative_count','min','q25','median','q75','max']]))
    add('![Reported effect ECDF](figures/policy_effect_ecdf_reported.png)\n\n![All-infections effect ECDF](figures/policy_effect_ecdf_all_infections.png)\n\nECDF横轴使用对称对数刻度以展示正负及大幅度变化，±10例内保持线性；CSV保留未变换病例数。纵轴是“效应不大于该值的解占比”，不作概率解释。')
    add('## 3. 同一3×3策略汇总\n\n'+md_table(policy[['measure','strategy','win_win_count','self_harm_neighbor_benefit_count','burden_shifting_count','lose_lose_count','Atotal_positive_count','Atotal_median']]))
    if ctx.mode=='reuse_frozen':add('上述30解敏感性汇总与原uniform_30_policy_summary_v2.csv一致：两口径HH优于LL及LL总量有害均30/30；HH双赢reported 20/30、all-infections diagnostic 23/30。它是ensemble-size sensitivity，不替代N=100正文主分析，也不能把相对排序当作绝对双赢。')
    audit=read_json(r/'baseline_audit.json');m=pd.read_csv(r/'ensemble_manifest.csv');dist=pd.read_csv(r/'ensemble_distances.csv',index_col=0).values;v=dist[np.triu_indices(ctx.n,1)]
    add('## 4. Baseline、ensemble与可复现性\n\n'+md_table(pd.DataFrame(audit['metrics']))+f'\n\nbeta固定为新baseline估计值{audit["beta"]:.12g}。全部{ctx.n}条解R²>0.98。函数距离最小/中位/最大为{v.min():.6f}/{np.median(v):.6f}/{v.max():.6f}。科学边界、ODE、观测方程、118天风险窗口和0.2倍干预保持原样。')
    add('![Parameters](figures/parameter_functions.png)\n\n![Distances](figures/ensemble_distances.png)\n\n![Fluxes](figures/spillover_functions.png)\n\n![Strategy medians](figures/strategy_heatmaps.png)\n\n![Policy signs](figures/policy_consistency.png)\n\n![Representative trajectories](figures/representative_trajectories.png)')
    add('## 5. 当前论文框架、数值验证范围与完整性\n\n现有30/50/100 analyses均已完成、冻结并通过各自保存的终检。N=100（analysis_runs/sensitivity_100）为manuscript main ensemble；N=30（历史路径main_30）和N=50仅为ensemble-size sensitivity。历史路径保留provenance，不决定论文角色。当前汇总见MainResults/current/MANUSCRIPT_SUMMARY.md，正文显示契约见MainResults/manuscript_config.json。\n\n候选来自固定baseline、有限30维deformation/inverse-reconstruction family，由policy-blind constrained maximin构造diverse admissible solution ensemble；不是完整无限维空间的uniform random sample，也不是posterior sample。\n\nBaseline及admissible members按原实现进行独立RK4/DOP853验证。九策略全部具有fine/coarse RK4 consistency checks；独立DOP853 strategy cross-check仅覆盖HH、LL。All-infections是诊断量，其流定义为beta*c_i*S_i*I_i/N_i，不按感染者最终所在地重新归属。\n\n原运行的config、source hashes、freeze、pipeline_state及历史FINAL_REPORT保留不变。当前维护代码不伪装成旧hash-pinned代码；外部审计入口为python -m scripts.audit_manuscript，只对保存产物作数组/代数检查。此生成器仅供尚未提交报告的独立复现运行使用，不重写现有冻结运行报告。')
    (ctx.root/'FINAL_REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    atomic_json(r/'ranking_overview.json',dict(ensemble_size=ctx.n,mode=ctx.mode,overview=overview,interpretation='ensemble fraction, not probability',ODE_calls_in_downstream=0))
    print('Final report generated')

if __name__=='__main__':main()
