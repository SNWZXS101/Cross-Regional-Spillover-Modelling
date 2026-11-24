"""Summarize observed trajectory facts, not a causal attribution."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T6=Path(__file__).resolve().parents[1];O=T6/'results/comparison';O.mkdir(parents=True,exist_ok=True)
start=pd.Timestamp('2022-01-29');lines=['# T6 轨迹与流量数值事实','', '全部结果限于236天；本报告从保存/限定复算的轨迹代数计算，不证明单一机制的因果贡献。每日病例和感染量来自累计状态差分；流量分量是按风险事件分段、0.025天网格梯形近似。','','## 分期结果']
windows=[];threshold=[];peak_rows=[];cross=[]
for sid in ['U01','U29']:
 path=T6/'results'/sid;daily=pd.read_csv(path/'daily_flow_integrals.csv');fine=pd.read_csv(path/'fine_states_and_instantaneous_flows.csv');g=pd.read_csv(path/'HH_LL_gap_time_series.csv');tot=float(g.reported_gap_total.iloc[-1]);
 LL=daily[daily.strategy=='LL'].set_index('day_start');HH=daily[daily.strategy=='HH'].set_index('day_start')
 lines+=['',f'### {sid}','',f'第236天累计报告差 LL−HH：总计 {tot:.6f}，广东 {g.reported_gap_GD.iloc[-1]:.6f}，广西 {g.reported_gap_GX.iloc[-1]:.6f}。']
 for frac in [.01,.1,.5,.9]:
  ix=np.flatnonzero(g.reported_gap_total.to_numpy()>=frac*tot);day=float(g.day.iloc[ix[0]]) if len(ix) else np.nan
  threshold.append(dict(solution=sid,threshold_fraction=frac,first_reach_day=day,date=(start+pd.Timedelta(days=int(day))).date().isoformat()))
 gap_by_day=g.set_index('day').reported_gap_total
 lines.append(f'- 第229天差值 {gap_by_day.loc[229.]:.2f}，第236天 {gap_by_day.loc[236.]:.2f}，最后7天内增长 {gap_by_day.loc[236.]-gap_by_day.loc[229.]:.2f}；仅说明研究窗内差值尚在变化。')
 for lo,hi in [(0,59),(59,118),(118,177),(177,236)]:
  a=LL.loc[lo:hi-1];b=HH.loc[lo:hi-1];row={'solution':sid,'day_start':lo,'day_end':hi}
  for reg in ['GD','GX']:
   row[f'{reg}_reported_gap_increment']=float((a[f'{reg}_reported_daily_exact']-b[f'{reg}_reported_daily_exact']).sum())
   for name in ['new_infections','unisolated_inflow_A','outgoing_mA','outgoing_dI','net_inflow']:
    row[f'{reg}_LL_{name}']=float(a[f'{reg}_{name}_integral'].sum());row[f'{reg}_HH_{name}']=float(b[f'{reg}_{name}_integral'].sum())
  row['reported_gap_total_increment']=row['GD_reported_gap_increment']+row['GX_reported_gap_increment'];row['fraction_of_final_gap']=row['reported_gap_total_increment']/tot;windows.append(row)
  lines.append(f'- 第 {lo}–{hi} 天报告病例差增加：广东 {row["GD_reported_gap_increment"]:.2f}、广西 {row["GX_reported_gap_increment"]:.2f}，合计 {row["reported_gap_total_increment"]:.2f}（最终差的 {row["fraction_of_final_gap"]:.2%}）。')
  if lo==177:lines.append(f'  同期LL的A积分约GD {row["GD_LL_unisolated_inflow_A"]:.2f}/GX {row["GX_LL_unisolated_inflow_A"]:.2f}，净感染者流入约GD {row["GD_LL_net_inflow"]:.2f}/GX {row["GX_LL_net_inflow"]:.2f}；这些是同一情景状态流的积分，不是HH–LL差异的因果份额。')
 for s in ['NN','HH','LL']:
  z=fine[fine.strategy==s];burden=pd.read_csv(path/'strategy_burdens_and_peaks.csv');v=burden[(burden.strategy==s)&(burden.region=='total')].iloc[0]
  lines.append(f'- {s}: 总报告 {v.reported:.3f}，总感染诊断 {v.all_infections:.3f}；分别占总参考人口 {v.reported_fraction:.6%}/{v.all_infections_fraction:.6%}。')
  for reg in ['GD','GX']:
   rr=burden[(burden.strategy==s)&(burden.region==reg)].iloc[0]; lines.append(f'  {reg}: reported {rr.reported:.2f}，all-infections {rr.all_infections:.2f}，未隔离I细网格峰 {rr.peak_unisolated:.2f}（第{rr.peak_day:.3f}天），最终 S/固定N={rr.final_susceptible_fraction:.6f}。')
 peak_rows.extend(pd.read_csv(path/'peaks_by_region_and_metric.csv').to_dict('records'))
 for lo,hi in [(118,177),(177,236)]:
  z=fine[(fine.strategy=='LL')&(fine.day>=lo)&(fine.day<hi)];rec={'solution':sid,'strategy':'LL','window_start':lo,'window_end':hi}
  for key in ['c1','c2','q1','q2','m1','m2','d1','d2']:rec[key+'_time_mean']=float(np.trapezoid(z[key],z.day)/(z.day.iloc[-1]-z.day.iloc[0]))
  cross.append(rec)
 # Diagnostic panels use calendar-daily state differences; no smoothing.
 fig,ax=plt.subplots(4,2,figsize=(13,12));colors={'NN':'#637b87','HH':'#007b7c','LL':'#bd8436'}
 for s in ['NN','HH','LL']:
  x=daily[daily.strategy==s];c=colors[s];day=x.day_start
  for j,reg in enumerate(['GD','GX']):
   ax[0,j].plot(day,x[f'{reg}_infections_daily_exact'],color=c,label=s);ax[1,j].plot(day,x[f'{reg}_outgoing_mA_integral'],color=c,ls='-',label=s+' mA');ax[1,j].plot(day,x[f'{reg}_outgoing_dI_integral'],color=c,ls='--',label=s+' dI')
   ax[2,j].plot(day,x[f'{reg}_net_inflow_integral'],color=c);z=fine[fine.strategy==s];ax[3,j].plot(z.day,z[f'q{j+1}'],color=c)
  
 for j,reg in enumerate(['GD','GX']):
  for i,label in enumerate(['new infections / day','outgoing mA, dI / day','net infectious inflow / day','q']):ax[i,j].set_ylabel(reg+' '+label);ax[i,j].set_xlabel('day')
 ax[0,0].legend(frameon=False);ax[1,0].legend(frameon=False,ncol=2,fontsize=8);fig.tight_layout();fig.savefig(O/(sid+'_flow_diagnostic.png'),dpi=180);plt.close(fig)
pd.DataFrame(windows).to_csv(O/'window_flow_and_gap.csv',index=False,float_format='%.17g');pd.DataFrame(threshold).to_csv(O/'gap_threshold_times.csv',index=False,float_format='%.17g');pd.DataFrame(peak_rows).to_csv(O/'peaks_both_solutions.csv',index=False,float_format='%.17g');pd.DataFrame(cross).to_csv(O/'late_rate_means.csv',index=False,float_format='%.17g')
lines+=['','## 动力学解释与限制','','U01 LL晚期两省局部感染量与未隔离感染者峰值显著上升，未隔离易感者可用量S亦明显变化；广西较早的未隔离感染峰与广东较晚的峰在时间上相隔约50天。S方程还包括易感者隔离、解除隔离及跨区易感者转移，不能将S变化或终点S下降全部归因于感染消耗。U01同一省份HH与LL的I全局峰日反而很接近：广东217.850与217.225天，广西167.025与166.950天；主要差异是峰值幅度，不能声称这两种策略在同一地区出现数十天峰日错位。病例日报峰日与I存量峰日为不同指标，见`peaks_both_solutions.csv`。177–236天区间内广西净未隔离感染者流入而广东相应净流出；q、接触率及两种方向流可在关联CSV逐时核查。以上说明局部增殖、地区间重分配、隔离比例与剩余易感者共同影响模型轨迹，但不能把非线性情景差异拆成可相加的因果贡献。','', '观察期终点处两策略仍可能处在不同波段或不同耗竭阶段；目前只报告236天内差值，不预测其后的最终疫情总量。未作单独 m/d 消融，因此哪一种传输分量对最终G的因果贡献最大仍无法确认。现存轨迹可解释差值的时间与地区来源，无需将多次额外模拟当作该结论前提。','', '流量积分按事件分段：每个区间采用中点固定的风险乘数，不跨窗口对阶跃做梯形。`quadrature_stability.json` 的半段结果使用轨迹与参数的线性插值，评估积分网格敏感性，不是额外ODE求解且不是严格误差上界。状态C/J差分用于每日病例。']
(O/'trajectory_facts.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
inputs=[]
for sid in ['U01','U29']:
 for name in ['daily_flow_integrals.csv','fine_states_and_instantaneous_flows.csv','HH_LL_gap_time_series.csv','strategy_burdens_and_peaks.csv','peaks_by_region_and_metric.csv','quadrature_stability.json']:inputs.append(T6/'results'/sid/name)
files=inputs+[p for p in O.iterdir() if p.is_file() and p.name!='sha256_manifest.json'];(O/'sha256_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(O)
