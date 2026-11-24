"""Build a traceable T6 numerical report from audited saved/derived data only."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np,pandas as pd

T6=Path(__file__).resolve().parents[1]; ROOT=T6.parent.parent
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 checks=[T6/'audit_protection/representative_saved_audit.json',T6/'audit_protection/fine_trajectories_independent_audit.json',T6/'audit_protection/flow_arrays_independent_audit.json']
 for p in checks:require(json.loads(p.read_text(encoding='utf8')).get('passed') is True,'missing or failed audit '+str(p))
 original=json.loads(checks[0].read_text(encoding='utf8'))
 for p,h in original['frozen_files'].items():require(Path(p).is_file() and sha(Path(p))==h,'changed original source '+p)
 # All data derived by the separate diagnostic producer must remain unchanged.
 manifests=[T6/'results/comparison/sha256_manifest.json',T6/'results/U01/sha256_manifest.json',T6/'results/U29/sha256_manifest.json']
 for mp in manifests:
  stored=json.loads(mp.read_text(encoding='utf8'))
  for name,h in stored.items():
   if isinstance(h,str):
    p=Path(name); p=p if p.is_absolute() else (ROOT/p if (ROOT/p).is_file() else mp.parent/p)
    require(p.is_file() and sha(p)==h,'mismatched derived input '+str(p))
 facts=(T6/'results/comparison/trajectory_facts.md').read_text(encoding='utf8')
 lines=['# T6 极端反事实差值：数值实验总结','','## 研究范围和代表解选择','','仅研究原冻结100解；未重新训练、生成新机制解或更改动力学、初值、观测与干预协议。H/L均保留原风险时段和0.2跨区乘数。所有数据截止第236天。','',f'对原100解报告病例 G=LL−HH 排序：最大成员 U01 为 {original["maximum"]["G"]:.9f}；中位数为 {original["median_G"]:.9f}。有偶数个成员时中位数是第50与第51的均值，U52={original["middle_two"]["U52"]:.9f} 和 U29={original["middle_two"]["U29"]:.9f} 与中位数数学上等距；本次以编号较小者作为确定性平局规则选择 U29，不将其描述成唯一中位成员。','','原机制函数及beta来自 `analysis_runs/sensitivity_100/solutions/U01.npz` 与 `U29.npz`，九策略逐解结果、风险协议和源SHA见 `audit_protection/representative_saved_audit.json`。原归档各策略只有237个日点的14维轨迹；由于流量积分、日内峰及非整数窗口切换需要细分，仅对这两个解的NN/HH/LL六个场景按照原参数、固定窗口和0.00625天RK4补算不大于0.025天的事件网格。六条轨迹全部237个日点、14个状态与冻结结果逐值相同，最大总体人口误差1.55e-6人。','','## 已核验的轨迹事实与解释边界','',facts.replace('# T6 轨迹与流量数值事实','').strip(),'','## 溯源与数值核验','','原存档中三策略都有细/粗RK4检查；HH/LL另有独立DOP853存档检查，NN没有存档DOP853策略记录，不将之表述为独立NN复验。原 U01 LL RK4细粗误差 0.000716545、DOP853差 0.000048913；U29 LL 分别为 0.000029268、0.000001965。各场景保存误差在 `audit_protection/representative_saved_audit.json`。本次细网格和后续只读审计没有重新执行DOP853。','','独立数组审计复算状态/流量公式和分段数值积分。mA、dI与净流的积分为0.025日事件分段梯形近似，不是精确ODE累计量。U01 LL GD最大单日 qβλ 积分与精确保存累计状态差分差1.079875例，I收支积分残差约−0.640711人；所有日报和新感染日量使用C/J状态差分。半步插值积分比较不是重新求解ODE，也非严格误差界。','','已保存轨迹和流量足以定位差异的时间和地区来源，无需运行额外m-only或d-only情景。仅凭现有轨迹不能把HH–LL总体差值分配给m或d的独立因果贡献；模型内部跨区流不等同于真实观测人员流或输入病例。','','## 数据、图表与代码','','- 两个补算细网格：`data/U01_fine_trajectories.npz`、`data/U29_fine_trajectories.npz`。','- 每个解的 `results/U01/`、`results/U29/`：`fine_states_and_instantaneous_flows.csv`、`fine_interval_flow_integrals.csv`、`daily_flow_integrals.csv`、`HH_LL_gap_time_series.csv`、`peaks_by_region_and_metric.csv`、`strategy_burdens_and_peaks.csv`、`quadrature_stability.json` 和 `trajectory_diagnostic.png`。','- 两解对照 `results/comparison/`：`trajectory_facts.md`、`window_flow_and_gap.csv`、`gap_threshold_times.csv`、`late_rate_means.csv`、`peaks_both_solutions.csv`、`U01_flow_diagnostic.png`、`U29_flow_diagnostic.png`。','- 轨迹求解 `code_saved_solver/verify_and_rollout.py`；流量/图表 `scripts/flow_diagnostics.py`；独立审计 `audit_protection/`。文件SHA清单见各结果文件夹 `sha256_manifest.json` 和本报告下方。','','## 关键来源 SHA256','']
 rates=pd.read_csv(T6/'results/comparison/late_rate_means.csv'); rates=rates[(rates.strategy=='LL')&(rates.window_start==177)&(rates.window_end==236)].set_index('solution')
 require(set(rates.index)=={'U01','U29'},'late rate comparison missing')
 section=['','## 两机制在LL晚期的参数对照','','以下为第177–236天的时间均值；c和q保持机制函数原值，m和d是LL干预乘数作用后的实际值。它们可提示不同时变机制对轨迹的敏感性，但不能将其当作HH–LL差值的因果归因。','','| 解 | c1 | c2 | q1 | q2 | m1（LL实际） | m2（LL实际） | d1（LL实际） | d2（LL实际） |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for ident in ['U01','U29']:
  row=rates.loc[ident];section.append('| '+ident+' | '+' | '.join(f'{row[k]:.6g}' for k in ['c1_time_mean','c2_time_mean','q1_time_mean','q2_time_mean','m1_time_mean','m2_time_mean','d1_time_mean','d2_time_mean'])+' |')
 section+=['','源表 `results/comparison/late_rate_means.csv` 包含其他预设时间窗。数值检验 `logs/flow_tests.log` 的3项流量/事件边界单元测试通过。最终原工程4784项、T2–T5/shared 9879项保护检查 `audit_protection/protected_check_final_delivery.json` 通过，未发现变动。','']
 idx=lines.index('## 关键来源 SHA256');lines[idx:idx]=section
 paths=checks+manifests+[T6/'audit_protection/protected_check_final_delivery.json',T6/'logs/flow_tests.log',T6/'data/U01_fine_trajectories.npz',T6/'data/U29_fine_trajectories.npz',T6/'results/comparison/trajectory_facts.md',T6/'results/comparison/window_flow_and_gap.csv',T6/'results/comparison/gap_threshold_times.csv',T6/'results/comparison/late_rate_means.csv',T6/'results/comparison/peaks_both_solutions.csv',T6/'results/comparison/U01_flow_diagnostic.png',T6/'results/comparison/U29_flow_diagnostic.png',T6/'results/U01/trajectory_diagnostic.png',T6/'results/U29/trajectory_diagnostic.png']
 for p in paths:lines.append(f'- `{p}`：`{sha(p)}`')
 out=T6/'reports/T6_numerical_summary.md';out.parent.mkdir(parents=True,exist_ok=True);temp=out.with_suffix('.md.tmp');temp.write_text('\n'.join(lines)+'\n',encoding='utf8');temp.replace(out);print(out)
if __name__=='__main__':main()
