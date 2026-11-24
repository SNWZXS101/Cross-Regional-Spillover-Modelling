from pathlib import Path
import hashlib
import pandas as pd
import train_ude
root=Path(__file__).resolve().parents[1]
for n in ('Data_all.xlsx','Data_all_sm.xlsx'):
 p=root/'data/input'/n
 frame=pd.read_excel(p)
 print(n, 'rows',len(frame),'columns',list(frame.columns),'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
 if len(frame)!=236 or any(c not in frame.columns for c in ('GD','GX')):raise RuntimeError('Unexpected input schema')
 if frame[['GD','GX']].isna().any().any():raise RuntimeError('Missing data')
train_ude.CONFIG.data_file=str(root/'data/input/Data_all_sm.xlsx')
train_ude.CONFIG.device='cpu'
data=train_ude.prepare_data(train_ude.CONFIG)
print('Training input prepared:',data['n_days'],'days; count scale',data['count_scale'])
