"""
health_extension_v2.py — Health Extension with chunked reading for large MIMIC-IV
"""
import os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 24
PATCH_LEN= 4
N_PATCHES= 6
RUL_CLIP = 14
NF_HEALTH= 5
BACKBONE  = r"E:\rul_project\iclr_backbone_v2.pt"
CHART_CSV = r"E:\rul_project\chartevents_full.csv"
STAYS_CSV = r"E:\rul_project\icustays_full.csv"

VITAL_ITEMS = {
    220045: 'heart_rate',
    220179: 'sbp',
    220210: 'resp_rate',
    220277: 'spo2',
    223761: 'temperature',
}

print(f"Device: {DEVICE}")
print("="*62)
print("  Health Extension v2: Chunked MIMIC-IV Loading")
print("="*62)

def load_mimic_chunked(chart_csv, stays_csv, max_patients=5000):
    print("\n[STEP 1] Loading ICU stays...")
    stays = pd.read_csv(stays_csv)
    stays['intime']  = pd.to_datetime(stays['intime'])
    stays['outtime'] = pd.to_datetime(stays['outtime'])
    stays['los_hours'] = (stays['outtime']-stays['intime']).dt.total_seconds()/3600
    stays = stays[stays['los_hours'] >= SEQ_LEN]
    # Limit to max_patients for memory
    if len(stays) > max_patients:
        stays = stays.sample(max_patients, random_state=42)
    valid_stays = set(stays['stay_id'].values)
    print(f"  Valid ICU stays: {len(stays):,}")

    # Build stay lookup
    stay_info = stays.set_index('stay_id')[['intime','los_hours']].to_dict('index')

    print(f"\n[STEP 2] Reading chartevents in chunks...")
    # Collect vital signs per stay
    vitals_per_stay = {sid: [] for sid in valid_stays}
    chunk_size = 500_000
    total_rows = 0
    useful_rows = 0

    for chunk in pd.read_csv(chart_csv, chunksize=chunk_size,
                              low_memory=False):
        total_rows += len(chunk)
        # Filter to vital items and valid stays
        chunk = chunk[chunk['itemid'].isin(VITAL_ITEMS.keys())]
        chunk = chunk[chunk['stay_id'].isin(valid_stays)]
        chunk = chunk.dropna(subset=['valuenum','charttime','stay_id'])
        chunk['vital'] = chunk['itemid'].map(VITAL_ITEMS)
        chunk['valuenum'] = pd.to_numeric(chunk['valuenum'], errors='coerce')
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        useful_rows += len(chunk)

        for sid, grp in chunk.groupby('stay_id'):
            vitals_per_stay[sid].append(grp[['charttime','vital','valuenum']])

        if total_rows % 5_000_000 == 0:
            print(f"  Read {total_rows/1e6:.0f}M rows, useful: {useful_rows:,}")

    print(f"  Total read: {total_rows/1e6:.1f}M rows, useful: {useful_rows:,}")

    # Build sequences
    print("\n[STEP 3] Building sequences...")
    vitals_list = ['heart_rate','sbp','resp_rate','spo2','temperature']
    sequences=[]; labels=[]; stay_ids=[]

    for sid, dfs in vitals_per_stay.items():
        if not dfs: continue
        info = stay_info[sid]
        los  = info['los_hours']
        intime = info['intime']

        pt = pd.concat(dfs, ignore_index=True)
        pt['hour'] = (pt['charttime']-intime).dt.total_seconds()/3600
        pt = pt[(pt['hour']>=0)&(pt['hour']<=los)]
        pt['hour_bin'] = pt['hour'].astype(int)

        pivot = pt.groupby(['hour_bin','vital'])['valuenum'].median().unstack()
        for v in vitals_list:
            if v not in pivot.columns: pivot[v]=np.nan
        pivot = pivot[vitals_list].ffill().bfill().dropna()

        if len(pivot) < SEQ_LEN: continue
        vals = pivot.values.astype(np.float32)

        for end in range(SEQ_LEN, min(len(vals)+1, int(los)+1)):
            seq = vals[end-SEQ_LEN:end]
            if seq.shape[0] != SEQ_LEN: continue
            rul = min(los-end, RUL_CLIP*24)/24
            sequences.append(seq); labels.append(rul); stay_ids.append(sid)

    if not sequences:
        print("No sequences found!")
        return None, None, None

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.float32)
    ids = np.array(stay_ids)
    print(f"  Sequences: {len(X):,}, shape={X.shape}")
    print(f"  RUL range: {y.min():.1f} - {y.max():.1f} days")
    return X, y, ids

# Model classes
class PatchEmbedding(nn.Module):
    def __init__(self,nf,patch_len,d_model,dropout=0.1):
        super().__init__()
        self.patch_len=patch_len
        self.norm=nn.LayerNorm(nf*patch_len)
        self.proj=nn.Linear(nf*patch_len,d_model)
        self.drop=nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight,gain=0.3)
        nn.init.zeros_(self.proj.bias)
    def forward(self,x):
        B,L,C=x.shape
        x=x.reshape(B,N_PATCHES,self.patch_len*C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_Health(nn.Module):
    def __init__(self,nf=NF_HEALTH,d_model=128,n_heads=4,n_layers=4,d_ff=256,dropout=0.1):
        super().__init__()
        self.nf=nf; self.d_model=d_model
        self.patch_embed=PatchEmbedding(nf,PATCH_LEN,d_model,dropout)
        self.pos=nn.Parameter(torch.randn(1,N_PATCHES,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,d_ff,dropout,
                                        batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,n_layers)
        self.rul_head=None
    def encode(self,x):
        return self.transformer(self.patch_embed(x)+self.pos).mean(dim=1)
    def add_head(self):
        self.rul_head=nn.Sequential(
            nn.Linear(self.d_model,64),nn.GELU(),
            nn.Dropout(0.1),nn.Linear(64,1))
    def forward_rul(self,x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMBaseline(nn.Module):
    def __init__(self,nf=NF_HEALTH,hidden=64):
        super().__init__()
        self.lstm=nn.LSTM(nf,hidden,2,batch_first=True,dropout=0.2)
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        o,_=self.lstm(x); return self.head(o[:,-1,:]).squeeze(1)

def load_backbone(model):
    state=torch.load(BACKBONE,map_location=DEVICE,weights_only=False)
    model_state=model.state_dict()
    transferable={k:v for k,v in state.items()
                  if k in model_state and model_state[k].shape==v.shape}
    model_state.update(transferable)
    model.load_state_dict(model_state)
    print(f"  Transferred {len(transferable)} layers from industrial backbone")
    return model

def train_model(model,X,y,epochs=30,batch=64,lr=5e-4,is_lstm=False):
    model.to(DEVICE)
    if not is_lstm: model.add_head()
    model.to(DEVICE)
    loader=DataLoader(TensorDataset(
        torch.tensor(np.clip(X,-4,4),dtype=torch.float32),
        torch.tensor(y,dtype=torch.float32)),
        batch_size=batch,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    lf=nn.MSELoss(); model.train()
    for ep in range(epochs):
        total=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            pred=model(xb) if is_lstm else model.forward_rul(xb)
            loss=lf(pred,yb)
            if not torch.isnan(loss): loss.backward(); opt.step()
            total+=loss.item()
        sched.step()
        if (ep+1)%10==0:
            print(f"    ep {ep+1}/{epochs} loss={total/len(loader):.3f}")
    return model

def predict(model,X,is_lstm=False,batch=256):
    model.eval(); preds=[]
    Xt=torch.tensor(np.clip(X,-4,4),dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(Xt),batch):
            xb=Xt[i:i+batch].to(DEVICE)
            p=(model(xb) if is_lstm else model.forward_rul(xb)).cpu().numpy()
            preds.append(np.nan_to_num(p,nan=3.0))
    return np.clip(np.concatenate(preds),0,RUL_CLIP)

def conf_q(r,cov=0.90):
    n=len(r); return float(np.quantile(r,min(np.ceil((n+1)*cov)/n,1.0)))

def main():
    # Load data with chunking — limit to 5000 patients for speed
    X,y,ids = load_mimic_chunked(CHART_CSV, STAYS_CSV, max_patients=5000)
    if X is None: return

    # Normalize
    sc=StandardScaler()
    B,L,F=X.shape
    X=sc.fit_transform(X.reshape(-1,F)).reshape(B,L,F).astype(np.float32)

    # Split
    unique_ids=np.unique(ids); np.random.seed(42); np.random.shuffle(unique_ids)
    n=len(unique_ids)
    n_test=max(5,int(n*0.15)); n_cal=max(5,int(n*0.15))
    test_set=set(unique_ids[:n_test]); cal_set=set(unique_ids[n_test:n_test+n_cal])
    mtr=np.array([i not in test_set and i not in cal_set for i in ids])
    mca=np.array([i in cal_set for i in ids])
    mte=np.array([i in test_set for i in ids])
    Xtr,ytr=X[mtr],y[mtr]; Xca,yca=X[mca],y[mca]; Xte,yte=X[mte],y[mte]
    print(f"\n  Train:{len(Xtr):,} Cal:{len(Xca):,} Test:{len(Xte):,}")

    results={}

    # A: FM transfer
    print("\n(A) PatchTST with Industrial Backbone Transfer...")
    fm=PatchTST_Health().to(DEVICE); fm=load_backbone(fm)
    fm=train_model(fm,Xtr,ytr,epochs=30)
    p_fm=predict(fm,Xte); cp_fm=predict(fm,Xca)
    q_fm=conf_q(np.abs(yca-cp_fm))
    mae_fm=mean_absolute_error(yte,p_fm)
    cov_fm=float(((yte>=p_fm-q_fm)&(yte<=p_fm+q_fm)).mean())
    print(f"  MAE={mae_fm:.3f} days  Coverage={cov_fm:.0%}")
    results['FM_transfer']={'MAE':round(mae_fm,3),'cov':f"{cov_fm:.0%}"}

    # B: Scratch
    print("\n(B) PatchTST scratch...")
    sc_m=PatchTST_Health().to(DEVICE)
    sc_m=train_model(sc_m,Xtr,ytr,epochs=30)
    p_sc=predict(sc_m,Xte); cp_sc=predict(sc_m,Xca)
    q_sc=conf_q(np.abs(yca-cp_sc))
    mae_sc=mean_absolute_error(yte,p_sc)
    cov_sc=float(((yte>=p_sc-q_sc)&(yte<=p_sc+q_sc)).mean())
    print(f"  MAE={mae_sc:.3f} days  Coverage={cov_sc:.0%}")
    results['Scratch']={'MAE':round(mae_sc,3),'cov':f"{cov_sc:.0%}"}

    # C: LSTM
    print("\n(C) LSTM baseline...")
    lstm=LSTMBaseline().to(DEVICE)
    lstm=train_model(lstm,Xtr,ytr,epochs=30,is_lstm=True)
    p_ls=predict(lstm,Xte,is_lstm=True); cp_ls=predict(lstm,Xca,is_lstm=True)
    q_ls=conf_q(np.abs(yca-cp_ls))
    mae_ls=mean_absolute_error(yte,p_ls)
    cov_ls=float(((yte>=p_ls-q_ls)&(yte<=p_ls+q_ls)).mean())
    print(f"  MAE={mae_ls:.3f} days  Coverage={cov_ls:.0%}")
    results['LSTM']={'MAE':round(mae_ls,3),'cov':f"{cov_ls:.0%}"}

    print(f"\n{'='*62}")
    print("  FULL MIMIC-IV RESULTS")
    print(f"{'='*62}")
    print(f"  {'Method':<28}{'MAE':>10}{'Coverage':>10}")
    print(f"  {'-'*50}")
    for name,r in results.items():
        tag=' <- transfer helps!' if name=='FM_transfer' and r['MAE']<results.get('Scratch',{}).get('MAE',999) else ''
        print(f"  {name:<28}{r['MAE']:>10}{r['cov']:>10}{tag}")
    print(f"{'='*62}")

    import json
    with open('health_extension_results.json','w') as f:
        json.dump(results,f,indent=2)
    print("  Saved: health_extension_results.json")

if __name__=="__main__":
    main()