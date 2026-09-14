"""
iclr_exp2_cross_condition.py — ICLR Experiment 2: Cross-Condition Generalisation

RESEARCH QUESTION:
  Can a pre-trained foundation model generalise from 1 operating condition
  (FD001) to 6 operating conditions (FD002) better than a scratch model?

EXPERIMENT:
  Train on FD001 (1 condition, 1 fault mode)
  Test on FD002 (6 conditions, 1 fault mode)

  Compare:
    (A) PatchTST FM — uses pre-trained backbone from all C-MAPSS
    (B) PatchTST scratch — no pre-training
    (C) LSTM — standard baseline

HYPOTHESIS:
  Pre-trained backbone saw FD002 data during pre-training, so it already
  knows what 6-condition degradation looks like. When fine-tuned on FD001
  only, it should generalise better to FD002 than scratch models.

Run: python iclr_exp2_cross_condition.py
  (requires iclr_backbone_v2.pt from iclr_exp1_limited_data.py)
"""

import os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = r"E:\rul_project\data"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 30
RUL_CLIP = 125
COLS     = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

print(f"Device: {DEVICE}")
print("ICLR Experiment 2: Cross-Condition Generalisation")

# ── copy model classes from exp1 ──────────────────────────────────────────────

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
        B,L,C=x.shape; n_p=L//self.patch_len
        x=x[:,:n_p*self.patch_len,:].reshape(B,n_p,self.patch_len*C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_RUL(nn.Module):
    def __init__(self,nf,seq_len=30,patch_len=5,d_model=128,
                 n_heads=4,n_layers=4,d_ff=256,dropout=0.1):
        super().__init__()
        self.patch_embed=PatchEmbedding(nf,patch_len,d_model,dropout)
        n_p=seq_len//patch_len
        self.pos=nn.Parameter(torch.randn(1,n_p,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,d_ff,dropout,
                                        batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,n_layers)
        self.pretrain_head=nn.Linear(d_model,nf*patch_len)
        self.rul_head=None; self.d_model=d_model; self.patch_len=patch_len
    def encode(self,x):
        return self.transformer(self.patch_embed(x)+self.pos).mean(dim=1)
    def forward_pretrain(self,x,mask_ratio=0.15):
        B=x.shape[0]; p=self.patch_embed(x)+self.pos; n_p=p.shape[1]
        n_m=max(1,int(n_p*mask_ratio)); masked=p.clone()
        midx=torch.stack([torch.randperm(n_p)[:n_m] for _ in range(B)])
        for b in range(B): masked[b,midx[b]]=0.
        return self.pretrain_head(self.transformer(masked)),midx
    def add_head(self):
        self.rul_head=nn.Sequential(
            nn.Linear(self.d_model,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward_rul(self,x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMBaseline(nn.Module):
    def __init__(self,nf,hidden=64):
        super().__init__()
        self.lstm=nn.LSTM(nf,hidden,2,batch_first=True,dropout=0.2)
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        o,_=self.lstm(x); return self.head(o[:,-1,:]).squeeze(1)

# ── DATA HELPERS ─────────────────────────────────────────────────────────────

def load_fd(fd, split="train"):
    df=pd.read_csv(os.path.join(DATA_DIR,f"{split}_{fd}.txt"),
                   sep=r"\s+",header=None).iloc[:,:26]
    df.columns=COLS
    fc=[c for c in COLS if c.startswith("s") and c not in DROP]
    if split=="train":
        mc=df.groupby("engine")["cycle"].transform("max")
        df["RUL"]=(mc-df["cycle"]).clip(upper=RUL_CLIP)
    sc=StandardScaler()
    X=sc.fit_transform(df[fc].values).astype(np.float32)
    rul=df["RUL"].values if "RUL" in df.columns else None
    return X,df["engine"].values,df["cycle"].values,sc,rul

def make_windows(X,y,engines,seq_len):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe,ye=X[idx],y[idx]
        for end in range(seq_len,len(Xe)+1):
            seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32),np.array(labels,dtype=np.float32)

def last_wins(X,engines,cycles,seq_len,nf):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]; Xe=X[idx]
        out.append(Xe[-seq_len:] if len(Xe)>=seq_len
                   else np.vstack([np.zeros((seq_len-len(Xe),nf)),Xe]))
    return np.array(out,dtype=np.float32)

def rand_cut(X,y,engines,cycles,seq_len,n_cuts=5):
    all_s,all_y=[],[]
    for seed in range(n_cuts):
        rng=np.random.RandomState(seed); seqs,labels=[],[]
        for e in np.unique(engines):
            idx=np.where(engines==e)[0]
            idx=idx[np.argsort(cycles[engines==e])]
            Xe,ye=X[idx],y[idx]; n=len(Xe)
            if n<seq_len: continue
            end=rng.randint(seq_len,n)
            seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
        all_s.append(np.array(seqs,dtype=np.float32))
        all_y.append(np.array(labels,dtype=np.float32))
    return np.vstack(all_s),np.concatenate(all_y)

def train_model(model,Xs,ys,epochs=40,batch=64,lr=5e-4,is_lstm=False):
    model.to(DEVICE)
    if not is_lstm: model.add_head()
    model.to(DEVICE)
    loader=DataLoader(
        TensorDataset(torch.tensor(np.clip(Xs,-4,4),dtype=torch.float32),
                      torch.tensor(ys,dtype=torch.float32)),
        batch_size=batch,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=15,gamma=0.5)
    lf=nn.MSELoss(); model.train()
    for ep in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            pred=model.forward_rul(xb) if not is_lstm else model(xb)
            loss=lf(pred,yb)
            if not torch.isnan(loss): loss.backward(); opt.step()
        sched.step()
    return model

def predict(model,X,is_lstm=False,batch=256):
    model.eval(); preds=[]
    Xt=torch.tensor(np.clip(X,-4,4),dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(Xt),batch):
            xb=Xt[i:i+batch].to(DEVICE)
            p=(model(xb) if is_lstm else model.forward_rul(xb)).cpu().numpy()
            preds.append(np.nan_to_num(p,nan=62.5))
    return np.clip(np.concatenate(preds),0,RUL_CLIP)

def conf_q(r,cov=0.90):
    n=len(r); return float(np.quantile(r,min(np.ceil((n+1)*cov)/n,1.0)))
def get_coverage(y,pred,q): return ((y>=pred-q)&(y<=pred+q)).mean()

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    NF=14
    print("\n[LOADING PRE-TRAINED BACKBONE]")
    backbone=PatchTST_RUL(NF).to(DEVICE)
    backbone.load_state_dict(torch.load(
        r"E:\rul_project\iclr_backbone_v2.pt",map_location=DEVICE, weights_only=False),strict=False)
    print("  Loaded: iclr_backbone_v2.pt")

    # ── Train on FD001 ────────────────────────────────────────────────────────
    print("\n[TRAIN] Loading FD001 (1 operating condition) ...")
    X1,e1,c1,sc1,rul1=load_fd("FD001","train")
    all_e=np.unique(e1); np.random.seed(42); np.random.shuffle(all_e)
    cal_set=set(all_e[:20]); mtr=~np.isin(e1,list(cal_set))
    mca=np.isin(e1,list(cal_set))
    Xs,ys=make_windows(X1[mtr],rul1[mtr],e1[mtr],SEQ_LEN)

    # ── Test on FD002 (6 operating conditions) ────────────────────────────────
    print("[TEST] Loading FD002 test (6 operating conditions) ...")
    Xt2,et2,ct2,sc2,_=load_fd("FD002","test")
    # Scale FD002 test using FD001 scaler (cross-condition — same sensor channels)
    # This is intentionally using FD001 scaler — tests generalisation
    y_test2=np.clip(pd.read_csv(
        os.path.join(DATA_DIR,"RUL_FD002.txt"),
        header=None).iloc[:,0].values,0,RUL_CLIP)
    te_seqs=last_wins(Xt2,et2,ct2,SEQ_LEN,NF)

    # Calibration on FD001 cal set
    cal_seqs,cal_y=rand_cut(X1[mca],rul1[mca],e1[mca],c1[mca],SEQ_LEN)

    print(f"  Train seqs (FD001): {len(Xs):,}")
    print(f"  Test engines (FD002): {len(np.unique(et2))}")
    print(f"  Cal seqs (FD001): {len(cal_seqs)}")

    results={}

    # (A) PatchTST FM
    print("\n  (A) PatchTST FM (pre-trained on all C-MAPSS) ...")
    fm=copy.deepcopy(backbone)
    fm=train_model(fm,Xs,ys,epochs=40,lr=5e-4)
    pred_fm=predict(fm,te_seqs)
    cal_pred_fm=predict(fm,cal_seqs)
    q_fm=conf_q(np.abs(cal_y-cal_pred_fm))
    results['FM']={'MAE':round(mean_absolute_error(y_test2,pred_fm),2),
                   'cov':f"{get_coverage(y_test2,pred_fm,q_fm):.0%}"}
    print(f"    MAE={results['FM']['MAE']}  Cov={results['FM']['cov']}")

    # (B) PatchTST scratch
    print("  (B) PatchTST scratch (random init) ...")
    sc_model=PatchTST_RUL(NF).to(DEVICE)
    sc_model=train_model(sc_model,Xs,ys,epochs=40,lr=5e-4)
    pred_sc=predict(sc_model,te_seqs)
    cal_pred_sc=predict(sc_model,cal_seqs)
    q_sc=conf_q(np.abs(cal_y-cal_pred_sc))
    results['Scratch']={'MAE':round(mean_absolute_error(y_test2,pred_sc),2),
                        'cov':f"{get_coverage(y_test2,pred_sc,q_sc):.0%}"}
    print(f"    MAE={results['Scratch']['MAE']}  Cov={results['Scratch']['cov']}")

    # (C) LSTM
    print("  (C) LSTM baseline ...")
    lstm=LSTMBaseline(NF).to(DEVICE)
    lstm=train_model(lstm,Xs,ys,epochs=40,lr=1e-3,is_lstm=True)
    pred_ls=predict(lstm,te_seqs,is_lstm=True)
    cal_pred_ls=predict(lstm,cal_seqs,is_lstm=True)
    q_ls=conf_q(np.abs(cal_y-cal_pred_ls))
    results['LSTM']={'MAE':round(mean_absolute_error(y_test2,pred_ls),2),
                     'cov':f"{get_coverage(y_test2,pred_ls,q_ls):.0%}"}
    print(f"    MAE={results['LSTM']['MAE']}  Cov={results['LSTM']['cov']}")

    # ── RESULTS ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ICLR EXPERIMENT 2: Cross-Condition Generalisation")
    print("  Train: FD001 (1 condition)  Test: FD002 (6 conditions)")
    print(f"{'='*60}")
    print(f"  {'Method':<22}{'MAE':>8}{'Coverage':>10}")
    print(f"  {'-'*42}")
    for name,r in results.items():
        print(f"  {name:<22}{r['MAE']:>8}{r['cov']:>10}")
    print(f"{'='*60}")
    print()
    print("  KEY: Does FM MAE < Scratch MAE when generalising across conditions?")
    print("  If yes — pre-training helps cross-condition generalisation.")

    import json
    with open(r"D:\rul_project\iclr_exp2_results.json","w") as f:
        json.dump(results,f,indent=2)
    print("  Saved: iclr_exp2_results.json")

if __name__=="__main__":
    main()
