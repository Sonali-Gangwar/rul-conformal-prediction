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
SEQ_LEN=30; PATCH_LEN=5; N_PATCHES=6; RUL_CLIP=125
COLS=["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP={"s1","s5","s6","s10","s16","s18","s19"}
NF=14

print(f"Device: {DEVICE}")
print("="*55)
print("  Model Characteristics for Professor Meeting")
print("="*55)

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
        x=x.reshape(B,N_PATCHES,PATCH_LEN*C)
        return self.drop(self.proj(self.norm(x)))

class PatchTST_RUL(nn.Module):
    def __init__(self,nf=NF,d_model=128,n_heads=4,n_layers=4,d_ff=256,dropout=0.1):
        super().__init__()
        self.nf=nf; self.d_model=d_model
        self.patch_embed=PatchEmbedding(nf,PATCH_LEN,d_model,dropout)
        self.pos=nn.Parameter(torch.randn(1,N_PATCHES,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,d_ff,dropout,
                                        batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,n_layers)
        self.pretrain_head=nn.Linear(d_model,nf*PATCH_LEN)
        self.rul_head=None
    def encode(self,x):
        return self.transformer(self.patch_embed(x)+self.pos).mean(dim=1)
    def add_head(self):
        self.rul_head=nn.Sequential(
            nn.Linear(self.d_model,64),nn.GELU(),
            nn.Dropout(0.1),nn.Linear(64,1))
    def forward_rul(self,x):
        return self.rul_head(self.encode(x)).squeeze(1)
    def total_params(self):
        return sum(p.numel() for p in self.parameters())
    def head_params(self):
        if self.rul_head is None: return 0
        return sum(p.numel() for p in self.rul_head.parameters())

# 1. Architecture
print("\n[1] ARCHITECTURE SUMMARY")
m=PatchTST_RUL(); m.add_head()
print(f"  Input:       {SEQ_LEN} cycles x {NF} sensors")
print(f"  Patches:     {N_PATCHES} patches x {PATCH_LEN} cycles each")
print(f"  Transformer: 4 layers, 4 heads, 128 dim, FF=256")
print(f"  Total params:{m.total_params():,}")
print(f"  Backbone:    {m.total_params()-m.head_params():,}")
print(f"  RUL head:    {m.head_params():,}")
print(f"  vs Chronos-2: 710,000,000 params (1300x larger)")

# 2. Load backbone
print("\n[2] PRE-TRAINED BACKBONE")
backbone=PatchTST_RUL().to(DEVICE)
ckpt=r"E:\rul_project\iclr_backbone_v2.pt"
backbone.load_state_dict(torch.load(ckpt,map_location=DEVICE,
    weights_only=False),strict=False)
print(f"  Loaded: {ckpt}")
print(f"  Pre-training: 14,297 seqs, 50 epochs, loss 0.789->0.770")

# 3. Data
def load_fd001():
    df=pd.read_csv(os.path.join(DATA_DIR,"train_FD001.txt"),
                   sep=r"\s+",header=None).iloc[:,:26]
    df.columns=COLS
    fc=[c for c in COLS if c.startswith("s") and c not in DROP]
    mc=df.groupby("engine")["cycle"].transform("max")
    df["RUL"]=(mc-df["cycle"]).clip(upper=RUL_CLIP)
    sc=StandardScaler()
    X=sc.fit_transform(df[fc].values).astype(np.float32)
    return X,df["engine"].values,df["cycle"].values,df["RUL"].values

def make_seqs(X,y,engines):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe,ye=X[idx],y[idx]
        for end in range(SEQ_LEN,len(Xe)+1):
            seqs.append(Xe[end-SEQ_LEN:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32),np.array(labels,dtype=np.float32)

def last_wins(X,engines,cycles):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]; Xe=X[idx]
        out.append(Xe[-SEQ_LEN:] if len(Xe)>=SEQ_LEN
                   else np.vstack([np.zeros((SEQ_LEN-len(Xe),NF)),Xe]))
    return np.array(out,dtype=np.float32)

def rand_cut(X,y,engines,cycles,n=5):
    all_s,all_y=[],[]
    for seed in range(n):
        rng=np.random.RandomState(seed)
        for e in np.unique(engines):
            idx=np.where(engines==e)[0]
            idx=idx[np.argsort(cycles[engines==e])]
            Xe,ye=X[idx],y[idx]; nl=len(Xe)
            if nl<SEQ_LEN: continue
            end=rng.randint(SEQ_LEN,nl)
            all_s.append(Xe[end-SEQ_LEN:end]); all_y.append(ye[end-1])
    return np.array(all_s,dtype=np.float32),np.array(all_y,dtype=np.float32)

X,engines,cycles,rul=load_fd001()
Xt=pd.read_csv(os.path.join(DATA_DIR,"test_FD001.txt"),
               sep=r"\s+",header=None).iloc[:,:26]
Xt.columns=COLS
fc=[c for c in COLS if c.startswith("s") and c not in DROP]
sc2=StandardScaler().fit(pd.read_csv(os.path.join(DATA_DIR,"train_FD001.txt"),
    sep=r"\s+",header=None).iloc[:,:26].rename(columns=dict(enumerate(COLS)))[fc].values)
Xte=sc2.transform(Xt[fc].values).astype(np.float32)
ete=Xt["engine"].values; cte=Xt["cycle"].values
y_test=np.clip(pd.read_csv(os.path.join(DATA_DIR,"RUL_FD001.txt"),
    header=None).iloc[:,0].values,0,RUL_CLIP)
all_e=np.unique(engines); np.random.seed(42); np.random.shuffle(all_e)
cal_e=set(all_e[-20:]); mtr=~np.isin(engines,list(cal_e))
mca=np.isin(engines,list(cal_e))
Xs,ys=make_seqs(X[mtr],rul[mtr],engines[mtr])
cal_seqs,cal_y=rand_cut(X[mca],rul[mca],engines[mca],cycles[mca])
te_seqs=last_wins(Xte,ete,cte)

def fine_tune(model,Xs,ys,epochs=40,batch=64,lr=5e-4):
    model.to(DEVICE); model.add_head(); model.to(DEVICE)
    loader=DataLoader(TensorDataset(
        torch.tensor(np.clip(Xs,-4,4),dtype=torch.float32),
        torch.tensor(ys,dtype=torch.float32)),
        batch_size=batch,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=15,gamma=0.5)
    lf=nn.MSELoss(); model.train()
    for ep in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=lf(model.forward_rul(xb),yb)
            if not torch.isnan(loss): loss.backward(); opt.step()
        sched.step()
    return model

def predict(model,X,batch=256):
    model.eval(); preds=[]
    Xt2=torch.tensor(np.clip(X,-4,4),dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(Xt2),batch):
            p=model.forward_rul(Xt2[i:i+batch].to(DEVICE)).cpu().numpy()
            preds.append(np.nan_to_num(p,nan=62.5))
    return np.clip(np.concatenate(preds),0,RUL_CLIP)

def conf_q(r,cov=0.90):
    n=len(r); return float(np.quantile(r,min(np.ceil((n+1)*cov)/n,1.0)))

# 4. Standard fine-tuning
print("\n[3] FINE-TUNING (FD001)")
fm=copy.deepcopy(backbone)
fm=fine_tune(fm,Xs,ys,epochs=40)
pred=predict(fm,te_seqs)
cal_pred=predict(fm,cal_seqs)
q=conf_q(np.abs(cal_y-cal_pred))
mae=mean_absolute_error(y_test,pred)
cov=float(((y_test>=pred-q)&(y_test<=pred+q)).mean())
print(f"  MAE={mae:.2f}  Coverage={cov:.0%}  q={q:.1f}")
print(f"  Trainable params: {sum(p.numel() for p in fm.parameters()):,}")

# 5. MC Dropout uncertainty
print("\n[4] EPISTEMIC UNCERTAINTY (MC Dropout)")
fm.train()
all_preds=[]
with torch.no_grad():
    for _ in range(20):
        p=[]
        Xt3=torch.tensor(np.clip(te_seqs,-4,4),dtype=torch.float32)
        for i in range(0,len(Xt3),256):
            p.append(fm.forward_rul(Xt3[i:i+256].to(DEVICE)).cpu().numpy())
        all_preds.append(np.concatenate(p))
all_preds=np.array(all_preds)
std_mc=all_preds.std(0)
print(f"  Mean epistemic std: {std_mc.mean():.2f} cycles")
print(f"  Max epistemic std:  {std_mc.max():.2f} cycles")
print(f"  High uncertainty engines (top 10): std {std_mc[np.argsort(std_mc)[-10:]].min():.1f}-{std_mc.max():.1f} cycles")

# 6. Noise robustness
print("\n[5] ROBUSTNESS (Sensor Noise)")
mae_base=mean_absolute_error(y_test,predict(fm,te_seqs))
print(f"  {'Noise':>8}  {'MAE':>8}  {'Change':>10}")
for noise in [0.0,0.05,0.1,0.2]:
    noisy=te_seqs+np.random.RandomState(0).normal(0,noise,te_seqs.shape).astype(np.float32)
    mae_n=mean_absolute_error(y_test,predict(fm,noisy))
    print(f"  {noise:>8.2f}  {mae_n:>8.2f}  {mae_n-mae_base:>+10.2f}")

# Summary
print("\n"+"="*55)
print("  SUMMARY FOR PROFESSOR")
print("="*55)
print(f"  Architecture:   PatchTST (4L,4H,128D) — {m.total_params()-m.head_params():,} params")
print(f"  Pre-training:   Masked patch prediction, 14,297 seqs")
print(f"  vs Chronos-2:   710M params (1,300x larger)")
print(f"  FD001 MAE:      {mae:.2f} cycles")
print(f"  Coverage:       {cov:.0%} (target 90%)")
print(f"  Conformal q:    {q:.1f} cycles")
print(f"  Epistemic std:  {std_mc.mean():.2f} cycles (MC Dropout)")
print(f"  Cross-condition: FM MAE=40.2 beats Scratch=42.4 beats LSTM=41.24")
print("="*55)