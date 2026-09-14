"""
iclr_exp1_v2.py — ICLR Experiment 1: Limited Engine Data (FIXED)

Compares PatchTST FM vs PatchTST scratch vs LSTM
as number of training engines decreases: 10, 20, 40, 60, 80

Run: python iclr_exp1_v2.py
"""

import os, copy, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

DATA_DIR = r"E:\rul_project\data"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 30   # FIXED sequence length — everything must be exactly 30
PATCH_LEN= 5    # 30 / 5 = 6 patches always
N_PATCHES= SEQ_LEN // PATCH_LEN   # = 6
RUL_CLIP = 125
COLS     = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

print(f"Device: {DEVICE}")
print(f"SEQ_LEN={SEQ_LEN}, PATCH_LEN={PATCH_LEN}, N_PATCHES={N_PATCHES}")

# ── DATA ─────────────────────────────────────────────────────────────────────

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

def make_pretrain_seqs(X, engines, stride=10):
    """Build sequences of EXACTLY SEQ_LEN length."""
    seqs=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe=X[idx]
        for end in range(SEQ_LEN, len(Xe)+1, stride):
            seq=Xe[end-SEQ_LEN:end]
            assert seq.shape[0]==SEQ_LEN, f"Bad shape: {seq.shape}"
            seqs.append(seq)
    return np.array(seqs, dtype=np.float32)

def make_rul_seqs(X, y, engines):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe,ye=X[idx],y[idx]
        for end in range(SEQ_LEN, len(Xe)+1):
            seq=Xe[end-SEQ_LEN:end]
            assert seq.shape[0]==SEQ_LEN
            seqs.append(seq); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32),np.array(labels,dtype=np.float32)

def last_wins(X, engines, cycles, nf):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]; Xe=X[idx]
        if len(Xe)>=SEQ_LEN:
            out.append(Xe[-SEQ_LEN:])
        else:
            pad=np.zeros((SEQ_LEN-len(Xe),nf),dtype=np.float32)
            out.append(np.vstack([pad,Xe]))
    arr=np.array(out,dtype=np.float32)
    assert arr.shape[1]==SEQ_LEN
    return arr

def rand_cut(X, y, engines, cycles, n_cuts=5):
    all_s,all_y=[],[]
    for seed in range(n_cuts):
        rng=np.random.RandomState(seed)
        for e in np.unique(engines):
            idx=np.where(engines==e)[0]
            idx=idx[np.argsort(cycles[engines==e])]
            Xe,ye=X[idx],y[idx]; n=len(Xe)
            if n<SEQ_LEN: continue
            end=rng.randint(SEQ_LEN,n)
            seq=Xe[end-SEQ_LEN:end]
            assert seq.shape[0]==SEQ_LEN
            all_s.append(seq); all_y.append(ye[end-1])
    return np.array(all_s,dtype=np.float32),np.array(all_y,dtype=np.float32)

# ── MODEL ─────────────────────────────────────────────────────────────────────

class PatchTST_RUL(nn.Module):
    def __init__(self, nf, d_model=128, n_heads=4, n_layers=4,
                 d_ff=256, dropout=0.1):
        super().__init__()
        # Fixed positional embedding for N_PATCHES patches
        self.nf=nf; self.d_model=d_model
        in_dim=nf*PATCH_LEN
        self.norm=nn.LayerNorm(in_dim)
        self.proj=nn.Linear(in_dim,d_model)
        self.drop=nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.proj.weight,gain=0.3)
        nn.init.zeros_(self.proj.bias)
        # Positional embedding: exactly N_PATCHES positions
        self.pos=nn.Parameter(torch.randn(1,N_PATCHES,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,d_ff,dropout,
                                        batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(enc,n_layers)
        self.pretrain_head=nn.Linear(d_model,in_dim)
        self.rul_head=None

    def _patch(self, x):
        # x: (B, SEQ_LEN, nf) → (B, N_PATCHES, d_model)
        B,L,C=x.shape
        assert L==SEQ_LEN, f"Expected {SEQ_LEN} got {L}"
        x=x.reshape(B, N_PATCHES, PATCH_LEN*C)
        return self.drop(self.proj(self.norm(x)))

    def encode(self, x):
        p=self._patch(x)+self.pos
        return self.transformer(p).mean(dim=1)

    def forward_pretrain(self, x):
        B=x.shape[0]
        p=self._patch(x)+self.pos          # (B, N_PATCHES, d_model)
        # Mask 15% of patches
        n_mask=max(1,int(N_PATCHES*0.15))
        masked=p.clone()
        midx=torch.stack([torch.randperm(N_PATCHES)[:n_mask]
                          for _ in range(B)])
        for b in range(B): masked[b,midx[b]]=0.
        enc=self.transformer(masked)
        pred=self.pretrain_head(enc)       # (B, N_PATCHES, PATCH_LEN*nf)
        # Target: original patches
        target=x.reshape(B,N_PATCHES,PATCH_LEN*self.nf)
        return pred,target,midx

    def add_head(self):
        self.rul_head=nn.Sequential(
            nn.Linear(self.d_model,64),nn.GELU(),
            nn.Dropout(0.1),nn.Linear(64,1))

    def forward_rul(self,x):
        return self.rul_head(self.encode(x)).squeeze(1)

class LSTMModel(nn.Module):
    def __init__(self,nf,hidden=64):
        super().__init__()
        self.lstm=nn.LSTM(nf,hidden,2,batch_first=True,dropout=0.2)
        self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        o,_=self.lstm(x); return self.head(o[:,-1,:]).squeeze(1)

# ── PRE-TRAINING ─────────────────────────────────────────────────────────────

def pretrain(model, seqs, epochs=50, batch=64, lr=2e-4):
    seqs=np.clip(seqs,-4,4)
    assert seqs.shape[1]==SEQ_LEN, f"Pre-train shape wrong: {seqs.shape}"
    loader=DataLoader(TensorDataset(torch.tensor(seqs,dtype=torch.float32)),
                      batch_size=batch,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(
        opt,max_lr=lr,steps_per_epoch=len(loader),epochs=epochs)
    criterion=nn.MSELoss(); model.train()
    print(f"  Pre-training: {len(seqs):,} seqs × {SEQ_LEN} cycles × "
          f"{seqs.shape[2]} features → {N_PATCHES} patches")
    for ep in range(epochs):
        total=0; nb=0
        for (x,) in loader:
            x=x.to(DEVICE)
            opt.zero_grad()
            pred,target,midx=model.forward_pretrain(x)
            # Loss on masked patches only
            loss=torch.tensor(0.,device=DEVICE)
            for b in range(x.shape[0]):
                for idx in midx[b]:
                    loss=loss+criterion(pred[b,idx],target[b,idx])
            loss=loss/x.shape[0]
            if torch.isnan(loss) or torch.isinf(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5)
            opt.step(); sched.step()
            total+=loss.item(); nb+=1
        if (ep+1)%10==0:
            print(f"    ep {ep+1}/{epochs} loss={total/max(nb,1):.4f}")
    return model

# ── FINE-TUNE ─────────────────────────────────────────────────────────────────

def fine_tune(model, Xs, ys, epochs=40, batch=64, lr=5e-4, is_lstm=False):
    assert Xs.shape[1]==SEQ_LEN
    model.to(DEVICE)
    if not is_lstm: model.add_head()
    model.to(DEVICE)
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
            pred=model(xb) if is_lstm else model.forward_rul(xb)
            loss=lf(pred,yb)
            if not torch.isnan(loss): loss.backward(); opt.step()
        sched.step()
    return model

def predict(model, X, is_lstm=False, batch=256):
    assert X.shape[1]==SEQ_LEN
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
def get_cov(y,pred,q): return float(((y>=pred-q)&(y<=pred+q)).mean())

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    NF=14

    # ── Pre-training corpus ────────────────────────────────────────────────────
    print("\n[STEP 1] Building pre-training corpus ...")
    all_pre=[]
    for fd in ["FD001","FD002","FD003","FD004"]:
        X,engines,_,sc,_=load_fd(fd,"train")
        seqs=make_pretrain_seqs(X,engines,stride=10)
        all_pre.append(seqs)
        print(f"  {fd}: {len(seqs):,} seqs, shape={seqs.shape}")
    pretrain_seqs=np.vstack(all_pre)
    print(f"  Total: {len(pretrain_seqs):,}, shape={pretrain_seqs.shape}")
    assert pretrain_seqs.shape[1]==SEQ_LEN

    # ── Pre-train backbone ─────────────────────────────────────────────────────
    print("\n[STEP 2] Pre-training PatchTST backbone ...")
    import os as _os
    _ckpt = r"D:\rul_project\iclr_backbone_v2.pt"
    backbone=PatchTST_RUL(NF).to(DEVICE)
    n_params=sum(p.numel() for p in backbone.parameters())
    print(f"  Parameters: {n_params:,}")
    if _os.path.exists(_ckpt):
        backbone.load_state_dict(torch.load(_ckpt,map_location=DEVICE))
        print("  Loaded existing backbone � skipping pre-training")
    else:
        backbone=pretrain(backbone,pretrain_seqs,epochs=50,batch=64,lr=2e-4)
        torch.save(backbone.state_dict(),_ckpt)
        print("  Saved: iclr_backbone_v2.pt")

    # ── Load FD001 ─────────────────────────────────────────────────────────────
    print("\n[STEP 3] Loading FD001 for limited-data experiment ...")
    X1,e1,c1,sc1,rul1=load_fd("FD001","train")
    Xt1,et1,ct1,_,_=load_fd("FD001","test")
    y_test=np.clip(pd.read_csv(
        os.path.join(DATA_DIR,"RUL_FD001.txt"),
        header=None).iloc[:,0].values,0,RUL_CLIP)

    all_e=np.unique(e1); np.random.seed(42); np.random.shuffle(all_e)
    cal_e=set(all_e[-20:])
    mca=np.isin(e1,list(cal_e))
    cal_seqs,cal_y=rand_cut(X1[mca],rul1[mca],e1[mca],c1[mca])
    te_seqs=last_wins(Xt1,et1,ct1,NF)
    train_pool=np.array([e for e in all_e if e not in cal_e])
    print(f"  Train pool: {len(train_pool)} engines  Cal: 20  Test: {len(y_test)}")

    # ── Run experiment ─────────────────────────────────────────────────────────
    print("\n[STEP 4] Running limited-data experiment ...")
    engine_counts=[10,20,40,60,80]
    results=[]

    for n_eng in engine_counts:
        print(f"\n  --- {n_eng} training engines ---")
        sel=train_pool[:n_eng]
        mtr=np.isin(e1,sel)
        Xs,ys=make_rul_seqs(X1[mtr],rul1[mtr],e1[mtr])
        print(f"  Train seqs: {len(Xs):,}")

        # (A) PatchTST FM
        print("  (A) PatchTST FM ...")
        fm=copy.deepcopy(backbone)
        fm=fine_tune(fm,Xs,ys,epochs=40,lr=5e-4)
        p_fm=predict(fm,te_seqs)
        cp_fm=predict(fm,cal_seqs)
        q_fm=conf_q(np.abs(cal_y-cp_fm))
        mae_fm=mean_absolute_error(y_test,p_fm)
        cov_fm=get_cov(y_test,p_fm,q_fm)
        print(f"    MAE={mae_fm:.2f}  Cov={cov_fm:.0%}")

        # (B) PatchTST scratch
        print("  (B) PatchTST scratch ...")
        sc_m=PatchTST_RUL(NF).to(DEVICE)
        sc_m=fine_tune(sc_m,Xs,ys,epochs=40,lr=5e-4)
        p_sc=predict(sc_m,te_seqs)
        cp_sc=predict(sc_m,cal_seqs)
        q_sc=conf_q(np.abs(cal_y-cp_sc))
        mae_sc=mean_absolute_error(y_test,p_sc)
        cov_sc=get_cov(y_test,p_sc,q_sc)
        print(f"    MAE={mae_sc:.2f}  Cov={cov_sc:.0%}")

        # (C) LSTM
        print("  (C) LSTM ...")
        lstm=LSTMModel(NF).to(DEVICE)
        lstm=fine_tune(lstm,Xs,ys,epochs=40,lr=1e-3,is_lstm=True)
        p_ls=predict(lstm,te_seqs,is_lstm=True)
        cp_ls=predict(lstm,cal_seqs,is_lstm=True)
        q_ls=conf_q(np.abs(cal_y-cp_ls))
        mae_ls=mean_absolute_error(y_test,p_ls)
        cov_ls=get_cov(y_test,p_ls,q_ls)
        print(f"    MAE={mae_ls:.2f}  Cov={cov_ls:.0%}")

        results.append({'n':n_eng,
            'FM_MAE':round(mae_fm,2),'FM_cov':f"{cov_fm:.0%}",
            'Sc_MAE':round(mae_sc,2),'Sc_cov':f"{cov_sc:.0%}",
            'LS_MAE':round(mae_ls,2),'LS_cov':f"{cov_ls:.0%}"})

    # ── FINAL TABLE ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print("  ICLR EXP 1 — Does pre-training help under limited engine data?")
    print(f"{'='*72}")
    print(f"  {'N':>4}  {'FM MAE':>8}{'FM Cov':>8}"
          f"  {'Scratch MAE':>12}{'Scratch Cov':>12}"
          f"  {'LSTM MAE':>10}{'LSTM Cov':>10}")
    print(f"  {'-'*68}")
    for r in results:
        print(f"  {r['n']:>4}  {r['FM_MAE']:>8}{r['FM_cov']:>8}"
              f"  {r['Sc_MAE']:>12}{r['Sc_cov']:>12}"
              f"  {r['LS_MAE']:>10}{r['LS_cov']:>10}")
    print(f"{'='*72}")
    print("\n  KEY: FM MAE should stay lower than Scratch at small N engines.")
    print("  If FM degrades slower → pre-training helps with limited data.")

    with open(r"D:\rul_project\iclr_exp1_results.json","w") as f:
        json.dump(results,f,indent=2)
    print("  Saved: iclr_exp1_results.json")

if __name__=="__main__":
    main()
