"""
proposal2_v3.py — Proposal 2: Stable Domain-Specific Foundation Model

ARCHITECTURE CHANGE from v2:
  Instead of full Transformer (unstable with small degradation data),
  use a CNN + GRU encoder which is:
  - More stable on small datasets
  - Faster to train
  - Still learns temporal degradation patterns
  - Still domain-specific (trained on degradation data only)

PRE-TRAINING: Masked channel prediction
  - Mask 1-2 sensor channels completely
  - Predict them from the other channels
  - Model learns inter-sensor relationships in degradation

This is still a foundation model — trained on multi-dataset degradation
data, fine-tuned per dataset with conformal calibration on top.

Run: python proposal2_v3.py
"""

import os, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN  = 30
D_MODEL  = 128
RUL_CLIP = 125
COVERAGE = 0.90
DATA_DIR = r"D:\rul_project\data"
CMAPSS_COLS = ["engine","cycle","op1","op2","op3"] + \
              [f"s{i}" for i in range(1,22)]
CMAPSS_DROP = {"s1","s5","s6","s10","s16","s18","s19"}
print(f"Device: {DEVICE}")

# ── DATA ─────────────────────────────────────────────────────────────────────

def load_cmapss(fd, split="train"):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{split}_{fd}.txt"),
                     sep=r"\s+", header=None).iloc[:,:26]
    df.columns = CMAPSS_COLS
    fc = [c for c in CMAPSS_COLS if c.startswith("s") and c not in CMAPSS_DROP]
    if split=="train":
        mc = df.groupby("engine")["cycle"].transform("max")
        df["RUL"] = (mc-df["cycle"]).clip(upper=RUL_CLIP)
    sc = StandardScaler()
    X  = sc.fit_transform(df[fc].values).astype(np.float32)
    rul= df["RUL"].values if "RUL" in df.columns else None
    return X, df["engine"].values, df["cycle"].values, sc, fc, rul

def make_windows(X, engines, seq_len, stride=5):
    seqs=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe=X[idx]
        for end in range(seq_len, len(Xe)+1, stride):
            seqs.append(Xe[end-seq_len:end])
    return np.array(seqs, dtype=np.float32)

def make_rul_windows(X, y, engines, seq_len):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe,ye=X[idx],y[idx]
        for end in range(seq_len, len(Xe)+1):
            seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32), np.array(labels,dtype=np.float32)

def load_femto(seq_len=SEQ_LEN):
    ld=os.path.join(DATA_DIR,"Learning_set"); all_s=[]
    for b in sorted(os.listdir(ld)):
        p=os.path.join(ld,b)
        if not os.path.isdir(p): continue
        rms=[]
        for f in sorted([x for x in os.listdir(p) if x.endswith('.csv')]):
            try:
                df=pd.read_csv(os.path.join(p,f),header=None)
                h=df.iloc[:,4].values.astype(float)
                v=df.iloc[:,5].values.astype(float)
                rms.append([np.sqrt(np.mean(h**2)),np.sqrt(np.mean(v**2))])
            except: continue
        if len(rms)<seq_len: continue
        rms=StandardScaler().fit_transform(np.array(rms,dtype=np.float32))
        for end in range(seq_len,len(rms)+1,2): all_s.append(rms[end-seq_len:end])
    return np.array(all_s,dtype=np.float32) if all_s else None

def load_ims(seq_len=SEQ_LEN):
    folder=os.path.join(DATA_DIR,"IMS","2nd_test","2nd_test")
    rms=[]
    for f in sorted(os.listdir(folder)):
        try:
            d=np.fromstring(open(os.path.join(folder,f)).read(),
                            sep='\t').reshape(-1,8)
            rms.append(np.sqrt(np.mean(d**2,axis=0)))
        except: continue
    if len(rms)<seq_len: return None
    rms=StandardScaler().fit_transform(np.array(rms,dtype=np.float32))
    seqs=[]
    for end in range(seq_len,len(rms)+1,2): seqs.append(rms[end-seq_len:end])
    return np.array(seqs,dtype=np.float32)

def pad_to(seqs, ref_nf):
    if seqs.shape[2]==ref_nf: return seqs
    p=np.zeros((len(seqs),SEQ_LEN,ref_nf),dtype=np.float32)
    p[:,:,:seqs.shape[2]]=seqs; return p

def last_wins(X, engines, cycles, seq_len, nf):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]; Xe=X[idx]
        out.append(Xe[-seq_len:] if len(Xe)>=seq_len
                   else np.vstack([np.zeros((seq_len-len(Xe),nf)),Xe]))
    return np.array(out,dtype=np.float32)

def rand_cut(X, y, engines, cycles, seq_len, seed=7):
    rng=np.random.RandomState(seed); seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]
        Xe,ye=X[idx],y[idx]; n=len(Xe)
        if n<seq_len: continue
        end=rng.randint(seq_len,n)
        seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32), np.array(labels,dtype=np.float32)

# ── MODEL: CNN + GRU FOUNDATION MODEL ────────────────────────────────────────

class DegradationFM(nn.Module):
    """
    CNN + GRU foundation model for degradation signals.
    More stable than Transformer on small datasets.
    CNN captures local sensor patterns.
    GRU captures temporal evolution.
    """
    def __init__(self, n_features, d_model=D_MODEL):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model

        # CNN feature extractor (local patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # GRU temporal encoder
        self.gru = nn.GRU(128, d_model, num_layers=2,
                          batch_first=True, dropout=0.1,
                          bidirectional=False)

        # Pre-training head: reconstruct masked channels
        self.pretrain_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, n_features)  # predict all features at last step
        )

        # RUL head (added during fine-tuning)
        self.rul_head = None

    def encode(self, x):
        # x: (B, seq_len, n_features)
        x_t = x.transpose(1, 2)           # (B, n_features, seq_len)
        c   = self.cnn(x_t)               # (B, 128, seq_len)
        c   = c.transpose(1, 2)           # (B, seq_len, 128)
        _, h = self.gru(c)                # h: (n_layers, B, d_model)
        return h[-1]                       # (B, d_model) last layer

    def forward_pretrain(self, x, masked_x):
        # Use masked input, predict original last timestep values
        emb  = self.encode(masked_x)      # (B, d_model)
        pred = self.pretrain_head(emb)    # (B, n_features)
        true = x[:, -1, :]               # (B, n_features) true last step
        return pred, true

    def add_rul_head(self):
        self.rul_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward_rul(self, x):
        return self.rul_head(self.encode(x)).squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── PRE-TRAINING: MASKED CHANNEL PREDICTION ──────────────────────────────────

class MaskedChannelDataset(Dataset):
    """
    Mask 2 random sensor channels completely.
    Predict the true values at the last timestep.
    """
    def __init__(self, seqs):
        self.seqs = torch.tensor(seqs, dtype=torch.float32)
        self.nf   = seqs.shape[2]

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        x      = self.seqs[idx].clone()   # (seq_len, n_features)
        n_mask = max(1, self.nf//7)       # mask ~15% of channels
        ch     = torch.randperm(self.nf)[:n_mask]
        masked = x.clone()
        masked[:, ch] = 0.0               # zero out masked channels
        return x, masked                  # original, masked


def pretrain(model, sequences, epochs=40, batch_size=256, lr=5e-4):
    dataset   = MaskedChannelDataset(sequences)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=3, factor=0.5, min_lr=1e-5)
    criterion = nn.MSELoss()
    model.train()
    print(f"\n  Pre-training: {len(sequences):,} seqs, {epochs} epochs")

    for ep in range(epochs):
        total=0; nb=0
        for x, masked_x in loader:
            x,masked_x=x.to(DEVICE),masked_x.to(DEVICE)
            opt.zero_grad()
            pred,true=model.forward_pretrain(x,masked_x)
            loss=criterion(pred,true)
            if torch.isnan(loss): continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            total+=loss.item(); nb+=1
        avg=total/max(nb,1)
        sched.step(avg)
        if (ep+1)%5==0:
            print(f"    Epoch {ep+1}/{epochs}  loss={avg:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")
    return model


# ── FINE-TUNING ───────────────────────────────────────────────────────────────

def finetune(model, Xs, ys, epochs=30, batch=256, lr=1e-3):
    model.add_rul_head(); model.to(DEVICE)
    # Phase 1: freeze encoder, train head only
    for p in model.cnn.parameters(): p.requires_grad=False
    for p in model.gru.parameters(): p.requires_grad=False
    loader=DataLoader(
        TensorDataset(torch.tensor(Xs,dtype=torch.float32),
                      torch.tensor(ys,dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    opt=torch.optim.Adam(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=lr, weight_decay=1e-4)
    lf=nn.MSELoss(); model.train()
    print(f"  Phase 1: head only ({epochs//2} epochs) ...")
    for ep in range(epochs//2):
        total=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad(); loss=lf(model.forward_rul(xb),yb)
            loss.backward(); opt.step(); total+=loss.item()
        if (ep+1)%(epochs//4)==0:
            print(f"    ep {ep+1} loss={total/len(loader):.2f}")
    # Phase 2: unfreeze all
    for p in model.parameters(): p.requires_grad=True
    opt2=torch.optim.Adam(model.parameters(),lr=lr*0.2,weight_decay=1e-4)
    print(f"  Phase 2: full model ({epochs//2} epochs) ...")
    for ep in range(epochs//2):
        total=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt2.zero_grad(); loss=lf(model.forward_rul(xb),yb)
            loss.backward(); opt2.step(); total+=loss.item()
        if (ep+1)%(epochs//4)==0:
            print(f"    ep {ep+1} loss={total/len(loader):.2f}")
    return model

def predict_rul(model, X, batch=256):
    model.eval(); preds=[]
    Xt=torch.tensor(X,dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(Xt),batch):
            preds.append(model.forward_rul(
                Xt[i:i+batch].to(DEVICE)).cpu().numpy())
    return np.clip(np.concatenate(preds),0,RUL_CLIP)

def conf_q(r,cov=0.90):
    n=len(r); return float(np.quantile(r,min(np.ceil((n+1)*cov)/n,1.0)))

def cov_wid(y,lo,hi): return ((y>=lo)&(y<=hi)).mean(),(hi-lo).mean()

def late_cov(y,lo,hi,pred):
    late=pred<50
    if late.sum()==0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*62)
    print("  PROPOSAL 2 v3: CNN+GRU DegradationFM (Stable)")
    print("="*62)

    # STEP 1: Pre-training corpus
    print("\n[STEP 1] Building pre-training corpus ...")
    all_seqs=[]; REF_NF=None

    for fd in ["FD001","FD002","FD003","FD004"]:
        X,engines,cycles,sc,fc,rul=load_cmapss(fd,"train")
        seqs=make_windows(X,engines,SEQ_LEN,stride=5)
        if REF_NF is None: REF_NF=X.shape[1]
        all_seqs.append(seqs)
        print(f"  C-MAPSS {fd}: {len(seqs):,} seqs")

    femto=load_femto()
    if femto is not None:
        all_seqs.append(pad_to(femto,REF_NF))
        print(f"  FEMTO: {len(femto):,} seqs")

    ims=load_ims()
    if ims is not None:
        all_seqs.append(pad_to(ims,REF_NF))
        print(f"  IMS: {len(ims):,} seqs")

    pretrain_seqs=np.vstack(all_seqs)
    np.random.seed(42); np.random.shuffle(pretrain_seqs)
    pretrain_seqs=np.clip(pretrain_seqs,-4.0,4.0)
    print(f"\n  Total: {len(pretrain_seqs):,} seqs, shape={pretrain_seqs.shape}")

    # STEP 2: Pre-train
    print("\n[STEP 2] Pre-training CNN+GRU DegradationFM ...")
    model=DegradationFM(n_features=REF_NF).to(DEVICE)
    print(f"  Parameters: {model.count_params():,}")
    model=pretrain(model,pretrain_seqs,epochs=40,batch_size=256,lr=5e-4)
    torch.save(model.state_dict(),
               r"D:\rul_project\degradation_fm_v3.pt")
    print("  Saved: degradation_fm_v3.pt")

    # STEP 3: Fine-tune on C-MAPSS
    print("\n[STEP 3] Fine-tuning on each C-MAPSS subset ...")
    results=[]

    for fd in ["FD001","FD002","FD003","FD004"]:
        print(f"\n  --- {fd} ---")
        X,engines,cycles,sc,fc,rul=load_cmapss(fd,"train")
        Xt,et,ct,_,_,_=load_cmapss(fd,"test")
        y_test=np.clip(pd.read_csv(
            os.path.join(DATA_DIR,f"RUL_{fd}.txt"),
            header=None).iloc[:,0].values,0,RUL_CLIP)

        all_e=np.unique(engines)
        np.random.seed(42); np.random.shuffle(all_e)
        n_cal=max(5,int(len(all_e)*0.20))
        cal_set=set(all_e[:n_cal])
        mtr=~np.isin(engines,list(cal_set))
        mca= np.isin(engines,list(cal_set))
        ytr=rul[mtr]; yca=rul[mca]
        eca=engines[mca]; cca=cycles[mca]

        Xs,ys=make_rul_windows(X[mtr],ytr,engines[mtr],SEQ_LEN)
        nf=Xs.shape[2]
        Xs_p=pad_to(np.clip(Xs,-4,4),REF_NF)

        ft=copy.deepcopy(model)
        ft=finetune(ft,Xs_p,ys,epochs=60)

        te_seqs=pad_to(np.clip(last_wins(Xt,et,ct,SEQ_LEN,nf),-4,4),REF_NF)
        # Multiple random cutpoints per calibration engine
cal_seqs_list, cal_y_list = [], []
for seed_i in range(5):
    cs, cy = rand_cut(X[mca],yca,eca,cca,SEQ_LEN,seed=seed_i)
    cal_seqs_list.append(cs); cal_y_list.append(cy)
cal_seqs = np.vstack(cal_seqs_list)
cal_y    = np.concatenate(cal_y_list)
                cal_seqs=pad_to(np.clip(cal_seqs,-4,4),REF_NF)

        pred    =predict_rul(ft,te_seqs)
        cal_pred=predict_rul(ft,cal_seqs)

        mae =mean_absolute_error(y_test,pred)
        rmse=np.sqrt(mean_squared_error(y_test,pred))
        r2  =r2_score(y_test,pred)
        resid=np.abs(cal_y-cal_pred)
        q=conf_q(resid)
        lo,hi=pred-q,pred+q
        cov,wid=cov_wid(y_test,lo,hi)
        lc=late_cov(y_test,lo,hi,pred)
        h="YES" if abs(cov-COVERAGE)<0.10 else "NO"

        print(f"  MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")
        print(f"  Cov={cov:.0%} Wid={wid:.1f} Late={lc:.0%} Honest={h}")
        results.append(dict(fd=fd,MAE=round(mae,2),RMSE=round(rmse,2),
                            R2=round(r2,3),cov=f"{cov:.0%}",
                            wid=round(wid,1),late=f"{lc:.0%}",honest=h))

    # Final comparison
    print(f"\n\n{'='*68}")
    print("  PROPOSAL 2 (CNN+GRU) vs PROPOSAL 1 (Chronos-2) vs LSTM")
    print(f"{'='*68}")
    print(f"  {'FD':<6}{'P2 MAE':>9}{'P1 MAE':>9}{'LSTM':>9}"
          f"{'Cov':>8}  Better?")
    print(f"  {'-'*52}")
    p1  ={'FD001':'15.17','FD002':'19.16','FD003':'12.71','FD004':'19.03'}
    lstm={'FD001':'10.42','FD002':'11.95','FD003':'11.88','FD004':'13.45'}
    for r in results:
        b=float(r['MAE'])<float(p1[r['fd']])
        print(f"  {r['fd']:<6}{r['MAE']:>9}{p1[r['fd']]:>9}"
              f"{lstm[r['fd']]:>9}{r['cov']:>8}  "
              f"{'YES <- P2 beats P1' if b else 'no'}")
    print(f"{'='*68}")
    print("\n  KEY: CNN+GRU pre-trained on degradation data from")
    print("  C-MAPSS + FEMTO + IMS. Does it beat Chronos-2?")

if __name__=="__main__":
    main()
