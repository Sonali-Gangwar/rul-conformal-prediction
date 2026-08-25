"""
cqr_v2.py — CQR baseline with fixed LSTM training

FIXES:
  1. LSTM output shape fixed (squeeze before loss)
  2. Use same LSTM as paper (ensemble with RF+XGB+LSTM)
  3. Proper late-life coverage calculation
  4. Clean output for direct copy to paper

Run: python cqr_v2.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fd(split, fd):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{split}_{fd}.txt"),
                     sep=r"\s+",header=None).iloc[:,:26]
    df.columns=COLS; return df

def prepare(fd):
    train=load_fd("train",fd); test=load_fd("test",fd)
    rul_t=pd.read_csv(os.path.join(DATA_DIR,f"RUL_{fd}.txt"),
                      header=None).iloc[:,0].values
    fc=[c for c in COLS if c.startswith("s") and c not in DROP]
    mc=train.groupby("engine")["cycle"].transform("max")
    train["RUL"]=(mc-train["cycle"]).clip(upper=RUL_CLIP)
    all_e=np.unique(train["engine"].values)
    np.random.seed(42); np.random.shuffle(all_e)
    n_cal=max(5,int(len(all_e)*0.20)); cal_set=set(all_e[:n_cal])
    mtr=~train["engine"].isin(cal_set); mca=train["engine"].isin(cal_set)
    sc=StandardScaler().fit(train.loc[mtr,fc].values)
    return (sc.transform(train.loc[mtr,fc].values),
            train.loc[mtr,"RUL"].values,
            train.loc[mtr,"engine"].values,
            train.loc[mtr,"cycle"].values,
            sc.transform(train.loc[mca,fc].values),
            train.loc[mca,"RUL"].values,
            train.loc[mca,"engine"].values,
            train.loc[mca,"cycle"].values,
            sc.transform(test[fc].values),
            test["engine"].values,test["cycle"].values,
            np.clip(rul_t,0,RUL_CLIP))

def make_seqs(X,y,engines,seq_len):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]; Xe,ye=X[idx],y[idx]
        for end in range(seq_len,len(Xe)+1):
            seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32),np.array(labels,dtype=np.float32)

def last_seqs(X,engines,cycles,seq_len,nf):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]; Xe=X[idx]
        out.append(Xe[-seq_len:] if len(Xe)>=seq_len
                   else np.vstack([np.zeros((seq_len-len(Xe),nf)),Xe]))
    return np.array(out,dtype=np.float32)

def rand_cut(X,y,engines,cycles,seq_len,seed=7):
    rng=np.random.RandomState(seed); seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]
        Xe,ye=X[idx],y[idx]; n=len(Xe)
        if n<seq_len: continue
        end=rng.randint(seq_len,n)
        seqs.append(Xe[end-seq_len:end]); labels.append(ye[end-1])
    return np.array(seqs,dtype=np.float32),np.array(labels,dtype=np.float32)

class LSTM(nn.Module):
    def __init__(self,nf):
        super().__init__()
        self.lstm=nn.LSTM(nf,64,num_layers=2,batch_first=True,dropout=0.2)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        o,_=self.lstm(x); return self.head(o[:,-1,:]).squeeze(-1)

def train_lstm(model,Xs,ys,epochs=30):
    loader=DataLoader(TensorDataset(
        torch.tensor(Xs),torch.tensor(ys)),
        batch_size=256,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    lf=nn.MSELoss(); model.train()
    for ep in range(epochs):
        total=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            pred=model(xb)           # (B,) NOT (B,1)
            loss=lf(pred,yb)         # both (B,) — no shape mismatch
            loss.backward(); opt.step(); total+=loss.item()
        if (ep+1)%10==0:
            print(f"    ep {ep+1}/{epochs} loss={total/len(loader):.2f}")
    return model

def pred_lstm(model,seqs):
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0,len(seqs),512):
            xb=torch.tensor(seqs[i:i+512]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(preds),0,RUL_CLIP)

def conf_q(r,cov=0.90):
    n=len(r); return float(np.quantile(r,min(np.ceil((n+1)*cov)/n,1.0)))

def cov_wid(y,lo,hi): return ((y>=lo)&(y<=hi)).mean(),(hi-lo).mean()

def late_cov(y,lo,hi,pred):
    late=pred<50
    if late.sum()==0: return float('nan')
    return ((y[late]>=lo[late])&(y[late]<=hi[late])).mean()

def run_cqr(tr_X_feat,tr_y,ca_X_feat,ca_y,te_X_feat,y_test,te_pred,alpha=0.10):
    """True CQR: train quantile regressors, then apply conformal correction."""
    qr_lo=QuantileRegressor(quantile=alpha/2,alpha=0.001,solver='highs')
    qr_hi=QuantileRegressor(quantile=1-alpha/2,alpha=0.001,solver='highs')
    qr_lo.fit(tr_X_feat,tr_y)
    qr_hi.fit(tr_X_feat,tr_y)
    # calibration conformity scores
    ca_lo=qr_lo.predict(ca_X_feat)
    ca_hi=qr_hi.predict(ca_X_feat)
    scores=np.maximum(ca_lo-ca_y, ca_y-ca_hi)
    n=len(scores)
    q_hat=float(np.quantile(scores,min(np.ceil((n+1)*(1-alpha))/n,1.0)))
    # test intervals
    te_lo=qr_lo.predict(te_X_feat)-q_hat
    te_hi=qr_hi.predict(te_X_feat)+q_hat
    cov,wid=cov_wid(y_test,te_lo,te_hi)
    lc=late_cov(y_test,te_lo,te_hi,te_pred)
    return cov,wid,lc,q_hat,te_lo,te_hi

def main():
    print("="*60)
    print("  CQR Baseline — Paper Tables 3 and 4")
    print(f"  Device: {DEVICE}")
    print("="*60)

    results=[]

    for fd in ["FD001","FD002","FD003","FD004"]:
        print(f"\n{'='*55}")
        print(f"  {fd}")
        print(f"{'='*55}")

        (Xtr,ytr,etr,ctr,Xca,yca,eca,cca,
         Xte,ete,cte,y_test)=prepare(fd)
        nf=Xtr.shape[1]

        # --- sequences ---
        Xs,ys=make_seqs(Xtr,ytr,etr,SEQ_LEN)
        te_seqs=last_seqs(Xte,ete,cte,SEQ_LEN,nf)

        # last-cycle cal seqs
        ca_last_seqs=last_seqs(Xca,eca,cca,SEQ_LEN,nf)
        ca_last_y=np.array([yca[np.where(eca==e)[0][-1]]
                            for e in np.unique(eca)],dtype=np.float32)

        # random-cutoff cal seqs
        ca_rc_seqs,ca_rc_y=rand_cut(Xca,yca,eca,cca,SEQ_LEN)

        print(f"  Train seqs: {len(Xs):,}  Test: {len(te_seqs)}"
              f"  Cal(last): {len(ca_last_y)}  Cal(rc): {len(ca_rc_y)}")

        # --- train RF + XGB + LSTM ensemble ---
        tr_feat=Xs[:,-1,:]  # last frame for RF/XGB
        print("  Training RF + XGB ...")
        rf=RandomForestRegressor(100,random_state=42,n_jobs=-1).fit(tr_feat,ys)
        xg=xgb.XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.1,
                             random_state=42,n_jobs=-1).fit(tr_feat,ys)
        print("  Training LSTM ...")
        lstm=LSTM(nf).to(DEVICE)
        lstm=train_lstm(lstm,Xs,ys,epochs=30)

        # --- predict ---
        def ensemble_pred(seqs):
            feat=seqs[:,-1,:]
            rf_p=rf.predict(feat)
            xg_p=xg.predict(feat)
            ls_p=pred_lstm(lstm,seqs)
            return np.clip((rf_p+xg_p+ls_p)/3,0,RUL_CLIP)

        tr_pred=ensemble_pred(Xs)
        te_pred=ensemble_pred(te_seqs)
        ca_last_pred=ensemble_pred(ca_last_seqs)
        ca_rc_pred=ensemble_pred(ca_rc_seqs)

        mae=mean_absolute_error(y_test,te_pred)
        print(f"  Ensemble MAE={mae:.2f}")

        # --- 1. Standard CP (last-cycle) ---
        r_std=np.abs(ca_last_y-ca_last_pred)
        q_std=conf_q(r_std)
        lo_s,hi_s=te_pred-q_std,te_pred+q_std
        cov_s,wid_s=cov_wid(y_test,lo_s,hi_s)
        lc_s=late_cov(y_test,lo_s,hi_s,te_pred)

        # --- 2. CQR (quantile regression + conformal) ---
        # Use last-frame features for quantile regression
        tr_feat_all=Xs[:,-1,:]
        ca_feat=(ca_rc_seqs[:,-1,:])  # use rc cal set for cqr calibration
        te_feat=te_seqs[:,-1,:]
        cov_cqr,wid_cqr,lc_cqr,q_cqr,_,_=run_cqr(
            tr_feat_all,ys,ca_feat,ca_rc_y,te_feat,y_test,te_pred)

        # --- 3. Global random-cutoff CP ---
        r_rc=np.abs(ca_rc_y-ca_rc_pred)
        q_rc=conf_q(r_rc)
        lo_r,hi_r=te_pred-q_rc,te_pred+q_rc
        cov_r,wid_r=cov_wid(y_test,lo_r,hi_r)
        lc_r=late_cov(y_test,lo_r,hi_r,te_pred)

        hn=lambda c: 'YES' if abs(c-COVERAGE)<0.10 else 'NO'
        lf=lambda v: f"{v:.0%}" if not np.isnan(v) else 'nan'

        print(f"\n  {'Method':<22}{'Cov':>8}{'Wid':>8}"
              f"{'Late':>8}{'q':>8}  Honest?")
        print(f"  {'-'*56}")
        for nm,cov,wid,lc,q in [
            ('Standard CP',cov_s,wid_s,lc_s,q_std),
            ('CQR',cov_cqr,wid_cqr,lc_cqr,q_cqr),
            ('Random-cutoff CP',cov_r,wid_r,lc_r,q_rc),
        ]:
            print(f"  {nm:<22}{cov:>7.1%}{wid:>8.1f}"
                  f"{lf(lc):>8}{q:>8.2f}  {hn(cov)}")

        results.append(dict(fd=fd,mae=round(mae,2),
            std_cov=f"{cov_s:.0%}",std_wid=round(wid_s,1),std_q=round(q_std,1),
            std_lc=lf(lc_s),
            cqr_cov=f"{cov_cqr:.0%}",cqr_wid=round(wid_cqr,1),cqr_q=round(q_cqr,1),
            cqr_lc=lf(lc_cqr),
            rc_cov=f"{cov_r:.0%}",rc_wid=round(wid_r,1),rc_q=round(q_rc,1),
            rc_lc=lf(lc_r)))

    print(f"\n\n{'='*70}")
    print("  COPY INTO PAPER TABLE 3 (Coverage comparison)")
    print(f"{'='*70}")
    print(f"  {'FD':<6}{'Std cov':>8}{'Std wid':>8}{'Std q':>6}"
          f"{'CQR cov':>8}{'CQR wid':>8}{'CQR q':>6}{'RC cov':>8}")
    for r in results:
        print(f"  {r['fd']:<6}{r['std_cov']:>8}{r['std_wid']:>8}"
              f"{r['std_q']:>6}{r['cqr_cov']:>8}{r['cqr_wid']:>8}"
              f"{r['cqr_q']:>6}{r['rc_cov']:>8}")

    print(f"\n  COPY INTO PAPER TABLE 4 (Late-life coverage)")
    print(f"  {'FD':<6}{'Std late':>10}{'CQR late':>10}{'RC late':>10}")
    for r in results:
        print(f"  {r['fd']:<6}{r['std_lc']:>10}{r['cqr_lc']:>10}{r['rc_lc']:>10}")
    print(f"{'='*70}")

if __name__=="__main__":
    main()
