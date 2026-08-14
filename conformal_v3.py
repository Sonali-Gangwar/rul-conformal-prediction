"""
conformal_v3.py  —  STEP 2 final version

Calibration strategy:
- Split training engines into 80 train / 20 calibration engines
- Train LSTM on 80 engines only
- Get LSTM predictions on the 20 calibration engines (last cycle of each)
- Residuals from those 20 engines = calibration residuals
- These match test difficulty because calibration engines are also "cut off"
  at their last cycle, just like test engines

Also computes variance and confidence intervals from the 3-model ensemble.
"""

import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_DIR = "data"
SEQ_LEN  = 30
RUL_CLIP = 125
COVERAGE = 0.90
COLS     = ["engine","cycle","op1","op2","op3"]+[f"s{i}" for i in range(1,22)]
DROP     = {"s1","s5","s6","s10","s16","s18","s19"}

def load(split):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{split}_FD001.txt"),
                     sep=r"\s+",header=None).iloc[:,:26]
    df.columns = COLS
    return df

def make_seqs(X,y,engines,seq_len):
    seqs,labels=[],[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        Xe,ye=X[idx],y[idx]
        for end in range(seq_len,len(Xe)+1):
            seqs.append(Xe[end-seq_len:end])
            labels.append(ye[end-1])
    return np.array(seqs),np.array(labels)

def last_seqs(X,engines,cycles,seq_len,nf):
    out=[]
    for e in np.unique(engines):
        idx=np.where(engines==e)[0]
        idx=idx[np.argsort(cycles[engines==e])]
        Xe=X[idx]
        if len(Xe)>=seq_len:
            out.append(Xe[-seq_len:])
        else:
            out.append(np.vstack([np.zeros((seq_len-len(Xe),nf)),Xe]))
    return np.array(out)

def last_row(X,engines):
    return np.array([X[np.where(engines==e)[0][-1]] for e in np.unique(engines)])

def train_lstm(Xtr,ytr,eng_tr,nf,device,epochs=30):
    Xs,ys=make_seqs(Xtr,ytr,eng_tr,SEQ_LEN)
    class LSTMReg(nn.Module):
        def __init__(self,n):
            super().__init__()
            self.lstm=nn.LSTM(n,64,num_layers=2,batch_first=True,dropout=0.2)
            self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
        def forward(self,x):
            o,_=self.lstm(x); return self.head(o[:,-1,:])
    model=LSTMReg(nf).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    lf=nn.MSELoss()
    loader=DataLoader(TensorDataset(torch.tensor(Xs,dtype=torch.float32),
                      torch.tensor(ys,dtype=torch.float32).unsqueeze(1)),
                      batch_size=256,shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(); lf(model(xb),yb).backward(); opt.step()
    return model

def lstm_predict(model,X,engines,cycles,nf,device):
    seqs=last_seqs(X,engines,cycles,SEQ_LEN,nf)
    model.eval()
    with torch.no_grad():
        p=model(torch.tensor(seqs,dtype=torch.float32).to(device)).cpu().numpy().ravel()
    return np.clip(p,0,RUL_CLIP)

def conformal_q(resid,coverage):
    n=len(resid)
    level=min(np.ceil((n+1)*coverage)/n,1.0)
    return np.quantile(resid,level)

def coverage_width(y,lo,hi):
    return ((y>=lo)&(y<=hi)).mean(),(hi-lo).mean()

def main():
    # load
    train=load("train"); test=load("test")
    true_rul=pd.read_csv(os.path.join(DATA_DIR,"RUL_FD001.txt"),
                         header=None).iloc[:,0].values
    fcols=[c for c in COLS if c.startswith("s") and c not in DROP]
    mc=train.groupby("engine")["cycle"].transform("max")
    train["RUL"]=(mc-train["cycle"]).clip(upper=RUL_CLIP)

    # split engines: 80 train / 20 calibration
    all_eng=np.unique(train["engine"].values)
    np.random.seed(42)
    np.random.shuffle(all_eng)
    eng_cal_set=set(all_eng[:20])
    mask_tr=~train["engine"].isin(eng_cal_set)
    mask_ca= train["engine"].isin(eng_cal_set)

    # scale on 80-engine training split only
    scaler=StandardScaler().fit(train.loc[mask_tr,fcols].values)
    Xtr=scaler.transform(train.loc[mask_tr,fcols].values)
    Xca=scaler.transform(train.loc[mask_ca,fcols].values)
    Xte=scaler.transform(test[fcols].values)

    ytr=train.loc[mask_tr,"RUL"].values
    yca=train.loc[mask_ca,"RUL"].values
    eng_tr=train.loc[mask_tr,"engine"].values
    eng_ca=train.loc[mask_ca,"engine"].values
    cyc_ca=train.loc[mask_ca,"cycle"].values
    eng_te=test["engine"].values
    cyc_te=test["cycle"].values
    y_test=np.clip(true_rul,0,RUL_CLIP)
    nf=Xtr.shape[1]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train models on 80 engines
    print("Training RF + XGBoost on 80 engines …")
    rf =RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    xg =xgb.XGBRegressor(n_estimators=100,max_depth=6,learning_rate=0.1,
                          random_state=42,n_jobs=-1).fit(Xtr,ytr)

    print(f"Training LSTM on 80 engines ({device}) …")
    model=train_lstm(Xtr,ytr,eng_tr,nf,device,epochs=30)

    # calibration residuals from 20 held-out engines (last cycle = same as test)
    print("Computing calibration residuals on 20 held-out engines …")
    cal_last_X =last_row(Xca,eng_ca)
    cal_y_true =np.array([yca[np.where(eng_ca==e)[0][-1]]
                          for e in np.unique(eng_ca)])
    rf_cal  =rf.predict(cal_last_X)
    xgb_cal =xg.predict(cal_last_X)
    lstm_cal=lstm_predict(model,Xca,eng_ca,cyc_ca,nf,device)
    ens_cal =(rf_cal+xgb_cal+lstm_cal)/3.0
    calib_resid=np.abs(cal_y_true - ens_cal)
    print(f"  calibration residuals: mean={calib_resid.mean():.2f}, "
          f"max={calib_resid.max():.2f}, n={len(calib_resid)}")

    # test predictions from all 3 models
    print("Predicting on 100 test engines …")
    te_last=last_row(Xte,eng_te)
    rf_pred  =rf.predict(te_last)
    xgb_pred =xg.predict(te_last)
    lstm_pred=lstm_predict(model,Xte,eng_te,cyc_te,nf,device)
    ens_pred =(rf_pred+xgb_pred+lstm_pred)/3.0

    # point metrics
    mae,rmse,r2=mean_absolute_error(y_test,ens_pred),\
                 np.sqrt(mean_squared_error(y_test,ens_pred)),\
                 r2_score(y_test,ens_pred)

    # 1. VARIANCE INTERVAL
    model_preds=np.stack([rf_pred,xgb_pred,lstm_pred],axis=1)
    std=model_preds.std(axis=1)
    # find z that gives ~90% empirical coverage on calibration set
    cal_stack=np.stack([rf_cal,xgb_cal,lstm_cal],axis=1)
    cal_std=cal_stack.std(axis=1)
    best_z,best_diff=1.0,999
    for z in np.arange(0.5,6.0,0.05):
        cov_z=((cal_y_true>=(ens_cal-z*cal_std))&
               (cal_y_true<=(ens_cal+z*cal_std))).mean()
        if abs(cov_z-COVERAGE)<best_diff:
            best_diff=abs(cov_z-COVERAGE); best_z=z
    var_lo=ens_pred-best_z*std; var_hi=ens_pred+best_z*std
    var_cov,var_w=coverage_width(y_test,var_lo,var_hi)

    # 2. CONFIDENCE INTERVAL (bootstrap on 3 models)
    ci_lo=ens_pred-best_z*(std/np.sqrt(3))
    ci_hi=ens_pred+best_z*(std/np.sqrt(3))
    ci_cov,ci_w=coverage_width(y_test,ci_lo,ci_hi)

    # 3. CONFORMAL INTERVAL
    q=conformal_q(calib_resid,COVERAGE)
    conf_lo=ens_pred-q; conf_hi=ens_pred+q
    conf_cov,conf_w=coverage_width(y_test,conf_lo,conf_hi)

    # print
    print()
    print("="*62)
    print("  ENSEMBLE POINT PREDICTION (trained on 80 engines)")
    print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")
    print("="*62)
    print(f"  {'Method':<28}{'Coverage':>9}{'Avg width':>10}  Honest?")
    print("-"*62)
    def h(c): return "YES" if abs(c-COVERAGE)<0.08 else "NO"
    print(f"  {'Variance interval':<28}{var_cov:>8.1%}{var_w:>10.2f}  {h(var_cov)}")
    print(f"  {'Confidence interval':<28}{ci_cov:>8.1%}{ci_w:>10.2f}  {h(ci_cov)}")
    print(f"  {'Conformal (target 90%)':<28}{conf_cov:>8.1%}{conf_w:>10.2f}  {h(conf_cov)}")
    print("="*62)
    print(f"\n  Conformal q (half-width) = {q:.2f} cycles")
    print(f"  Calibrated z for variance = {best_z:.2f}")
    print(f"  Avg model std (test)      = {std.mean():.2f} cycles")
    print()
    print("  KEY RESEARCH FINDING:")
    print("  Compare which interval is honest AND narrowest.")
    print("  The gap between methods IS your paper's contribution.")
    print("="*62)

if __name__=="__main__":
    main()
