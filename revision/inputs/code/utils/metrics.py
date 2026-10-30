"""评测指标(proposal §7):分布级/排序级/协同性/分类/全局锚定 + 蛋白层 bootstrap CI。"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr, ks_2samp, wasserstein_distance
from scipy.special import rel_entr
from sklearn.metrics import roc_auc_score, average_precision_score, adjusted_rand_score

# ---- 分布级 ----
def w1(pred, exp): return float(wasserstein_distance(pred, exp))
def ks(pred, exp): return float(ks_2samp(pred, exp).statistic)
def kl(pred, exp, bins=20):
    lo=min(pred.min(),exp.min()); hi=max(pred.max(),exp.max())
    p,_=np.histogram(pred,bins=bins,range=(lo,hi),density=True); q,_=np.histogram(exp,bins=bins,range=(lo,hi),density=True)
    p=p+1e-9; q=q+1e-9; p/=p.sum(); q/=q.sum(); return float(np.sum(rel_entr(p,q)))
def spread(x): return dict(iqr=float(np.percentile(x,75)-np.percentile(x,25)), rng=float(x.max()-x.min()), sd=float(np.std(x)))

# ---- 排序级 ----
def rank_metrics(pred, exp):
    return dict(spearman=float(spearmanr(pred,exp)[0]), kendall=float(kendalltau(pred,exp)[0]),
                pairwise_acc=_pairwise_acc(pred,exp))
def _pairwise_acc(pred, exp):
    n=len(pred); c=t=0
    pr=np.asarray(pred); ex=np.asarray(exp)
    for i in range(n):
        for j in range(i+1,n):
            if ex[i]==ex[j]: continue
            t+=1; c+= ((pr[i]>pr[j])==(ex[i]>ex[j]))
    return float(c/t) if t else np.nan

# ---- 协同性/阶梯结构 ----
def adjacent_corr(resi, dG):
    df=pd.DataFrame({'r':resi,'g':dG}).sort_values('r'); a=df.g.values
    pairs=[(a[i],a[i+1]) for i in range(len(a)-1) if df.r.values[i+1]-df.r.values[i]==1]
    if len(pairs)<3: return np.nan
    return float(pearsonr([p[0] for p in pairs],[p[1] for p in pairs])[0])
def foldon_ari(pred_labels, true_labels): return float(adjusted_rand_score(true_labels, pred_labels))

# ---- 分类(Tier-2) ----
def auroc(y_true_bin, score): 
    try: return float(roc_auc_score(y_true_bin, score))
    except: return np.nan
def auprc(y_true_bin, score):
    try: return float(average_precision_score(y_true_bin, score))
    except: return np.nan

# ---- 全局锚定(Tier-3) ----
def global_anchor(pred_global, exp_dGfold):
    return dict(pearson=float(pearsonr(pred_global,exp_dGfold)[0]), spearman=float(spearmanr(pred_global,exp_dGfold)[0]))

# ---- 蛋白层 bootstrap CI ----
def protein_bootstrap_ci(per_protein_values, n_boot=2000, seed=0, ci=95):
    rng=np.random.RandomState(seed); v=np.asarray(per_protein_values,float); v=v[~np.isnan(v)]
    if len(v)==0: return (np.nan,np.nan,np.nan)
    boots=[np.mean(rng.choice(v,size=len(v),replace=True)) for _ in range(n_boot)]
    lo=np.percentile(boots,(100-ci)/2); hi=np.percentile(boots,100-(100-ci)/2)
    return (float(np.mean(v)), float(lo), float(hi))

# ---- 模型间配对 bootstrap 差异 ----
def paired_bootstrap_diff(per_protein_A, per_protein_B, n_boot=2000, seed=0):
    rng=np.random.RandomState(seed); A=np.asarray(per_protein_A,float); B=np.asarray(per_protein_B,float)
    m=~(np.isnan(A)|np.isnan(B)); A,B=A[m],B[m]
    if len(A)==0: return (np.nan,np.nan,np.nan,np.nan)
    d=A-B; boots=[np.mean(rng.choice(d,size=len(d),replace=True)) for _ in range(n_boot)]
    p=2*min(np.mean(np.array(boots)<=0),np.mean(np.array(boots)>=0))
    return (float(np.mean(d)), float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5)), float(p))

def bh_fdr(pvals):
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); ranked=p[order]*n/np.arange(1,n+1)
    q=np.minimum.accumulate(ranked[::-1])[::-1]; out=np.empty(n); out[order]=np.minimum(q,1); return out
