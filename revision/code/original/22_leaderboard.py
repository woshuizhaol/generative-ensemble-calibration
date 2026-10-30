"""Stage2 主线①评测:给定各生成器对各蛋白的预测ΔG_open(operator输出),与实测对比→
分布/排序/协同指标 + 蛋白层bootstrap CI + 对照算子上限。
预测文件约定:eval/pred/<model>/<protein>.csv(列 resSeq,dGopen_pred[, NC_mean,NH_mean])。
"""
import os as _os, sys as _sys
_GEC_ROOT = _os.environ.get("GEC_PROJECT_ROOT")
for _i, _x in enumerate(_sys.argv[1:], 1):
    if _x.startswith("--project-root="):
        _GEC_ROOT = _x.split("=", 1)[1]; del _sys.argv[_i]; break
if not _GEC_ROOT:
    raise SystemExit("需要环境变量 GEC_PROJECT_ROOT 或参数 --project-root=<包内 inputs 目录>")
_os.environ["GEC_PROJECT_ROOT"] = _GEC_ROOT   # 被导入的库模块（例如 operator/forward.py）也读同一个根目录
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0,(_GEC_ROOT + '/code/utils')); import metrics as M
PROJ=_GEC_ROOT; LN10=np.log(10)
def exp_table():
    a=pd.read_parquet(f"{PROJ}/data/processed/tier1a_all.parquet")  # protein_id,resi,dGopen_kcal,grade
    return a
def load_pred(model,prot):
    f=f"{PROJ}/eval/pred/{model}/{prot}.csv"
    return pd.read_csv(f) if os.path.exists(f) else None
A1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def _to1(x):
    x=str(x); return A1.get(x.upper(),x[:1].upper()) if len(x)==3 else (x[:1].upper() if x.strip() else '')
def best_offset(pred,g,default):
    eid={int(r.resi):_to1(r.resn) for _,r in g.iterrows() if str(r.resn).strip()}
    if len(eid)<5: return default
    pres={int(r.resSeq):_to1(r.resn) for _,r in pred.iterrows()} if 'resn' in pred.columns else {}
    if not pres: return default
    best,bg=default,-1
    for off in range(-8,9):
        gg=sum(1 for er,aa in eid.items() if pres.get(er+off)==aa)
        if gg>bg: bg,best=gg,off
    return best
def evaluate(models=None):
    exp=exp_table(); maps=json.load(open(f"{PROJ}/data/processed/structure_maps.json"))
    if models is None: models=sorted([os.path.basename(d) for d in glob.glob(f"{PROJ}/eval/pred/*") if os.path.isdir(d)])
    out=[]
    for model in models:
        per_prot_spearman=[]
        for prot,g in exp.groupby('protein_id'):
            pred=load_pred(model,prot)
            if pred is None: continue
            off=best_offset(pred,g,maps.get(prot,{}).get('gen_offset',maps.get(prot,{}).get('offset',0)))
            pred=pred.copy(); pred['resi']=pred['resSeq']-off
            m=g.merge(pred,on='resi',how='inner').dropna(subset=['dGopen_kcal','dGopen_pred'])
            if len(m)<5: continue
            rm=M.rank_metrics(m.dGopen_pred.values, m.dGopen_kcal.values)
            row=dict(model=model,protein=prot,n=len(m),grade=g.grade.iloc[0],
                     spearman=round(rm['spearman'],3),kendall=round(rm['kendall'],3),pairwise=round(rm['pairwise_acc'],3),
                     w1=round(M.w1(m.dGopen_pred.values,m.dGopen_kcal.values),3),
                     ks=round(M.ks(m.dGopen_pred.values,m.dGopen_kcal.values),3),
                     adj_corr=round(M.adjacent_corr(m.resi.values,m.dGopen_pred.values),3))
            out.append(row); per_prot_spearman.append(rm['spearman'])
    df=pd.DataFrame(out)
    if len(df):
        df.to_parquet(f"{PROJ}/eval/leaderboard.parquet",index=False)
        # 模型层汇总(蛋白层bootstrap CI of spearman)
        summ=[]
        for model,gd in df.groupby('model'):
            mean,lo,hi=M.protein_bootstrap_ci(gd.spearman.values)
            summ.append(dict(model=model,n_prot=gd.protein.nunique(),mean_spearman=round(mean,3),ci_lo=round(lo,3),ci_hi=round(hi,3)))
        S=pd.DataFrame(summ); S.to_csv(f"{PROJ}/results/tables/leaderboard_summary.csv",index=False)
        print(df.to_string(index=False)); print("\n模型汇总(蛋白层bootstrap CI):\n",S.to_string(index=False))
    else: print("无预测文件(eval/pred/<model>/<protein>.csv);Stage2生成后运行。")
    return df
if __name__=='__main__': evaluate()
