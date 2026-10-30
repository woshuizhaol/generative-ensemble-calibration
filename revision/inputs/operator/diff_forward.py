"""E9: 可微 BV 算子(PyTorch)。N_C 软 burial=Σ sigmoid((cut-d)/τ);N_H 软氢键=距离软门控。
τ→0 复现硬 BV。可反传→未来推理时 HDX 引导。"""
import numpy as np, torch
def soft_nc_nh(xyz, amideN_idx, amideH_idx, heavy_idx, O_idx, resid, cut_Nc=0.65, cut_Nh=0.26, ij_Nc=3, ij_Nh=2, tau=0.02):
    # xyz: [A,3] torch(nm); 各idx: 长度对应残基的原子索引(-1=缺)
    dev=xyz.device; R=len(amideN_idx)
    NC=torch.zeros(R,device=dev); NH=torch.zeros(R,device=dev)
    heavy=torch.tensor(heavy_idx,device=dev); hres=torch.tensor([resid[i] for i in heavy_idx],device=dev)
    Os=torch.tensor(O_idx,device=dev); Ores=torch.tensor([resid[i] for i in O_idx],device=dev)
    for r in range(R):
        ni=amideN_idx[r]
        if ni<0: continue
        d=torch.norm(xyz[heavy]-xyz[ni],dim=1)  # N到所有重原子
        mask=(torch.abs(hres-resid[ni])>=ij_Nc).float()
        NC[r]=torch.sum(torch.sigmoid((cut_Nc-d)/tau)*mask)
        hi=amideH_idx[r]
        if hi>=0:
            dh=torch.norm(xyz[Os]-xyz[hi],dim=1); mh=(torch.abs(Ores-resid[ni])>=ij_Nh).float()
            NH[r]=torch.sum(torch.sigmoid((cut_Nh-dh)/tau)*mh)
    return NC,NH
def validate(pdb, tau=0.02):
    import mdtraj as md, sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import forward
    t=forward._fix_frame(md.load(pdb)[0],True); top=t.topology
    hard=forward._nc_nh(t,0.65,0.26,3,2)  # {resSeq:(resn,NC,NH)}
    # 构建索引
    resid=[a.residue.resSeq for a in top.atoms]
    aN={};aH={};
    for a in top.atoms:
        if a.name=='N': aN[a.residue.resSeq]=a.index
        if a.name=='H' or a.name=='HN': aH[a.residue.resSeq]=a.index
    rss=sorted(aN); amideN=[aN[r] for r in rss]; amideH=[aH.get(r,-1) for r in rss]
    heavy=[a.index for a in top.atoms if a.element.symbol!='H']
    Oidx=[a.index for a in top.atoms if a.name=='O']
    xyz=torch.tensor(t.xyz[0],dtype=torch.float64,requires_grad=True)
    NC,NH=soft_nc_nh(xyz,amideN,amideH,heavy,Oidx,resid,tau=tau)
    # 对齐硬 BV
    from scipy.stats import pearsonr
    hc=[hard[r][1] for r in rss if r in hard]; sc=[NC[i].item() for i,r in enumerate(rss) if r in hard]
    hh=[hard[r][2] for r in rss if r in hard]; sh=[NH[i].item() for i,r in enumerate(rss) if r in hard]
    lnpf_hard=np.array([0.35*hard[r][1]+2.0*hard[r][2] for r in rss if r in hard])
    lnpf_soft=np.array([0.35*sc[i]+2.0*sh[i] for i in range(len(sc))])
    # 可反传验证
    loss=NC.sum(); loss.backward(); grad_ok=xyz.grad is not None and torch.isfinite(xyz.grad).all().item()
    return dict(pdb=pdb.split('/')[-1],n=len(hc),NC_pearson=pearsonr(hc,sc)[0],NH_pearson=pearsonr(hh,sh)[0] if np.std(hh)>0 else 1.0,
                lnPF_pearson=pearsonr(lnpf_hard,lnpf_soft)[0],grad_finite=grad_ok)
if __name__=='__main__':
    import os
    PROJ=os.environ.get('GEC_PROJECT_ROOT','revision/inputs')   # folder with data/external/pdb/*.pdb
    print("=== E9 可微 BV 算子验证(soft τ→0 vs 硬 BV)===")
    for pdb in ['1A2P','1SNO','1HRC']:
        for tau in [0.05,0.02,0.005]:
            r=validate(f'{PROJ}/data/external/pdb/{pdb}.pdb',tau)
            print(f"  {r['pdb']} τ={tau}: NC Pearson={r['NC_pearson']:.4f} NH={r['NH_pearson']:.4f} lnPF={r['lnPF_pearson']:.4f} grad可反传={r['grad_finite']}")
