# scripts/eval_unet_multitask_boundary.py
import os, re, json, argparse, warnings
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Constants
# --------------------------
RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']
CLASS_BG, CLASS_P, CLASS_QRS, CLASS_T = 0, 1, 2, 3
BOUNDARY_TYPES = [
    ("P_on", CLASS_P, "on"), ("P_off", CLASS_P, "off"),
    ("QRS_on", CLASS_QRS, "on"), ("QRS_off", CLASS_QRS, "off"),
    ("T_on", CLASS_T, "on"), ("T_off", CLASS_T, "off")
]

# --------------------------
# Dataset (mask-from-meta)
# --------------------------
def parse_record_and_rhythm_from_name(path):
    base = os.path.basename(path)
    rec = base.split('_ii_')[0]
    m = re.match(r'^([A-Z]+)\d+', rec)
    rhythm = m.group(1) if m else None
    return rec, rhythm

class ECGSegments(Dataset):
    def __init__(self, record_ids, data_dir, rhythms=RHYTHMS, lead_filter='_ii_'):
        self.data_dir = data_dir
        self.rhythm_to_id = {r:i for i,r in enumerate(rhythms)}
        self.files, self.rhythm_ids = [], []
        with os.scandir(data_dir) as it:
            for f in it:
                if f.is_file() and f.name.endswith('.npz') and lead_filter in f.name:
                    rec = f.name.split('_ii_')[0]
                    if rec in record_ids:
                        _, rstr = parse_record_and_rhythm_from_name(f.path)
                        rid = self.rhythm_to_id.get(rstr, 0)
                        self.files.append(f.path); self.rhythm_ids.append(rid)
        order = np.argsort(self.files)
        self.files = [self.files[i] for i in order]
        self.rhythm_ids = [self.rhythm_ids[i] for i in order]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        z = np.load(p, allow_pickle=True)
        sig = z['signal'].astype(np.float32)   # [512]
        lab = z['labels'].astype(np.int64)     # [512]
        rid = self.rhythm_ids[idx]

        L = lab.shape[0]
        meta = z['meta'].item() if 'meta' in z else {}
        if 'orig_start' in meta and 'orig_end' in meta:
            orig_len = int(meta['orig_end']) - int(meta['orig_start']) + 1
            if orig_len >= L:
                msk = np.ones(L, dtype=np.uint8)
            else:
                pad_left = (L - orig_len)//2
                msk = np.zeros(L, dtype=np.uint8); msk[pad_left:pad_left+orig_len] = 1
        else:
            msk = np.ones(L, dtype=np.uint8)

        name = os.path.basename(p)
        return (torch.from_numpy(sig).unsqueeze(0),  # [1, L]
                torch.from_numpy(lab),
                torch.tensor(rid, dtype=torch.long),
                torch.from_numpy(msk),
                name)

# --------------------------
# Model (BN/dropout configurable)
# --------------------------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, p_drop=0.0):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn: layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        if p_drop > 0: layers.append(nn.Dropout(p_drop))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNet1D_MTL(nn.Module):
    def __init__(self, in_ecg=1, K=len(RHYTHMS), use_bn=False, p_drop=0.0):
        super().__init__()
        self.K = K
        in_total = in_ecg + K
        self.enc1 = ConvBlock1D(in_total, 32, use_bn, p_drop)
        self.pool = nn.MaxPool1d(2,2)
        self.enc2 = ConvBlock1D(32, 64, use_bn, p_drop)
        self.enc3 = ConvBlock1D(64,128, use_bn, p_drop)

        self.up1  = nn.ConvTranspose1d(128,64,2,2)
        self.dec1 = ConvBlock1D(64+64, 64, use_bn, p_drop)
        self.up2  = nn.ConvTranspose1d(64,32,2,2)
        self.dec2 = ConvBlock1D(32+32, 32, use_bn, p_drop)
        self.out  = nn.Conv1d(32, 4, kernel_size=1)

        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_fc   = nn.Linear(128, K)

        for m in self.modules():
            if isinstance(m,(nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m,'bias',None) is not None: nn.init.zeros_(m.bias)

    def forward(self, x_ecg, cond_vec):
        B, _, L = x_ecg.shape
        cond = cond_vec.unsqueeze(-1).expand(-1, -1, L)
        x = torch.cat([x_ecg, cond], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        z = self.cls_pool(e3).squeeze(-1)
        cls_logits = self.cls_fc(z)

        d1 = self.up1(e3)
        e2a = F.interpolate(e2, size=d1.size(2), mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2a], dim=1))
        d2 = self.up2(d1)
        e1a = F.interpolate(e1, size=d2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1a], dim=1))
        seg_logits = self.out(d2)
        return seg_logits, cls_logits

# --------------------------
# Mandatory post-processing
#   1) Short blob fix (< min_len): merge into surround if both sides agree; else -> bg
#   2) Between consecutive QRS: keep only the longest P and longest T
# --------------------------
def _runs_of_class(labels):
    L = len(labels); runs=[]
    if L==0: return runs
    s=0; cur=labels[0]
    for i in range(1,L):
        if labels[i]!=cur:
            runs.append((s,i,cur)); s=i; cur=labels[i]
    runs.append((s,L,cur))
    return runs

def _apply_short_blob_fix(y, min_len):
    L=len(y); 
    for (s,e,c) in _runs_of_class(y.copy()):
        if c==CLASS_BG: continue
        if e-s < min_len:
            prev_c = y[s-1] if s>0 else None
            next_c = y[e]   if e<L else None
            if prev_c is not None and next_c is not None and prev_c==next_c and prev_c!=c:
                y[s:e]=prev_c
            else:
                y[s:e]=CLASS_BG
    return y

def _enforce_one_P_T_between_QRS(y):
    runs=_runs_of_class(y); qrs=[(s,e) for (s,e,c) in runs if c==CLASS_QRS]
    if len(qrs)<2: return y
    for i in range(len(qrs)-1):
        Lend = qrs[i][1]; Rstart = qrs[i+1][0]
        if Rstart<=Lend: continue
        seg=_runs_of_class(y[Lend:Rstart])
        seg=[(Lend+s, Lend+e, c) for (s,e,c) in seg]
        p_runs=[(s,e) for (s,e,c) in seg if c==CLASS_P]
        if len(p_runs)>1:
            lengths=[e-s for (s,e) in p_runs]; keep=int(np.argmax(lengths))
            for j,(s,e) in enumerate(p_runs):
                if j!=keep: y[s:e]=CLASS_BG
        t_runs=[(s,e) for (s,e,c) in seg if c==CLASS_T]
        if len(t_runs)>1:
            lengths=[e-s for (s,e) in t_runs]; keep=int(np.argmax(lengths))
            for j,(s,e) in enumerate(t_runs):
                if j!=keep: y[s:e]=CLASS_BG
    return y

def post_process(y_pred_1d, fs, min_len_ms=40):
    y=y_pred_1d.copy()
    min_len = max(1, int(round(min_len_ms*fs/1000.0)))
    y=_apply_short_blob_fix(y, min_len)
    y=_enforce_one_P_T_between_QRS(y)
    return y

# --------------------------
# Boundary extraction & matching
# --------------------------
def extract_boundaries(labels):
    L=len(labels); out={bt[0]:[] for bt in BOUNDARY_TYPES}
    # onsets
    prev = CLASS_BG
    for i in range(L):
        cur=labels[i]
        if cur!=prev:
            # previous class ended at i-1; current class starts at i
            if cur==CLASS_P: out["P_on"].append(i)
            if cur==CLASS_QRS: out["QRS_on"].append(i)
            if cur==CLASS_T: out["T_on"].append(i)
        prev=cur
    # offsets
    for i in range(L-1):
        cur, nxt = labels[i], labels[i+1]
        if cur!=nxt:
            if cur==CLASS_P: out["P_off"].append(i)
            if cur==CLASS_QRS: out["QRS_off"].append(i)
            if cur==CLASS_T: out["T_off"].append(i)
    if L>0:
        # tail end
        if labels[-1]==CLASS_P: out["P_off"].append(L-1)
        if labels[-1]==CLASS_QRS: out["QRS_off"].append(L-1)
        if labels[-1]==CLASS_T: out["T_off"].append(L-1)
    return out

def match_events(gt_idxs, pr_idxs, tol_samples):
    gt = list(gt_idxs); pr = list(pr_idxs)
    taken=[False]*len(gt)
    TP=0; errs=[]
    for p in pr:
        best=None; best_j=-1
        for j,g in enumerate(gt):
            if taken[j]: continue
            d=abs(p-g)
            if d<=tol_samples and (best is None or d<best):
                best=d; best_j=j
        if best_j>=0:
            TP+=1; taken[best_j]=True; errs.append(p-gt[best_j])
    FP = len(pr)-TP
    FN = len(gt)-TP
    return TP,FP,FN,errs

# --------------------------
# Metrics accumulator
# --------------------------
class BoundaryStats:
    def __init__(self):
        self.stats = {r:{bn:{'TP':0,'FP':0,'FN':0,'errs':[]} for (bn,_,_) in BOUNDARY_TYPES}
                      for r in range(len(RHYTHMS))}
    def add(self, r, bn, TP, FP, FN, errs):
        s = self.stats[r][bn]; s['TP']+=TP; s['FP']+=FP; s['FN']+=FN; s['errs'].extend(errs)
    def finalize(self, fs):
        # returns dict[rhythm][boundary] = (Se, PPV, F1, mu_ms, sd_ms)
        out={}
        for r in range(len(RHYTHMS)):
            out[RHYTHMS[r]]={}
            for (bn,_,_) in BOUNDARY_TYPES:
                d=self.stats[r][bn]
                TP,FP,FN = d['TP'], d['FP'], d['FN']
                se = TP/(TP+FN) if (TP+FN)>0 else 0.0
                ppv= TP/(TP+FP) if (TP+FP)>0 else 0.0
                f1 = 2*se*ppv/(se+ppv) if (se+ppv)>0 else 0.0
                if len(d['errs'])>0:
                    errs_ms=(np.array(d['errs'])*(1000.0/fs))
                    mu=float(np.mean(errs_ms)); sd=float(np.std(errs_ms))
                else:
                    mu=0.0; sd=0.0
                out[RHYTHMS[r]][bn]=(se,ppv,f1,mu,sd)
        # macro row (unweighted mean across rhythms)
        out['macro']={}
        for (bn,_,_) in BOUNDARY_TYPES:
            vals=[out[RHYTHMS[r]][bn] for r in range(len(RHYTHMS))]
            if len(vals)==0:
                out['macro'][bn]=(0,0,0,0,0); continue
            se=float(np.mean([v[0] for v in vals]))
            ppv=float(np.mean([v[1] for v in vals]))
            f1=float(np.mean([v[2] for v in vals]))
            mu=float(np.mean([v[3] for v in vals]))
            sd=float(np.mean([v[4] for v in vals]))
            out['macro'][bn]=(se,ppv,f1,mu,sd)
        return out

# --------------------------
# Eval loop
# --------------------------
def run_eval(model, loader, fs, tol_ms, min_len_ms, device, cond_mode):
    model.eval()
    tol_samp = int(round(tol_ms*fs/1000.0))
    stats = BoundaryStats()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for sig, lab, rid, msk, _ in tqdm(loader, desc="Eval", disable=False):
            sig = sig.to(device); lab=lab.to(device); rid=rid.to(device); msk=msk.to(device).bool()
            if cond_mode=='gt':
                cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float().to(device)
            else:
                cond = torch.zeros((sig.size(0), len(RHYTHMS)), device=device)

            logits, _ = model(sig, cond)
            pred = torch.argmax(logits, dim=1)  # [B,L]

            for b in range(sig.size(0)):
                valid = msk[b].nonzero(as_tuple=True)[0]
                if valid.numel()==0: continue
                s0, s1 = valid[0].item(), valid[-1].item()+1
                gt_ = lab[b, s0:s1].cpu().numpy()
                pr_ = pred[b, s0:s1].cpu().numpy()
                pr_pp = post_process(pr_, fs, min_len_ms=min_len_ms)

                gtB = extract_boundaries(gt_)
                prB = extract_boundaries(pr_pp)
                r = rid[b].item()
                for (bn,_,_) in BOUNDARY_TYPES:
                    TP,FP,FN,errs = match_events(gtB[bn], prB[bn], tol_samp)
                    stats.add(r, bn, TP, FP, FN, errs)
    return stats

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate boundary metrics after post-processing (per rhythm × boundary)")
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', choices=['train','val','test'], default='test')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--fs', type=int, default=500)
    ap.add_argument('--boundary_tol_ms', type=int, default=150)
    ap.add_argument('--min_len_ms', type=int, default=40)
    ap.add_argument('--use_batchnorm', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--cond', choices=['gt','zero'], default='gt', help='conditioning during eval')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load split
    with open(os.path.join(args.split_dir, f"{args.split}_records.txt"), 'r') as f:
        records = [ln.strip() for ln in f if ln.strip()]

    ds = ECGSegments(records, args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers>0))

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D_MTL(use_bn=args.use_batchnorm, p_drop=args.dropout).to(device)
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd, strict=True)

    # Evaluate
    stats = run_eval(model, loader, fs=args.fs, tol_ms=args.boundary_tol_ms,
                     min_len_ms=args.min_len_ms, device=device, cond_mode=args.cond)
    table = stats.finalize(args.fs)

    # Write CSV: rows = rhythms (+ macro), cols = for each boundary: Se/PPV/F1/mu/sd
    csv_path = os.path.join(args.out_dir, f"boundary_metrics_{args.split}.csv")
    with open(csv_path, "w") as f:
        cols = ["rhythm"]
        for (bn,_,_) in BOUNDARY_TYPES:
            cols += [f"{bn}_Se", f"{bn}_PPV", f"{bn}_F1", f"{bn}_mu_ms", f"{bn}_sd_ms"]
        f.write(",".join(cols) + "\n")
        for r in RHYTHMS + ['macro']:
            row = [r]
            for (bn,_,_) in BOUNDARY_TYPES:
                se,ppv,f1,mu,sd = table[r][bn]
                row += [f"{se:.4f}", f"{ppv:.4f}", f"{f1:.4f}", f"{mu:.2f}", f"{sd:.2f}"]
            f.write(",".join(row) + "\n")

    # Also write a compact summary JSON with macro means across boundaries
    macro_f1 = float(np.mean([table['macro'][bn][2] for (bn,_,_) in BOUNDARY_TYPES]))
    summary = {
        "split": args.split,
        "fs": args.fs,
        "boundary_tol_ms": args.boundary_tol_ms,
        "min_len_ms": args.min_len_ms,
        "macro_boundary_F1": round(macro_f1, 4),
        "notes": "All metrics computed after mandatory post-processing."
    }
    with open(os.path.join(args.out_dir, f"boundary_summary_{args.split}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not args.quiet:
        print("Wrote:", csv_path)
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
