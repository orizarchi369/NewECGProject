import os, re, json, time, math, argparse, warnings
from collections import defaultdict, namedtuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Config / constants
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
    rec = base.split('_ii_')[0]  # e.g., AF0001
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

        # valid mask from meta['orig_start/end']; fallback all ones
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
# Model (same topology; BN/dropout configurable)
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

        # rhythm head (kept for compatibility; λ may be 0)
        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_fc   = nn.Linear(128, K)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None: nn.init.zeros_(m.bias)

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
# Losses (default: CE+Dice)
# --------------------------
def ce_loss_masked(logits, targets, mask, class_weights=None):
    # logits: [B,C,L], targets: [B,L], mask: [B,L] in {0,1}
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')  # [B,L]
    ce = ce * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return ce.sum() / denom

def dice_loss_multiclass_masked(logits, targets, mask, exclude_bg=True, eps=1e-6):
    # softmax probs
    probs = F.softmax(logits, dim=1)  # [B,C,L]
    B, C, L = probs.shape
    num_classes = C
    if exclude_bg:
        cls_range = range(1, num_classes)
    else:
        cls_range = range(0, num_classes)
    # one-hot targets
    tgt_oh = F.one_hot(targets.clamp_min(0), num_classes=num_classes).permute(0,2,1).float()  # [B,C,L]
    mask_f = mask.float().unsqueeze(1)  # [B,1,L]
    probs = probs * mask_f
    tgt_oh = tgt_oh * mask_f
    dices = []
    for c in cls_range:
        p = probs[:, c, :]
        t = tgt_oh[:, c, :]
        num = 2.0 * (p * t).sum(dim=(0,1))
        den = (p + t).sum(dim=(0,1)).clamp_min(eps)
        dices.append(1.0 - (num / den))  # dice loss for class c
    return torch.stack(dices).mean()

# --------------------------
# Post-processing (MANDATORY, eval-only)
#   1) Short blob fix (< min_len): merge to surrounding class if both sides agree; else -> bg
#   2) Between consecutive QRS: keep only the longest P and longest T
# --------------------------
def _runs_of_class(labels):
    """Return list of (start, end_exclusive, cls) runs across labels."""
    L = len(labels)
    runs = []
    if L == 0: return runs
    s = 0; cur = labels[0]
    for i in range(1, L):
        if labels[i] != cur:
            runs.append((s, i, cur))
            s = i; cur = labels[i]
    runs.append((s, L, cur))
    return runs

def _apply_short_blob_fix(y, min_len):
    L = len(y)
    if L == 0: return y
    runs = _runs_of_class(y)
    for (s,e,c) in runs:
        if c == CLASS_BG: continue
        if e - s < min_len:
            prev_c = y[s-1] if s > 0 else None
            next_c = y[e]   if e < L else None
            if prev_c is not None and next_c is not None and prev_c == next_c and prev_c != c:
                # interrupting a longer region -> merge into surrounding class
                y[s:e] = prev_c
            else:
                # isolated short blob -> drop to background
                y[s:e] = CLASS_BG
    return y

def _enforce_one_P_T_between_QRS(y):
    runs = _runs_of_class(y)
    qrs_runs = [(s,e,c) for (s,e,c) in runs if c == CLASS_QRS]
    if len(qrs_runs) < 2:  # nothing to enforce between QRS
        return y
    for i in range(len(qrs_runs)-1):
        left_end = qrs_runs[i][1]
        right_start = qrs_runs[i+1][0]
        if right_start <= left_end:  # overlapping QRS (unlikely after short-blob fix)
            continue
        # collect P and T runs strictly within (left_end, right_start)
        seg_runs = _runs_of_class(y[left_end:right_start])
        # adjust coordinates to absolute
        seg_runs = [(left_end+s, left_end+e, c) for (s,e,c) in seg_runs]
        # keep only longest P
        p_runs = [(s,e) for (s,e,c) in seg_runs if c == CLASS_P]
        if len(p_runs) > 1:
            lengths = [e-s for (s,e) in p_runs]
            keep_idx = int(np.argmax(lengths))
            for j,(s,e) in enumerate(p_runs):
                if j != keep_idx:
                    y[s:e] = CLASS_BG
        # keep only longest T
        t_runs = [(s,e) for (s,e,c) in seg_runs if c == CLASS_T]
        if len(t_runs) > 1:
            lengths = [e-s for (s,e) in t_runs]
            keep_idx = int(np.argmax(lengths))
            for j,(s,e) in enumerate(t_runs):
                if j != keep_idx:
                    y[s:e] = CLASS_BG
    return y

def post_process(y_pred_1d, fs, min_len_ms=40):
    """Apply mandatory post-processing rules to a 1D label array."""
    y = y_pred_1d.copy()
    min_len = max(1, int(round(min_len_ms * fs / 1000.0)))  # samples
    y = _apply_short_blob_fix(y, min_len)
    y = _enforce_one_P_T_between_QRS(y)
    return y

# --------------------------
# Boundary extraction & matching
# --------------------------
def extract_boundaries(labels):
    """Return dict: {'P_on': [idx...], 'P_off': [...], ...} from label array."""
    L = len(labels)
    d = {bt[0]: [] for bt in BOUNDARY_TYPES}
    for cname, c, side in BOUNDARY_TYPES:
        if side == "on":
            # onset: position i where labels[i]==c and (i==0 or labels[i-1]!=c)
            idxs = []
            prev = CLASS_BG
            for i in range(L):
                cur = labels[i]
                if cur == c and prev != c:
                    idxs.append(i)
                prev = cur
            d[cname] = idxs
        else:
            # offset: position i where labels[i]==c and (i==L-1 or labels[i+1]!=c)
            idxs = []
            nextv = CLASS_BG
            for i in range(L-1, -1, -1):
                cur = labels[i]
                if cur == c:
                    if i == L-1 or labels[i+1] != c:
                        idxs.append(i)
                nextv = cur
            d[cname] = sorted(idxs)
    return d

def match_events(gt_idxs, pr_idxs, tol_samples):
    """Greedy one-to-one matching within ±tol; returns (TP, FP, FN, errors_ms_list (signed in samples))."""
    gt_idxs = list(gt_idxs); pr_idxs = list(pr_idxs)
    gt_taken = [False]*len(gt_idxs)
    TP = 0; errors = []
    for p in pr_idxs:
        # find nearest unmatched gt within tol
        best_j = -1; best_dist = None
        for j,g in enumerate(gt_idxs):
            if gt_taken[j]: continue
            dist = abs(p - g)
            if dist <= tol_samples and (best_dist is None or dist < best_dist):
                best_dist = dist; best_j = j
        if best_j >= 0:
            TP += 1
            gt_taken[best_j] = True
            errors.append(p - gt_idxs[best_j])  # signed (samples)
    FP = len(pr_idxs) - TP
    FN = len(gt_idxs) - TP
    return TP, FP, FN, errors

# --------------------------
# Metrics accumulator (per boundary × per rhythm)
# --------------------------
class BoundaryStats:
    def __init__(self):
        # dict[rhythm_id][boundary_name] -> {'TP':int,'FP':int,'FN':int,'errs':list}
        self.stats = {r:{bt[0]:{'TP':0,'FP':0,'FN':0,'errs':[]} for bt in BOUNDARY_TYPES}
                      for r in range(len(RHYTHMS))}
    def add(self, rhythm_id, boundary_name, TP, FP, FN, errs):
        s = self.stats[rhythm_id][boundary_name]
        s['TP'] += TP; s['FP'] += FP; s['FN'] += FN; s['errs'].extend(errs)
    def finalize(self, fs, tol_ms):
        # produce per-rhythm table with Se, PPV, F1, mean±sd(ms)
        def _row_from_counts(d):
            TP, FP, FN = d['TP'], d['FP'], d['FN']
            se  = TP / (TP + FN) if (TP+FN)>0 else 0.0
            ppv = TP / (TP + FP) if (TP+FP)>0 else 0.0
            f1  = 2*se*ppv/(se+ppv) if (se+ppv)>0 else 0.0
            if len(d['errs'])>0:
                errs_ms = (np.array(d['errs']) * (1000.0/fs)).tolist()
                mu = float(np.mean(errs_ms)); sd = float(np.std(errs_ms))
            else:
                mu = 0.0; sd = 0.0
            return se, ppv, f1, mu, sd
        # build nested dict
        out = {}
        for r in range(len(RHYTHMS)):
            out[RHYTHMS[r]] = {}
            for (bn,_,_) in BOUNDARY_TYPES:
                out[RHYTHMS[r]][bn] = _row_from_counts(self.stats[r][bn])
        # macro (unweighted mean across boundary types)
        out['macro'] = {}
        for (bn,_,_) in BOUNDARY_TYPES:
            vals = [out[RHYTHMS[r]][bn] for r in range(len(RHYTHMS))]
            if len(vals)==0: out['macro'][bn] = (0,0,0,0,0); continue
            se = float(np.mean([v[0] for v in vals]))
            ppv= float(np.mean([v[1] for v in vals]))
            f1 = float(np.mean([v[2] for v in vals]))
            mu = float(np.mean([v[3] for v in vals]))
            sd = float(np.mean([v[4] for v in vals]))
            out['macro'][bn] = (se,ppv,f1,mu,sd)
        return out

# --------------------------
# Class-weight computation (on train, masked)
# --------------------------
def compute_class_weights(dataset):
    counts = np.zeros(4, dtype=np.float64)
    for i in range(len(dataset)):
        _, lab, _, msk, _ = dataset[i]
        lab = lab.numpy(); msk = msk.numpy().astype(bool)
        for c in range(4):
            counts[c] += np.sum((lab==c) & msk)
    # inverse frequency (avoid zero)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * 4.0  # normalize approx to mean 1
    return torch.tensor(w, dtype=torch.float32)

# --------------------------
# Training / Validation
# --------------------------
def evaluate_val(model, loader, fs, boundary_tol_ms, min_len_ms, device, use_bn, p_drop, cond_mode="gt"):
    """Return BoundaryStats (with post-processing), and macro-F1 across boundary types."""
    model.eval()
    tol_samp = int(round(boundary_tol_ms * fs / 1000.0))
    stats = BoundaryStats()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for sig, lab, rid, msk, _ in loader:
            sig = sig.to(device); lab = lab.to(device); rid = rid.to(device); msk = msk.to(device).bool()
            if cond_mode == "gt":
                cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float().to(device)
            else:
                cond = torch.zeros((sig.size(0), len(RHYTHMS)), device=device)

            logits, _ = model(sig, cond)
            pred = torch.argmax(logits, dim=1)  # [B, L]
            # apply mask before post-proc (keep only valid portion)
            for b in range(sig.size(0)):
                valid_idx = msk[b].nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    continue
                s0, s1 = valid_idx[0].item(), valid_idx[-1].item()+1
                gt_ = lab[b, s0:s1].cpu().numpy()
                pr_ = pred[b, s0:s1].cpu().numpy()
                # mandatory post-processing
                pr_pp = post_process(pr_, fs, min_len_ms=min_len_ms)
                # extract boundaries and accumulate
                gtB = extract_boundaries(gt_)
                prB = extract_boundaries(pr_pp)
                r = rid[b].item()
                for (bn, _, _) in BOUNDARY_TYPES:
                    TP, FP, FN, errs = match_events(gtB[bn], prB[bn], tol_samp)
                    stats.add(r, bn, TP, FP, FN, errs)
    # macro-F1 across boundary types and rhythms (unweighted mean)
    table = stats.finalize(fs, boundary_tol_ms)
    f1_macro_types = []
    for (bn,_,_) in BOUNDARY_TYPES:
        f1_macro_types.append(table['macro'][bn][2])  # index 2 is F1
    macro_f1 = float(np.mean(f1_macro_types)) if len(f1_macro_types) else 0.0
    return stats, table, macro_f1

def main():
    ap = argparse.ArgumentParser(description="Train 1D U-Net (boundary metrics, mandatory post-processing)")
    # paths / basic
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    # model
    ap.add_argument('--use_batchnorm', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--lambda_cls', type=float, default=0.0)  # default off, head kept for compat
    # loss
    ap.add_argument('--seg_loss', choices=['dicece','ce','dice'], default='dicece')
    ap.add_argument('--dice_exclude_bg', action='store_true', default=True)
    ap.add_argument('--no_class_weights', action='store_true')
    # scheduler
    ap.add_argument('--lr_sched', choices=['cosine','plateau'], default='cosine')
    ap.add_argument('--t0', type=int, default=10)
    ap.add_argument('--t_mult', type=int, default=2)
    ap.add_argument('--eta_min', type=float, default=1e-6)
    ap.add_argument('--plateau_patience', type=int, default=8)
    ap.add_argument('--plateau_factor', type=float, default=0.5)
    # evaluation / post-proc
    ap.add_argument('--fs', type=int, default=500)
    ap.add_argument('--boundary_tol_ms', type=int, default=150)  # Joung-style
    ap.add_argument('--min_len_ms', type=int, default=40)
    ap.add_argument('--cond', choices=['gt','zero'], default='gt', help='validation conditioning')
    # misc
    ap.add_argument('--checkpoint_interval', type=int, default=25)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--suppress_warnings', action='store_true')
    args = ap.parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # splits
    splits = {}
    for sp in ['train','val','test']:
        with open(os.path.join(args.split_dir, f'{sp}_records.txt'), 'r') as f:
            splits[sp] = [ln.strip() for ln in f if ln.strip()]

    # datasets / loaders
    ds_train = ECGSegments(splits['train'], args.data_dir)
    ds_val   = ECGSegments(splits['val'],   args.data_dir)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(ds_train).to(device)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers>0))

    # model
    model = UNet1D_MTL(use_bn=args.use_batchnorm, p_drop=args.dropout).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_sched == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=args.plateau_factor, patience=args.plateau_patience)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    # bookkeeping
    best_metric = -1.0
    best_path = os.path.join(args.output_dir, "best_unet_mtl_boundary.pth")
    last_path = os.path.join(args.output_dir, "last_unet_mtl_boundary.pth")
    log_csv = os.path.join(args.output_dir, "train_log.csv")
    with open(log_csv, "w") as f:
        f.write("epoch,lr,train_loss,macro_boundary_f1\n")

    for epoch in range(1, args.num_epochs+1):
        # ---- Train ----
        model.train()
        ep_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", disable=args.quiet)
        for sig, lab, rid, msk, _ in pbar:
            sig, lab, rid, msk = sig.to(device), lab.to(device), rid.to(device), msk.to(device).bool()
            cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float()

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                seg_logits, cls_logits = model(sig, cond)
                # masked CE
                loss_ce = ce_loss_masked(seg_logits, lab, msk, class_weights=class_weights)
                # masked Dice
                loss_dice = dice_loss_multiclass_masked(seg_logits, lab, msk, exclude_bg=args.dice_exclude_bg)
                if args.seg_loss == 'dicece':
                    seg_loss = loss_ce + loss_dice
                elif args.seg_loss == 'ce':
                    seg_loss = loss_ce
                else:  # 'dice'
                    seg_loss = loss_dice
                # rhythm head (likely 0.0 unless user sets otherwise)
                cls_loss = F.cross_entropy(cls_logits, rid)
                loss = seg_loss + args.lambda_cls * cls_loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()

            ep_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        ep_loss /= max(1, len(train_loader))

        # ---- Val (boundary metrics AFTER mandatory post-proc) ----
        stats, table, macro_f1 = evaluate_val(
            model, val_loader, fs=args.fs,
            boundary_tol_ms=args.boundary_tol_ms,
            min_len_ms=args.min_len_ms,
            device=device, use_bn=args.use_batchnorm, p_drop=args.dropout,
            cond_mode=args.cond
        )

        # step scheduler
        if args.lr_sched == 'cosine':
            sched.step(epoch-1)  # warm restarts keyed by epoch count
        else:
            sched.step(macro_f1)

        # save periodic checkpoint
        if (epoch % args.checkpoint_interval) == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))

        # save last
        torch.save(model.state_dict(), last_path)

        # track best on boundary macro-F1
        if macro_f1 > best_metric:
            best_metric = macro_f1
            torch.save(model.state_dict(), best_path)

        # log
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{opt.param_groups[0]['lr']:.6g},{ep_loss:.6f},{macro_f1:.6f}\n")

        if not args.quiet:
            print(f"Epoch {epoch}/{args.num_epochs} | LR {opt.param_groups[0]['lr']:.2e} | Train {ep_loss:.4f} | Val boundary macro-F1 {macro_f1:.4f}")

        # also dump per-boundary × per-rhythm CSV each epoch (small)
        per_epoch_dir = os.path.join(args.output_dir, "val_metrics")
        os.makedirs(per_epoch_dir, exist_ok=True)
        csv_path = os.path.join(per_epoch_dir, f"val_boundary_metrics_epoch_{epoch}.csv")
        # table structure: table[rhythm][boundary] = (se, ppv, f1, mu, sd)
        rhythms_out = RHYTHMS + ['macro']
        with open(csv_path, "w") as f:
            # header
            cols = ["rhythm"]
            for (bn,_,_) in BOUNDARY_TYPES:
                cols += [f"{bn}_Se", f"{bn}_PPV", f"{bn}_F1", f"{bn}_mu_ms", f"{bn}_sd_ms"]
            f.write(",".join(cols) + "\n")
            for r in rhythms_out:
                row = [r]
                for (bn,_,_) in BOUNDARY_TYPES:
                    se,ppv,f1,mu,sd = table[r][bn]
                    row += [f"{se:.4f}", f"{ppv:.4f}", f"{f1:.4f}", f"{mu:.2f}", f"{sd:.2f}"]
                f.write(",".join(row) + "\n")

    if not args.quiet:
        print("Done. Best boundary macro-F1:", f"{best_metric:.4f}")
        print("Saved:", best_path, "and", last_path)

if __name__ == "__main__":
    main()
