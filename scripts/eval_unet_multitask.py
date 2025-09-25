# scripts/eval_unet_multitask.py
import os, re, json, csv, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']  # consistent with training

# ---------- Args ----------
ap = argparse.ArgumentParser(description="Evaluate rhythm-aware U-Net (segment-level)")
ap.add_argument('--data_dir', required=True)
ap.add_argument('--split_dir', required=True)
ap.add_argument('--model_path', required=True, help='Path to best/last .pth')
ap.add_argument('--out_dir', required=True, help='Folder to write eval outputs')
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--num_workers', type=int, default=2)
ap.add_argument('--tol_samples', type=int, default=6, help='±N for tolerant F1 (P/QRS/T)')
ap.add_argument('--report_unmasked_strict_wF1', action='store_true',
                help='Also compute strict weighted F1 without masking (for apples-to-apples with old runs)')
ap.add_argument('--quiet', action='store_true')
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ---------- Dataset (mask-from-meta) ----------
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
        for f in os.scandir(data_dir):
            if f.is_file() and f.name.endswith('.npz') and lead_filter in f.name:
                rec = f.name.split('_ii_')[0]
                if rec in record_ids:
                    _, rstr = parse_record_and_rhythm_from_name(f.path)
                    rid = self.rhythm_to_id.get(rstr, 0)
                    self.files.append(f.path)
                    self.rhythm_ids.append(rid)
        order = np.argsort(self.files)
        self.files = [self.files[i] for i in order]
        self.rhythm_ids = [self.rhythm_ids[i] for i in order]

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        npz = np.load(p, allow_pickle=True)
        sig = npz['signal'].astype(np.float32)     # [512]
        lab = npz['labels'].astype(np.int64)       # [512]
        rhythm_idx = self.rhythm_ids[idx]

        # build valid mask from meta['orig_start']/['orig_end']; fallback = all ones
        L = lab.shape[0]
        meta = npz['meta'].item() if 'meta' in npz else {}
        if 'orig_start' in meta and 'orig_end' in meta:
            orig_len = int(meta['orig_end']) - int(meta['orig_start']) + 1
            if orig_len >= L:
                msk = np.ones(L, dtype=np.uint8)
            else:
                pad_left = (L - orig_len) // 2
                msk = np.zeros(L, dtype=np.uint8)
                msk[pad_left:pad_left + orig_len] = 1
        else:
            msk = np.ones(L, dtype=np.uint8)

        return (torch.from_numpy(sig).unsqueeze(0),   # [1,512]
                torch.from_numpy(lab),                # [512]
                torch.tensor(rhythm_idx, dtype=torch.long),
                torch.from_numpy(msk),                # [512]
                os.path.basename(p))

# ---------- Model (match training) ----------
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, p_drop=0.0):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)]
        if use_bn: layers += [nn.BatchNorm1d(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        if p_drop > 0: layers += [nn.Dropout(p_drop)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNet1D_MTL(nn.Module):
    def __init__(self, in_ecg_channels=1, num_seg_classes=4, num_rhythms=len(RHYTHMS),
                 use_bn=False, p_drop=0.0):
        super().__init__()
        self.K = num_rhythms
        in_total = in_ecg_channels + self.K

        self.enc1 = ConvBlock1D(in_total, 32, use_bn, p_drop)
        self.pool = nn.MaxPool1d(2,2)
        self.enc2 = ConvBlock1D(32, 64, use_bn, p_drop)
        self.enc3 = ConvBlock1D(64, 128, use_bn, p_drop)

        self.up1  = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(64+64, 64, use_bn, p_drop)
        self.up2  = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(32+32, 32, use_bn, p_drop)
        self.out  = nn.Conv1d(32, 4, kernel_size=1)

        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_fc   = nn.Linear(128, self.K)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_ecg, cond_vec=None):
        B, _, L = x_ecg.shape
        if cond_vec is None:
            cond = x_ecg.new_zeros((B, self.K))
        else:
            cond = cond_vec
        cond_tiled = cond.unsqueeze(-1).expand(-1, -1, L)
        x = torch.cat([x_ecg, cond_tiled], dim=1)

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

# ---------- Metrics ----------
def _flatten_masked(arr_list):
    return np.concatenate(arr_list) if len(arr_list) else np.array([], dtype=np.int64)

def strict_scores(seg_preds, seg_labels, num_classes=4, macro_classes=(1,2,3)):
    y_pred = _flatten_masked(seg_preds)
    y_true = _flatten_masked(seg_labels)
    if y_true.size == 0:
        return 0.0, 0.0, np.zeros(num_classes)
    f1w = f1_score(y_true, y_pred, average='weighted', labels=list(range(num_classes)))
    per = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    valid = [c for c in macro_classes if (y_true == c).any()]
    f1m = float(np.mean([per[c] for c in valid])) if len(valid) else 0.0
    return f1w, f1m, per

def dilate_bool_mask(mask_t, N):
    if N <= 0: return mask_t
    x = mask_t.unsqueeze(1).float()
    y = F.max_pool1d(x, kernel_size=2*N+1, stride=1, padding=N)
    return (y > 0).squeeze(1)

def tolerant_macro_f1(seg_preds, seg_labels, classes=(1,2,3), N=6):
    preds = torch.tensor(_flatten_masked(seg_preds), dtype=torch.long)
    gts   = torch.tensor(_flatten_masked(seg_labels), dtype=torch.long)
    if preds.numel() == 0: return 0.0, np.zeros(max(classes)+1)
    B, L = 1, preds.numel()
    preds = preds.view(B, L); gts = gts.view(B, L)
    f1s, per_cls = [], np.zeros(max(classes)+1)
    eps = 1e-8
    for c in classes:
        p = (preds == c); g = (gts == c)
        if p.sum() == 0 and g.sum() == 0: 
            per_cls[c] = 0.0
            continue
        dil_g = dilate_bool_mask(g, N)
        dil_p = dilate_bool_mask(p, N)
        tp_prec = (p & dil_g).sum().item()
        tp_rec  = (g & dil_p).sum().item()
        prec = tp_prec / max(1, p.sum().item())
        rec  = tp_rec  / max(1, g.sum().item())
        f1 = 0.0 if (prec+rec) < eps else (2*prec*rec)/(prec+rec+eps)
        f1s.append(f1); per_cls[c] = f1
    f1m = float(np.mean(f1s)) if len(f1s) else 0.0
    return f1m, per_cls

# ---------- Eval ----------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load splits
    splits = {}
    for sp in ['train','val','test']:
        with open(os.path.join(args.split_dir, f'{sp}_records.txt'), 'r') as f:
            splits[sp] = [ln.strip() for ln in f if ln.strip()]

    # Dataset/Loader (test only)
    ds_test = ECGSegments(splits['test'], args.data_dir)
    loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=(args.num_workers>0))

    # Model
    model = UNet1D_MTL()
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    # Accumulators (overall + per-rhythm)
    K = len(RHYTHMS)
    per_rhythm_pred_strict = [[] for _ in range(K)]
    per_rhythm_true_strict = [[] for _ in range(K)]
    per_rhythm_pred_unmask = [[] for _ in range(K)]
    per_rhythm_true_unmask = [[] for _ in range(K)]
    overall_pred_strict, overall_true_strict = [], []
    overall_pred_unmask, overall_true_unmask = [], []

    # Classifier accuracy (just for curiosity; we condition seg on GT rhythm)
    cls_preds_all, cls_labels_all = [], []

    pbar = tqdm(loader, desc='Eval', disable=args.quiet)
    with torch.no_grad():
        for sig, lab, rid, msk, names in pbar:
            sig, lab, rid, msk = sig.to(device), lab.to(device), rid.to(device), msk.to(device).bool()

            # hard one-hot GT rhythm for segmentation
            cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float()
            seg_logits, cls_logits = model(sig, cond)
            preds = torch.argmax(seg_logits, dim=1)  # [B, L]

            # STRICT (masked by valid)
            for p, y, m, r in zip(preds, lab, msk, rid):
                m = m.bool()
                overall_pred_strict.append(p[m].cpu().numpy())
                overall_true_strict.append(y[m].cpu().numpy())
                per_rhythm_pred_strict[r.item()].append(p[m].cpu().numpy())
                per_rhythm_true_strict[r.item()].append(y[m].cpu().numpy())

            # OPTIONAL: UNMASKED strict (treat all 512 valid) for apples-to-apples with old runs
            if args.report_unmasked_strict_wF1:
                for p, y, r in zip(preds, lab, rid):
                    overall_pred_unmask.append(p.cpu().numpy())
                    overall_true_unmask.append(y.cpu().numpy())
                    per_rhythm_pred_unmask[r.item()].append(p.cpu().numpy())
                    per_rhythm_true_unmask[r.item()].append(y.cpu().numpy())

            # classifier eval (zero conditioning, no leak)
            zero_cond = torch.zeros((sig.size(0), len(RHYTHMS)), device=device)
            _, cls_logits0 = model(sig, zero_cond)
            cls_pred = torch.argmax(cls_logits0, dim=1)
            cls_preds_all.append(cls_pred.cpu().numpy()); cls_labels_all.append(rid.cpu().numpy())

    # Overall strict
    f1w_strict, f1m_strict, per_strict = strict_scores(overall_pred_strict, overall_true_strict)
    # Overall tolerant macro-F1
    f1m_tol, per_tol = tolerant_macro_f1(overall_pred_strict, overall_true_strict, classes=(1,2,3), N=args.tol_samples)
    # Optional overall unmasked strict weighted F1
    f1w_unmasked = None
    if args.report_unmasked_strict_wF1:
        f1w_unmasked, _, _ = strict_scores(overall_pred_unmask, overall_true_unmask)

    # Per-rhythm tables
    per_rhythm_rows = []
    for r_id, r_name in enumerate(RHYTHMS):
        fw, fm, pc = strict_scores(per_rhythm_pred_strict[r_id], per_rhythm_true_strict[r_id])
        fm_tol, pc_tol = tolerant_macro_f1(per_rhythm_pred_strict[r_id], per_rhythm_true_strict[r_id],
                                           classes=(1,2,3), N=args.tol_samples)
        row = {
            'rhythm': r_name,
            'F1w_strict': round(fw, 4),
            'F1m_strict': round(fm, 4),
            'F1_P_strict': round(pc[1], 4) if len(pc)>1 else 0.0,
            'F1_QRS_strict': round(pc[2],4) if len(pc)>2 else 0.0,
            'F1_T_strict': round(pc[3], 4) if len(pc)>3 else 0.0,
            'F1m_tol±{}'.format(args.tol_samples): round(fm_tol, 4),
            'F1_P_tol': round(pc_tol[1], 4) if len(pc_tol)>1 else 0.0,
            'F1_QRS_tol': round(pc_tol[2],4) if len(pc_tol)>2 else 0.0,
            'F1_T_tol': round(pc_tol[3], 4) if len(pc_tol)>3 else 0.0,
        }
        per_rhythm_rows.append(row)

    # Rhythm accuracy (classifier, zero-cond)
    cls_preds_all = np.concatenate(cls_preds_all) if len(cls_preds_all) else np.array([], dtype=np.int64)
    cls_labels_all = np.concatenate(cls_labels_all) if len(cls_labels_all) else np.array([], dtype=np.int64)
    val_cls_acc = float((cls_preds_all == cls_labels_all).mean()) if cls_labels_all.size else 0.0

    # Write summary JSON
    summary = {
        'overall': {
            'F1w_strict': round(f1w_strict, 4),
            'F1m_strict': round(f1m_strict, 4),
            'F1m_tol±{}'.format(args.tol_samples): round(f1m_tol, 4),
            'per_class_F1_strict': {'bg': round(per_strict[0],4), 'P': round(per_strict[1],4), 'QRS': round(per_strict[2],4), 'T': round(per_strict[3],4)},
            'per_class_F1_tol': {'P': round(per_tol[1],4), 'QRS': round(per_tol[2],4), 'T': round(per_tol[3],4)},
            'F1w_strict_unmasked': round(f1w_unmasked,4) if f1w_unmasked is not None else None
        },
        'classifier_acc_zero_cond': round(val_cls_acc, 4),
        'settings': {
            'tol_samples': args.tol_samples,
            'data_dir': args.data_dir,
            'split_dir': args.split_dir,
            'model_path': args.model_path
        }
    }
    with open(os.path.join(args.out_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Write per-rhythm CSV
    csv_path = os.path.join(args.out_dir, 'per_rhythm_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rhythm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rhythm_rows)

    # Simple barplot for per-rhythm tolerant macro-F1
    try:
        import matplotlib.pyplot as plt
        rhythms = [r['rhythm'] for r in per_rhythm_rows]
        vals = [r['F1m_tol±{}'.format(args.tol_samples)] for r in per_rhythm_rows]
        plt.figure(figsize=(10,4))
        plt.bar(rhythms, vals)
        plt.ylim(0,1); plt.ylabel('Macro-F1 tol (P/QRS/T)'); plt.title('Per-rhythm macro-F1 (±{} samples)'.format(args.tol_samples))
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'per_rhythm_f1m_tol_bar.png')); plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    if not args.quiet:
        print("Done. Wrote:", args.out_dir)
        print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
