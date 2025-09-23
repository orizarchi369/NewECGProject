# scripts/train_unet_multitask_mask_tol.py

# Updated training script, with: padding-mask (ignores padded samples in loss/metrics),
# tolerant eval (±N samples) logging F1m_tol alongside strict F1s, rhythm-aware multitask U-Net
# (segmentation + bottleneck rhythm head with one-hot conditioning), ReduceLROnPlateau LR schedule,
# early stopping on tolerant macro-F1, periodic checkpoints + best/last saves, and optional live plots.

import os, csv, argparse, warnings, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- Rhythm order (edit if needed) --------
RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']  # K = len(RHYTHMS)

# ---------------- Args ----------------
ap = argparse.ArgumentParser(description='Rhythm-aware multitask U-Net (1D) with padding mask + tolerant eval')
ap.add_argument('--data_dir', type=str, required=True)
ap.add_argument('--split_dir', type=str, required=True)
ap.add_argument('--output_dir', type=str, required=True)

ap.add_argument('--num_epochs', type=int, default=200)
ap.add_argument('--batch_size', type=int, default=32)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--num_workers', type=int, default=2)
ap.add_argument('--seed', type=int, default=42)

# Loss / training
ap.add_argument('--lambda_cls', type=float, default=0.3, help='weight for rhythm classification loss')
ap.add_argument('--plateau_patience', type=int, default=5)
ap.add_argument('--plateau_factor', type=float, default=0.5)
ap.add_argument('--early_stop_patience', type=int, default=20)
ap.add_argument('--min_delta', type=float, default=1e-4)
ap.add_argument('--no_class_weights', action='store_true', help='disable seg class weights')
ap.add_argument('--dropout', type=float, default=0.0)
ap.add_argument('--use_batchnorm', action='store_true')

# Eval tolerance
ap.add_argument('--tol_samples', type=int, default=6, help='±N-sample tolerance for tolerant macro-F1 (P,QRS,T)')

# UX
ap.add_argument('--live_plot', action='store_true')
ap.add_argument('--plot_interval', type=int, default=5)
ap.add_argument('--checkpoint_interval', type=int, default=0)
ap.add_argument('--quiet', action='store_true')
ap.add_argument('--suppress_warnings', action='store_true')

args = ap.parse_args()
if args.suppress_warnings:
    warnings.filterwarnings('ignore', category=FutureWarning)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# -------------- Dataset --------------
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
        self.files = []
        for f in os.listdir(data_dir):
            if not (f.endswith('.npz') and lead_filter in f):
                continue
            rec = f.split('_ii_')[0]
            if rec in record_ids:
                self.files.append(os.path.join(data_dir, f))
        self.files.sort()

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        npz = np.load(p, allow_pickle=True)
        sig = npz['signal'].astype(np.float32)     # [L=512]
        lab = npz['labels'].astype(np.int64)       # [L=512]

        # rhythm: from meta if present, else from filename prefix
        rhythm_idx = None
        if 'meta' in npz and 'rhythm_type' in npz['meta'].item():
            rstr = str(npz['meta'].item()['rhythm_type'])
            rhythm_idx = self.rhythm_to_id.get(rstr, None)
        if rhythm_idx is None:
            _, rstr = parse_record_and_rhythm_from_name(p)
            rhythm_idx = self.rhythm_to_id.get(rstr, 0)

        # build valid mask from meta['orig_start']/['orig_end']; fallback = all ones
        meta = npz['meta'].item() if 'meta' in npz else {}
        L = lab.shape[0]  # 512
        if 'orig_start' in meta and 'orig_end' in meta:
            orig_len = int(meta['orig_end']) - int(meta['orig_start']) + 1  # inclusive
            if orig_len >= L:
                msk = np.ones(L, dtype=np.uint8)  # resampled long beat → all valid
            else:
                pad_left = (L - orig_len) // 2    # same centering used when saving
                msk = np.zeros(L, dtype=np.uint8)
                msk[pad_left:pad_left + orig_len] = 1
        else:
            msk = np.ones(L, dtype=np.uint8)

        return (torch.from_numpy(sig).unsqueeze(0),   # [1,L]
                torch.from_numpy(lab),                # [L]
                torch.tensor(rhythm_idx, dtype=torch.long),
                torch.from_numpy(msk))                # [L], 1=valid, 0=padding

# -------------- Model --------------
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
    """U-Net encoder-decoder for segmentation + bottleneck classifier for rhythm.
       Conditioning (Option A): concatenate K constant channels (one-hot/probs) to input.
    """
    def __init__(self, in_ecg_channels=1, num_seg_classes=4, num_rhythms=len(RHYTHMS),
                 use_bn=False, p_drop=0.0):
        super().__init__()
        self.K = num_rhythms
        in_total = in_ecg_channels + self.K  # ECG + K conditioning channels

        # Encoder
        self.enc1 = ConvBlock1D(in_total, 32, use_bn, p_drop)
        self.pool = nn.MaxPool1d(2,2)
        self.enc2 = ConvBlock1D(32, 64, use_bn, p_drop)
        self.enc3 = ConvBlock1D(64, 128, use_bn, p_drop)

        # Decoder
        self.up1  = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(64+64, 64, use_bn, p_drop)
        self.up2  = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(32+32, 32, use_bn, p_drop)
        self.out  = nn.Conv1d(32, num_seg_classes, kernel_size=1)

        # Classifier from bottleneck
        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_fc   = nn.Linear(128, self.K)

        # Init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_ecg, cond_vec=None):
        """
        x_ecg: [B, 1, L]
        cond_vec: [B, K] probabilities or one-hot. If None, zeros used.
        """
        B, _, L = x_ecg.shape
        if cond_vec is None:
            cond = x_ecg.new_zeros((B, self.K))
        else:
            cond = cond_vec
        cond_tiled = cond.unsqueeze(-1).expand(-1, -1, L)   # [B,K,L]
        x = torch.cat([x_ecg, cond_tiled], dim=1)           # [B,1+K,L]

        # Encoder
        e1 = self.enc1(x)                   # [B,32,L]
        e2 = self.enc2(self.pool(e1))       # [B,64,L/2]
        e3 = self.enc3(self.pool(e2))       # [B,128,L/4]

        # Classifier
        z = self.cls_pool(e3).squeeze(-1)   # [B,128]
        cls_logits = self.cls_fc(z)         # [B,K]

        # Decoder
        d1 = self.up1(e3)                   # [B,64,L/2]
        e2a = F.interpolate(e2, size=d1.size(2), mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2a], dim=1))  # [B,64,L/2]

        d2 = self.up2(d1)                   # [B,32,L]
        e1a = F.interpolate(e1, size=d2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1a], dim=1))  # [B,32,L]

        seg_logits = self.out(d2)           # [B,4,L]
        return seg_logits, cls_logits

# -------------- Metrics (strict + tolerant) --------------
def _flatten_masked(arr_list):
    """Concatenate list of 1D numpy arrays."""
    if len(arr_list) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(arr_list)

def strict_scores(seg_preds_all, seg_labels_all, num_classes=4, macro_classes=(1,2,3)):
    y_pred = _flatten_masked(seg_preds_all)
    y_true = _flatten_masked(seg_labels_all)
    if y_true.size == 0:
        return 0.0, 0.0
    f1w = f1_score(y_true, y_pred, average='weighted', labels=list(range(num_classes)))
    # macro over selected classes
    f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    valid = [c for c in macro_classes if (y_true == c).any()]
    if len(valid) == 0:
        f1m = 0.0
    else:
        f1m = float(np.mean([f1_per[c] for c in valid]))
    return f1w, f1m

def dilate_bool_mask(mask_t, N):
    """
    mask_t: torch.bool [B,L]
    Return dilated torch.bool [B,L] using 1D max-pool with kernel=2N+1.
    """
    if N <= 0:
        return mask_t
    x = mask_t.unsqueeze(1).float()  # [B,1,L]
    y = F.max_pool1d(x, kernel_size=2*N+1, stride=1, padding=N)
    return (y > 0).squeeze(1)

def tolerant_macro_f1(seg_preds_all, seg_labels_all, classes=(1,2,3), N=6):
    """
    Symmetric tolerance:
      precision: TP = pred ∧ dilate(GT, N)
      recall:    TP = GT   ∧ dilate(pred, N)
    F1 per class, then macro over available classes.
    """
    if len(seg_preds_all) == 0:
        return 0.0
    # Convert lists of arrays to a single boolean tensor per class
    preds = torch.tensor(_flatten_masked(seg_preds_all), dtype=torch.long)
    gts   = torch.tensor(_flatten_masked(seg_labels_all), dtype=torch.long)
    if preds.numel() == 0:
        return 0.0
    B = 1  # we flattened; treat as one long sequence
    L = preds.numel()
    preds = preds.view(B, L)
    gts   = gts.view(B, L)

    f1s = []
    eps = 1e-8
    for c in classes:
        p = (preds == c)  # [B,L] bool
        g = (gts   == c)

        p_sum = p.sum().item()
        g_sum = g.sum().item()
        if p_sum == 0 and g_sum == 0:
            # skip class with no presence in either
            continue

        dil_g = dilate_bool_mask(g, N)
        dil_p = dilate_bool_mask(p, N)

        # tolerant precision / recall
        tp_prec = (p & dil_g).sum().item()
        tp_rec  = (g & dil_p).sum().item()
        prec = tp_prec / max(1, p_sum)
        rec  = tp_rec  / max(1, g_sum)

        f1 = 0.0 if (prec+rec) < eps else (2*prec*rec)/(prec+rec+eps)
        f1s.append(f1)

    if len(f1s) == 0:
        return 0.0
    return float(np.mean(f1s))

# -------------- Utils --------------
def current_lr(opt): return opt.param_groups[0]['lr']

def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch','train_loss','val_loss',
                'val_f1w_strict','val_f1m_strict','val_f1m_tol',
                'val_cls_acc','lr'
            ])

def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def plot_curves(train_losses, val_losses, val_f1m_tol, out_path):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_f1m_tol, label='Val Macro-F1 tol (P/QRS/T)')
    plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def compute_class_weights(dataset, num_classes=4, clip=(0.5, 5.0)):
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, lab, _, _ = dataset[i]
        lab_np = lab.numpy()
        for c in range(num_classes):
            counts[c] += (lab_np == c).sum()
    freq = counts / max(1.0, counts.sum())
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean()
    w = np.clip(w, clip[0], clip[1])
    return torch.tensor(w, dtype=torch.float)

# -------------- Train/Eval --------------
class Trainer:
    def __init__(self, model, loaders, out_dir, class_weights=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.use_cuda = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_cuda)

        # per-element CE (masked mean later)
        self.seg_criterion = nn.CrossEntropyLoss(
            reduction='none',
            weight=class_weights.to(self.device) if class_weights is not None else None
        )
        self.cls_criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience
        )

        self.loaders = loaders
        self.out_dir = out_dir
        self.history_csv = os.path.join(out_dir, 'training_history.csv')
        ensure_csv(self.history_csv)

        self.best_f1 = -1.0
        self.es_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1m_tol = []

    def _masked_ce_mean(self, logits, target, mask_bool):
        # logits: [B,C,L]; target: [B,L]; mask_bool: [B,L]
        loss_all = self.seg_criterion(logits, target)  # [B,L]
        num = (loss_all * mask_bool.float()).sum()
        den = mask_bool.float().sum().clamp_min(1.0)
        return num / den

    def train(self):
        for epoch in range(1, args.num_epochs+1):
            # ---- Train ----
            self.model.train()
            running = 0.0
            pbar = tqdm(self.loaders['train'], desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.quiet)
            for sig, lab, rid, msk in pbar:
                sig, lab, rid, msk = sig.to(self.device), lab.to(self.device), rid.to(self.device), msk.to(self.device).bool()
                cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float()  # [B,K]

                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.use_cuda):
                    seg_logits, cls_logits = self.model(sig, cond)
                    loss_seg = self._masked_ce_mean(seg_logits, lab, msk)
                    loss_cls = self.cls_criterion(cls_logits, rid)
                    loss = loss_seg + args.lambda_cls * loss_cls

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer); self.scaler.update()
                running += loss.item()
                if not args.quiet:
                    pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr(self.optimizer):.2e}')
            train_loss = running / max(1, len(self.loaders['train']))
            self.train_losses.append(train_loss)

            # ---- Validation ----
            self.model.eval()
            vloss = 0.0
            seg_preds_all_strict, seg_labels_all_strict = [], []
            cls_preds_all, cls_labels_all = [], []

            with torch.no_grad():
                for sig, lab, rid, msk in self.loaders['val']:
                    sig, lab, rid, msk = sig.to(self.device), lab.to(self.device), rid.to(self.device), msk.to(self.device).bool()

                    # Segmentation eval with GT rhythm (hard one-hot)
                    cond_hard = F.one_hot(rid, num_classes=len(RHYTHMS)).float()
                    with torch.amp.autocast('cuda', enabled=self.use_cuda):
                        seg_logits, _ = self.model(sig, cond_hard)
                        loss_seg = self._masked_ce_mean(seg_logits, lab, msk)

                    # Classifier eval with zero-conditioning (avoid leak)
                    zero_cond = torch.zeros((sig.size(0), len(RHYTHMS)), device=self.device)
                    with torch.amp.autocast('cuda', enabled=self.use_cuda):
                        _, cls_logits = self.model(sig, zero_cond)
                        loss_cls = self.cls_criterion(cls_logits, rid)

                    loss = loss_seg + args.lambda_cls * loss_cls
                    vloss += loss.item()

                    preds = torch.argmax(seg_logits, dim=1)  # [B,L]
                    # collect STRICT arrays masked by valid positions only
                    for p, y, m in zip(preds, lab, msk):
                        m = m.bool()
                        seg_preds_all_strict.append(p[m].cpu().numpy())
                        seg_labels_all_strict.append(y[m].cpu().numpy())

                    cls_pred = torch.argmax(cls_logits, dim=1)
                    cls_preds_all.append(cls_pred.detach().cpu().numpy())
                    cls_labels_all.append(rid.detach().cpu().numpy())

            val_loss = vloss / max(1, len(self.loaders['val']))
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            # STRICT metrics
            f1w_strict, f1m_strict = strict_scores(seg_preds_all_strict, seg_labels_all_strict, num_classes=4, macro_classes=(1,2,3))
            # TOLERANT macro-F1 (P/QRS/T) with ±N
            f1m_tol = tolerant_macro_f1(seg_preds_all_strict, seg_labels_all_strict, classes=(1,2,3), N=args.tol_samples)
            self.val_f1m_tol.append(f1m_tol)

            # Rhythm acc
            cls_preds_all = np.concatenate(cls_preds_all) if len(cls_preds_all) else np.array([], dtype=np.int64)
            cls_labels_all = np.concatenate(cls_labels_all) if len(cls_labels_all) else np.array([], dtype=np.int64)
            val_cls_acc = accuracy_score(cls_labels_all, cls_preds_all) if cls_labels_all.size else 0.0

            if not args.quiet:
                print(f"Epoch {epoch}/{args.num_epochs} | LR {current_lr(self.optimizer):.2e} | "
                      f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                      f"F1w(strict) {f1w_strict:.4f} | F1m(strict) {f1m_strict:.4f} | F1m(tol±{args.tol_samples}) {f1m_tol:.4f} | "
                      f"Acc(rhythm) {val_cls_acc:.4f}")

            append_csv(self.history_csv, [
                epoch, f'{train_loss:.6f}', f'{val_loss:.6f}',
                f'{f1w_strict:.6f}', f'{f1m_strict:.6f}', f'{f1m_tol:.6f}',
                f'{val_cls_acc:.6f}', f'{current_lr(self.optimizer):.8f}'
            ])

            # checkpoints
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'last_unet_mtl_mask_tol.pth'))
            # choose best by tolerant macro-F1 (P/QRS/T)
            if f1m_tol > self.best_f1 + args.min_delta:
                self.best_f1 = f1m_tol
                self.es_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, 'best_unet_mtl_mask_tol.pth'))
            else:
                self.es_counter += 1

            if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f'checkpoint_epoch_{epoch}.pth'))

            # live plot
            if args.live_plot and (epoch % args.plot_interval == 0 or epoch == args.num_epochs):
                plot_curves(self.train_losses, self.val_losses, self.val_f1m_tol,
                            os.path.join(self.out_dir, 'training_curves_live.png'))

            # early stop
            if self.es_counter >= args.early_stop_patience:
                if not args.quiet:
                    print(f"Early stopping at epoch {epoch} (no tol-F1 improvement for {args.early_stop_patience} epochs).")
                break

        # final plot
        plot_curves(self.train_losses, self.val_losses, self.val_f1m_tol,
                    os.path.join(self.out_dir, 'training_curves.png'))

# -------------- Main --------------
if __name__ == '__main__':
    # Load record lists
    splits = {}
    for sp in ['train','val','test']:
        with open(os.path.join(args.split_dir, f'{sp}_records.txt'), 'r') as f:
            splits[sp] = [ln.strip() for ln in f if ln.strip()]

    # Datasets/Loaders
    ds_train = ECGSegments(splits['train'], args.data_dir)
    ds_val   = ECGSegments(splits['val'],   args.data_dir)
    ds_test  = ECGSegments(splits['test'],  args.data_dir)

    # Class weights (from training set) unless disabled
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(ds_train, num_classes=4)

    loaders = {
        'train': DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
        'val':   DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
        'test':  DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0)),
    }

    # Model
    model = UNet1D_MTL(in_ecg_channels=1, num_seg_classes=4, num_rhythms=len(RHYTHMS),
                       use_bn=args.use_batchnorm, p_drop=args.dropout)

    # Train
    trainer = Trainer(model, loaders, args.output_dir, class_weights=class_weights)
    trainer.train()
