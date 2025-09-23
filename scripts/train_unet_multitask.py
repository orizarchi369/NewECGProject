# scripts/train_unet_multitask.py
import os, csv, argparse, warnings, re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------- Configurable rhythm order (edit if needed) --------
RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']  # K = len(RHYTHMS)

# ---------------- Args ----------------
ap = argparse.ArgumentParser(description='Rhythm-aware multitask U-Net (1D) for ECG delineation')
ap.add_argument('--data_dir', type=str, default='/home/orizarchi/projects/NewECGProject/data/processed/lead_ii_segments')
ap.add_argument('--split_dir', type=str, default='/home/orizarchi/projects/NewECGProject/data/splits')
ap.add_argument('--output_dir', type=str, default='/content/drive/My Drive/ecg_project/models')

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
        sig = npz['signal'].astype(np.float32)     # [512]
        lab = npz['labels'].astype(np.int64)       # [512]
        # rhythm: from meta if present, else from filename
        rhythm_idx = None
        if 'meta' in npz and 'rhythm_type' in npz['meta'].item():
            rstr = str(npz['meta'].item()['rhythm_type'])
            rhythm_idx = self.rhythm_to_id.get(rstr, None)
        if rhythm_idx is None:
            _, rstr = parse_record_and_rhythm_from_name(p)
            rhythm_idx = self.rhythm_to_id.get(rstr, 0)
        return torch.from_numpy(sig).unsqueeze(0), torch.from_numpy(lab), torch.tensor(rhythm_idx, dtype=torch.long)

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
       Option A conditioning: concatenate K constant channels (one-hot/probs) to input.
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

        # Classifier head from bottleneck (enc3)
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
            cond = x_ecg.new_zeros((B, self.K))  # [B,K]
        else:
            cond = cond_vec
        # Tile across time and concat to input
        cond_tiled = cond.unsqueeze(-1).expand(-1, -1, L)   # [B,K,L]
        x = torch.cat([x_ecg, cond_tiled], dim=1)           # [B,1+K,L]

        # Encoder
        e1 = self.enc1(x)                   # [B,32,L]
        e2 = self.enc2(self.pool(e1))       # [B,64,L/2]
        e3 = self.enc3(self.pool(e2))       # [B,128,L/4]

        # Classifier (from bottleneck)
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

# -------------- Utils --------------
def current_lr(optimizer): return optimizer.param_groups[0]['lr']

def ensure_csv(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','train_loss','val_loss','val_f1','val_cls_acc','lr'])

def append_csv(path, row):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def plot_curves(train_losses, val_losses, val_f1s, out_path):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.plot(val_f1s, label='Val F1 (seg)')
    plt.xlabel('Epoch'); plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

def compute_class_weights(dataset, num_classes=4, clip=(0.5, 5.0)):
    counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, lab, _ = dataset[i]
        lab_np = lab.numpy()
        for c in range(num_classes):
            counts[c] += (lab_np == c).sum()
    freq = counts / max(1.0, counts.sum())
    w = 1.0 / (freq + 1e-6)
    w = w / w.mean()
    w = np.clip(w, clip[0], clip[1])
    return torch.tensor(w, dtype=torch.float)

# -------------- Train/Eval --------------
def train(model, loaders, out_dir, class_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    seg_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    cls_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.plateau_factor, patience=args.plateau_patience
    )

    history_csv = os.path.join(out_dir, 'training_history.csv')
    ensure_csv(history_csv)

    best_f1, es_counter = -1.0, 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(1, args.num_epochs+1):
        # ---- Train
        model.train()
        running = 0.0
        pbar = tqdm(loaders['train'], desc=f'Epoch {epoch}/{args.num_epochs}', disable=args.quiet)
        for sig, lab, rid in pbar:
            sig, lab, rid = sig.to(device), lab.to(device), rid.to(device)
            # hard one-hot conditioning from ground-truth rhythm
            cond = F.one_hot(rid, num_classes=len(RHYTHMS)).float()  # [B,K]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                seg_logits, cls_logits = model(sig, cond)   # [B,4,L], [B,K]
                loss_seg = seg_criterion(seg_logits, lab)
                loss_cls = cls_criterion(cls_logits, rid)
                loss = loss_seg + args.lambda_cls * loss_cls
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += loss.item()
            if not args.quiet:
                pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr(optimizer):.2e}')
        train_loss = running / max(1, len(loaders['train']))
        train_losses.append(train_loss)

        # ---- Validation
        model.eval()
        vloss = 0.0
        seg_preds_all, seg_labels_all = [], []
        cls_preds_all, cls_labels_all = [], []

        with torch.no_grad():
            for sig, lab, rid in loaders['val']:
                sig, lab, rid = sig.to(device), lab.to(device), rid.to(device)

                # segmentation eval with hard one-hot conditioning (GT rhythm)
                cond_hard = F.one_hot(rid, num_classes=len(RHYTHMS)).float()
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    seg_logits, _ = model(sig, cond_hard)
                    loss_seg = seg_criterion(seg_logits, lab)

                # classifier eval with zero conditioning (to avoid trivial leak)
                zero_cond = torch.zeros((sig.size(0), len(RHYTHMS)), device=device)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    _, cls_logits = model(sig, zero_cond)
                    loss_cls = cls_criterion(cls_logits, rid)

                loss = loss_seg + args.lambda_cls * loss_cls
                vloss += loss.item()

                seg_preds = torch.argmax(seg_logits, dim=1)  # [B,L]
                seg_preds_all.append(seg_preds.detach().cpu().numpy().ravel())
                seg_labels_all.append(lab.detach().cpu().numpy().ravel())

                cls_pred = torch.argmax(cls_logits, dim=1)   # [B]
                cls_preds_all.append(cls_pred.detach().cpu().numpy())
                cls_labels_all.append(rid.detach().cpu().numpy())

        val_loss = vloss / max(1, len(loaders['val']))
        scheduler.step(val_loss)
        seg_preds_all = np.concatenate(seg_preds_all)
        seg_labels_all = np.concatenate(seg_labels_all)
        val_f1 = f1_score(seg_labels_all, seg_preds_all, average='weighted')
        val_losses.append(val_loss); val_f1s.append(val_f1)

        cls_preds_all = np.concatenate(cls_preds_all)
        cls_labels_all = np.concatenate(cls_labels_all)
        val_cls_acc = accuracy_score(cls_labels_all, cls_preds_all)

        if not args.quiet:
            print(f"Epoch {epoch}/{args.num_epochs} | LR {current_lr(optimizer):.2e} | "
                  f"Train {train_loss:.4f} | Val {val_loss:.4f} | F1(seg) {val_f1:.4f} | Acc(rhythm) {val_cls_acc:.4f}")

        append_csv(history_csv, [epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_f1:.6f}', f'{val_cls_acc:.6f}', f'{current_lr(optimizer):.8f}'])

        # checkpoints
        torch.save(model.state_dict(), os.path.join(out_dir, 'last_unet_mtl.pth'))
        if val_f1 > best_f1 + args.min_delta:
            best_f1 = val_f1; es_counter = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_unet_mtl.pth'))
        else:
            es_counter += 1

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth'))

        # live plot
        if args.live_plot and (epoch % args.plot_interval == 0 or epoch == args.num_epochs):
            plot_curves(train_losses, val_losses, val_f1s, os.path.join(out_dir, 'training_curves_live.png'))

        # early stop
        if es_counter >= args.early_stop_patience:
            if not args.quiet:
                print(f"Early stopping at epoch {epoch} (no F1 improvement for {args.early_stop_patience} epochs).")
            break

    # final plot
    plot_curves(train_losses, val_losses, val_f1s, os.path.join(out_dir, 'training_curves.png'))

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
    
    print("Class weights:", class_weights.tolist() if class_weights is not None else None) # print class weights

    # Train
    train(model, loaders, args.output_dir, class_weights=class_weights)
