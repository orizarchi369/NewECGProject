# scripts/viz_segments.py
import os, re, argparse, random, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']
COLORS = {0:(1,1,1), 1:(1.0,0.65,0.0), 2:(0.85,0.1,0.1), 3:(0.2,0.4,1.0)}  # bg,P,QRS,T

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
        for f in os.scandir(data_dir):
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
        npz = np.load(p, allow_pickle=True)
        sig = npz['signal'].astype(np.float32)     # [512]
        lab = npz['labels'].astype(np.int64)       # [512]
        rid = self.rhythm_ids[idx]
        # mask-from-meta (for completeness; plotting still shows full length)
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
        return sig, lab, rid, msk, os.path.basename(p)

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
    def __init__(self, K=len(RHYTHMS), use_bn=False, p_drop=0.0):
        super().__init__()
        self.K = K
        self.enc1 = ConvBlock1D(1+K, 32, use_bn, p_drop)
        self.pool = nn.MaxPool1d(2,2)
        self.enc2 = ConvBlock1D(32, 64, use_bn, p_drop)
        self.enc3 = ConvBlock1D(64,128, use_bn, p_drop)
        self.up1  = nn.ConvTranspose1d(128,64,2,2)
        self.dec1 = ConvBlock1D(64+64, 64, use_bn, p_drop)
        self.up2  = nn.ConvTranspose1d(64,32,2,2)
        self.dec2 = ConvBlock1D(32+32, 32, use_bn, p_drop)
        self.out  = nn.Conv1d(32,4,1)
        self.cls_pool = nn.AdaptiveAvgPool1d(1)
        self.cls_fc   = nn.Linear(128, K)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if getattr(m,'bias',None) is not None: nn.init.zeros_(m.bias)
    def forward(self, x, cond):
        B,_,L = x.shape
        cond_tiled = cond.unsqueeze(-1).expand(-1,-1,L)
        x = torch.cat([x, cond_tiled], dim=1)
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        d1 = self.up1(e3); e2a = F.interpolate(e2, size=d1.size(2), mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e2a], dim=1))
        d2 = self.up2(d1); e1a = F.interpolate(e1, size=d2.size(2), mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1a], dim=1))
        return self.out(d2)  # [B,4,L]

def make_cmap():
    from matplotlib.colors import ListedColormap
    arr = np.array([COLORS[i] for i in range(4)])
    return ListedColormap(arr)

def plot_example(sig, lab, pred, msk, out_png, title):
    L = sig.shape[0]
    fig = plt.figure(figsize=(10, 3.5))
    gs = fig.add_gridspec(3,1, height_ratios=[2.2,0.6,0.6], hspace=0.15)
    ax0 = fig.add_subplot(gs[0,0]); ax0.plot(np.arange(L), sig, linewidth=1.0)
    ax0.set_xlim(0, L-1); ax0.set_ylabel('norm'); ax0.set_title(title); ax0.grid(True, alpha=0.2)
    ax1 = fig.add_subplot(gs[1,0]); ax1.imshow(lab[None,:], aspect='auto', cmap=make_cmap(), vmin=0, vmax=3); ax1.set_yticks([]); ax1.set_ylabel('GT')
    ax2 = fig.add_subplot(gs[2,0]); ax2.imshow(pred[None,:], aspect='auto', cmap=make_cmap(), vmin=0, vmax=3); ax2.set_yticks([]); ax2.set_ylabel('Pred')
    for ax in (ax1, ax2): ax.set_xlim(0, L-1)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--per_rhythm', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use_batchnorm', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Load test split
    with open(os.path.join(args.split_dir, 'test_records.txt'), 'r') as f:
        test_records = [ln.strip() for ln in f if ln.strip()]

    ds = ECGSegments(test_records, args.data_dir)
    idx_by_rhythm = {r:[] for r in range(len(RHYTHMS))}
    for i, rid in enumerate(ds.rhythm_ids):
        idx_by_rhythm[rid].append(i)

    # Model (match training cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D_MTL(use_bn=args.use_batchnorm, p_drop=args.dropout).to(device).eval()
    sd = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(sd, strict=True)

    # Sample and plot
    for rid, indices in idx_by_rhythm.items():
        if len(indices)==0: continue
        take = min(args.per_rhythm, len(indices))
        picks = np.random.choice(indices, size=take, replace=False)
        for j, idx in enumerate(picks):
            sig, lab, rid_, msk, name = ds[idx]
            sigT = torch.from_numpy(sig).float().unsqueeze(0).unsqueeze(0).to(device)  # [1,1,L]
            cond = F.one_hot(torch.tensor([rid_], device=device), num_classes=len(RHYTHMS)).float()
            with torch.no_grad():
                logits = model(sigT, cond)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            out_png = os.path.join(args.out_dir, f"{RHYTHMS[rid]}_{j+1}_{name.replace('.npz','')}.png")
            title = f"{name} | rhythm={RHYTHMS[rid]}"
            plot_example(sig.numpy(), lab.numpy(), pred, msk.numpy(), out_png, title)

if __name__ == '__main__':
    main()
