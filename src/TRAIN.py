import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MODEL import UNet3p
from LOSSES import ce_loss_masked, dice_loss_multiclass_masked
from POSTPROCESS import post_process, extract_boundaries, match_events
from METRICS import BoundaryStats
from UTILS import compute_class_weights
from CONFIG import FS, get_args

def train_epoch(model, loader, opt, scaler, device, args, class_weights):
    model.train()
    ep_loss = 0.0
    pbar = tqdm(loader, disable=args.quiet, desc="Training")
    for sig, lab, rid, msk, _ in pbar:
        sig, lab, rid, msk = sig.to(device), lab.to(device), rid.to(device), msk.to(device)
        p_absent_mask = torch.isin(rid, torch.tensor(P_ABSENT_IDS, device=device))
        opt.zero_grad()
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            seg_logits = model(sig, rid)
            # Crop target if mismatch
            L_log = seg_logits.size(2)
            if L_log < lab.size(1):
                diff = lab.size(1) - L_log
                start = diff // 2
                lab = lab[:, start:start + L_log]
                msk = msk[:, start:start + L_log]
            loss_ce = ce_loss_masked(seg_logits, lab, msk, class_weights, p_absent_mask)
            loss_dice = dice_loss_multiclass_masked(seg_logits, lab, msk, args.dice_exclude_bg, p_absent_mask)
            if args.seg_loss == 'dicece':
                seg_loss = loss_ce + loss_dice
            elif args.seg_loss == 'ce':
                seg_loss = loss_ce
            else:
                seg_loss = loss_dice
            loss = seg_loss

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ep_loss += loss.item()
    return ep_loss / len(loader)

def evaluate_val(model, loader, args, device, class_weights):
    model.eval()
    tol_samp = int(round(args.boundary_tol_ms * FS / 1000.0))
    stats = BoundaryStats()
    with torch.no_grad():
        for sig, lab, rid, msk, _ in loader:
            sig, lab, rid, msk = sig.to(device), lab.to(device), rid.to(device), msk.to(device).bool()
            seg_logits = model(sig, rid, suppress_p=True)  # Suppress during eval
            # Crop target if mismatch
            L_log = seg_logits.size(2)
            if L_log < lab.size(1):
                diff = lab.size(1) - L_log
                start = diff // 2
                lab = lab[:, start:start + L_log]
                msk = msk[:, start:start + L_log]
            pred = torch.argmax(seg_logits, dim=1).cpu().numpy()
            lab_np = lab.cpu().numpy()
            msk_np = msk.cpu().numpy()
            for b in range(sig.size(0)):
                valid_idx = np.where(msk_np[b])[0]
                if len(valid_idx) == 0: continue
                s0, s1 = valid_idx[0], valid_idx[-1] + 1
                gt_ = lab_np[b, s0:s1]
                pr_ = pred[b, s0:s1]
                pr_pp = post_process(pr_, args.min_len_ms)
                gtB = extract_boundaries(gt_)
                prB = extract_boundaries(pr_pp)
                r = rid[b].item()
                for (bn, _, _) in BOUNDARY_TYPES:
                    TP, FP, FN, errs = match_events(gtB[bn], prB[bn], tol_samp)
                    stats.add(r, bn, TP, FP, FN, errs)
    table = stats.finalize(args.boundary_tol_ms)
    macro_f1 = np.mean([table['macro'][bn][2] for (bn, _, _) in BOUNDARY_TYPES])
    return table, macro_f1

def train_model(args, splits):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3p(n_channels=4).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_sched == 'cosine':
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.t0, T_mult=args.t_mult, eta_min=args.eta_min)
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=args.plateau_factor, patience=args.plateau_patience)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    train_loader, val_loader, _ = get_dataloaders(args, splits)
    class_weights = compute_class_weights(train_loader.dataset, device) if not args.no_class_weights else None

    best_metric = -1.0
    log_csv = os.path.join(args.output_dir, "train_log.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(log_csv, "w") as f:
        f.write("epoch,lr,train_loss,macro_f1\n")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, scaler, device, args, class_weights)
        table, macro_f1 = evaluate_val(model, val_loader, args, device, class_weights)
        sched.step(macro_f1 if args.lr_sched == 'plateau' else epoch - 1)

        # Checkpoint
        if epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))

        # Best/last
        if macro_f1 > best_metric:
            best_metric = macro_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))

        with open(log_csv, "a") as f:
            f.write(f"{epoch},{opt.param_groups[0]['lr']},{train_loss},{macro_f1}\n")