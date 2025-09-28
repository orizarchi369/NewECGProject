import torch
import torch.nn.functional as F
from CONFIG import P_ABSENT_IDS

def ce_loss_masked(logits, targets, mask, class_weights=None, p_absent_mask=None):
    ce = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')  # [B,L]
    if p_absent_mask is not None:
        # Zero loss for P in absent rhythms (expand to L)
        p_absent = p_absent_mask.unsqueeze(-1).expand(-1, ce.size(1)).float()
        ce = ce * (1 - p_absent)  # Or specific P channel, but since per-position, approximate
    ce = ce * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return ce.sum() / denom

def dice_loss_multiclass_masked(logits, targets, mask, exclude_bg=True, p_absent_mask=None, eps=1e-6):
    probs = F.softmax(logits, dim=1)  # [B,C,L]
    B, C, L = probs.shape
    cls_range = range(1, C) if exclude_bg else range(C)
    tgt_oh = F.one_hot(targets.clamp_min(0), num_classes=C).permute(0,2,1).float()  # [B,C,L]
    mask_f = mask.float().unsqueeze(1)  # [B,1,L]
    probs = probs * mask_f
    tgt_oh = tgt_oh * mask_f
    dices = []
    for c in cls_range:
        p = probs[:, c, :]
        t = tgt_oh[:, c, :]
        if c == 1 and p_absent_mask is not None:  # P channel, zero for absent
            p_absent = p_absent_mask.unsqueeze(-1).expand(-1, L).float()
            p = p * (1 - p_absent)
            t = t * (1 - p_absent)
        num = 2.0 * (p * t).sum(dim=1)
        den = (p + t).sum(dim=1).clamp_min(eps)
        dices.append(1.0 - (num / den).mean())  # Avg per batch
    return sum(dices) / len(dices)