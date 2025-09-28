import torch
from MODEL import UNet1D
from POSTPROCESS import post_process, extract_boundaries, match_events
from METRICS import BoundaryStats
from CONFIG import RHYTHM_TO_ID
from TRAIN import evaluate_val

def evaluate(model_path, loader, args, device, rhythm=None):
    model = UNet1D(use_bn=args.use_batchnorm, p_drop=args.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    suppress_p = rhythm is not None
    rid = torch.tensor(RHYTHM_TO_ID.get(rhythm, 0), device=device) if rhythm else None  # Single for batch or per-sample
    table, macro_f1 = evaluate_val(model, loader, args, device, None)  # Reuse val func, but with optional rid/suppress
    return table, macro_f1