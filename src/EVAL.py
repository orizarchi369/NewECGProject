import torch
from MODEL import UNet3p
from POSTPROCESS import post_process, extract_boundaries, match_events
from METRICS import BoundaryStats
from CONFIG import RHYTHM_TO_ID
from TRAIN import evaluate_val

def evaluate(model_path, loader, args, device, rhythm=None):
    model = UNet3p(n_channels=4).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    table, macro_f1 = evaluate_val(model, loader, args, device, None)
    return table, macro_f1