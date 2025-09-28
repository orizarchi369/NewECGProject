import json
import os
import re
import numpy as np
import torch
from CONFIG import RHYTHM_TO_ID, FS, SIGNAL_LENGTH, P_ABSENT_IDS, DRIVE_DATA_DIR, DRIVE_ANN_DIR, DRIVE_SPLIT_DIR, DRIVE_OUTPUT_DIR

def parse_record_and_rhythm_from_name(filename):
    base = os.path.basename(filename)
    rec = base.split('.')[0]  # e.g., AF0001
    m = re.match(r'^([A-Z]+)\d+', rec)
    rhythm = m.group(1) if m else None
    return rec, rhythm

def parse_annotations(ann_path, lead='ii'):
    """Parse TXT annotation for lead II to label array [5000]."""
    labels = np.zeros(SIGNAL_LENGTH, dtype=np.int64)
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3 or parts[2] != lead: continue  # Skip non-lead II
        sample = int(parts[0])
        symbol = parts[1]
        if symbol == '(':
            # Onset
            cur_class = None
        elif symbol == ')':
            # Offset, fill from onset to offset
            if cur_class is not None:
                labels[start:sample+1] = cur_class
            cur_class = None
        else:
            # Peak, set class
            if symbol == 'p':
                cur_class = 1  # P
            elif symbol == 'N':
                cur_class = 2  # QRS
            elif symbol == 't':
                cur_class = 3  # T
            start = sample  # Assume onset before peak, but TXT has ( onset, peak, ) offset
    return labels

def load_splits(split_dir):
    splits = {}
    for sp in ['train', 'val', 'test']:
        path = os.path.join(split_dir, f'{sp}_records.txt')
        with open(path, 'r') as f:
            splits[sp] = [ln.strip() for ln in f if ln.strip()]
    with open(os.path.join(split_dir, 'split_summary.json'), 'r') as f:
        summary = json.load(f)
    return splits, summary

def compute_class_weights(dataset, device):
    counts = np.zeros(4, dtype=np.float64)
    for i in range(len(dataset)):
        _, lab, _, msk, _ = dataset[i]
        lab = lab.numpy(); msk = msk.numpy().astype(bool)
        for c in range(4):
            counts[c] += np.sum((lab == c) & msk)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * 4.0
    return torch.tensor(w, dtype=torch.float32).to(device)

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def adjust_paths_for_colab(args):
    if is_colab():
        #from google.colab import drive
        #drive.mount('/content/drive')
        args.data_dir = DRIVE_DATA_DIR
        args.ann_dir = DRIVE_ANN_DIR
        args.split_dir = DRIVE_SPLIT_DIR
        args.output_dir = DRIVE_OUTPUT_DIR
    return args