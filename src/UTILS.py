import json
import os
import re
import csv
import numpy as np
import torch
from CONFIG import RHYTHM_TO_ID, FS, SIGNAL_LENGTH, P_ABSENT_IDS, DRIVE_DATA_DIR, DRIVE_ANN_DIR, DRIVE_SPLIT_DIR, DRIVE_OUTPUT_DIR, CLASS_P, CLASS_QRS, CLASS_T, SIGNAL_LENGTH

def parse_record_and_rhythm_from_name(filename):
    base = os.path.basename(filename)
    rec = base.split('.')[0]  # e.g., AF0001
    m = re.match(r'^([A-Z]+)\d+', rec)
    rhythm = m.group(1) if m else None
    return rec, rhythm

TYPE2CLASS = {0: CLASS_P, 1: CLASS_QRS, 2: CLASS_T}

def parse_annotations(ann_path):
    """Parse 'TYPE,START,END' TXT into a [SIGNAL_LENGTH] label array (END inclusive)."""
    labels = np.zeros(SIGNAL_LENGTH, dtype=np.int64)
    with open(ann_path, 'r', newline='') as f:
        rdr = csv.reader(f)
        first = next(rdr, None)
        # Detect header
        has_header = first and any(x.strip().isalpha() for x in first)
        rows = rdr if has_header else ([first] if first else [])
        for row in rows:
            if not row:
                continue
            t, s, e = int(row[0]), int(row[1]), int(row[2])
            c = TYPE2CLASS.get(t)
            if c is None:
                continue
            s = max(0, s)
            e = min(SIGNAL_LENGTH - 1, e)  # END is inclusive in your files
            if e >= s:
                labels[s:e+1] = c
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