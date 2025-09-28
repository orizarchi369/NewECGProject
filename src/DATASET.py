import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from CONFIG import SIGNAL_LENGTH, RHYTHM_TO_ID
from UTILS import parse_record_and_rhythm_from_name, parse_annotations

class ECGSegments(Dataset):
    def __init__(self, record_ids, data_dir, ann_dir):
        self.files = []
        self.rhythm_ids = []
        for rec in record_ids:
            signal_path = os.path.join(data_dir, f"{rec}.csv")
            ann_path = os.path.join(ann_dir, f"{rec}_ii.txt")  # Assume naming
            if os.path.exists(signal_path) and os.path.exists(ann_path):
                _, rhythm = parse_record_and_rhythm_from_name(signal_path)
                if rhythm in RHYTHM_TO_ID:  # Skip invalid
                    self.files.append((signal_path, ann_path))
                    self.rhythm_ids.append(RHYTHM_TO_ID[rhythm])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal_path, ann_path = self.files[idx]
        # Load signal (CSV, one column, no header)
        sig = pd.read_csv(signal_path, header=None).values.flatten().astype(np.float32)
        assert len(sig) == SIGNAL_LENGTH, f"Signal length mismatch: {len(sig)}"
        # Load labels from ann
        lab = parse_annotations(ann_path)
        # Mask: All 1s since full signals, no padding
        msk = np.ones(SIGNAL_LENGTH, dtype=np.uint8)
        rid = self.rhythm_ids[idx]
        name = os.path.basename(signal_path)
        return (
            torch.from_numpy(sig).unsqueeze(0),  # [1, 5000]
            torch.from_numpy(lab),  # [5000]
            torch.tensor(rid, dtype=torch.long),
            torch.from_numpy(msk),  # [5000]
            name
        )

def get_dataloaders(args, splits):
    ds_train = ECGSegments(splits['train'], args.data_dir, args.ann_dir)
    ds_val = ECGSegments(splits['val'], args.data_dir, args.ann_dir)
    ds_test = ECGSegments(splits['test'], args.data_dir, args.ann_dir)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader