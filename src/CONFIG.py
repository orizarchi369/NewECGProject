import argparse
import os

# Global constants and paths
DATA_DIR = "/home/orizarchi/projects/NewECGProject/data/processed/lead_ii_filtered_normalized"  # Filtered/normalized CSVs
ANN_DIR = "C:/Users/orizarchi/Desktop/Courses/Resting_ECG_Dataset/ann_txt"  # TXT annotations (adapt for WSL/Colab)
SPLIT_DIR = "/home/orizarchi/projects/NewECGProject/splits"  # Splits TXT/JSON
OUTPUT_DIR = "/home/orizarchi/projects/NewECGProject/outputs"  # For models, logs

# For Colab/Drive
DRIVE_MOUNT = "/content/drive"
DRIVE_PROJECT_DIR = os.path.join(DRIVE_MOUNT, "My Drive/NewECGProject")
DRIVE_DATA_DIR = os.path.join(DRIVE_PROJECT_DIR, "data/processed/lead_ii_filtered_normalized")
DRIVE_ANN_DIR = os.path.join(DRIVE_PROJECT_DIR, "ann_txt")
DRIVE_SPLIT_DIR = os.path.join(DRIVE_PROJECT_DIR, "splits")
DRIVE_OUTPUT_DIR = os.path.join(DRIVE_PROJECT_DIR, "outputs")

RHYTHMS = ["AF", "AFIB", "SB", "SI", "SR", "ST", "VT"]
NUM_RHYTHMS = len(RHYTHMS)
RHYTHM_TO_ID = {r: i for i, r in enumerate(RHYTHMS)}

CLASS_BG, CLASS_P, CLASS_QRS, CLASS_T = 0, 1, 2, 3
NUM_CLASSES = 4

BOUNDARY_TYPES = [
    ("P_on", CLASS_P, "on"), ("P_off", CLASS_P, "off"),
    ("QRS_on", CLASS_QRS, "on"), ("QRS_off", CLASS_QRS, "off"),
    ("T_on", CLASS_T, "on"), ("T_off", CLASS_T, "off")
]

FS = 500  # Sampling frequency
SIGNAL_LENGTH = 5000  # 10 seconds at 500 Hz
EMBED_DIM = 16  # For rhythm embeddings

# P-absent rhythms for suppression and loss weighting
P_ABSENT_RHYTHMS = ["AF", "AFIB", "VT"]
P_ABSENT_IDS = [RHYTHM_TO_ID[r] for r in P_ABSENT_RHYTHMS]

# Hyperparameters (overridable via args)
DEFAULTS = {
    "num_epochs": 200,
    "batch_size": 32,
    "lr": 5e-4,
    "use_batchnorm": True,
    "dropout": 0.1,
    "seg_loss": "dicece",
    "dice_exclude_bg": True,
    "no_class_weights": False,
    "lr_sched": "cosine",
    "t0": 10,
    "t_mult": 2,
    "eta_min": 1e-6,
    "plateau_patience": 8,
    "plateau_factor": 0.5,
    "boundary_tol_ms": 150,
    "min_len_ms": 40,
    "seed": 42,
    "checkpoint_interval": 25,
    "quiet": False,
    "suppress_warnings": False,
    "mode": "train",  # "train" or "eval"
    "rhythm": None,  # For eval suppression, e.g., "AF"
    "model_path": None  # For eval, path to checkpoint
}

def get_args():
    parser = argparse.ArgumentParser(description="ECG Delineation with Rhythm Conditioning")
    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true" if val else "store_false", default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val) if val is not None else str, default=val)
    # Path overrides
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--ann_dir", default=ANN_DIR)
    parser.add_argument("--split_dir", default=SPLIT_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    return parser.parse_args()