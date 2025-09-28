from CONFIG import get_args
from UTILS import load_splits, adjust_paths_for_colab, is_colab
from TRAIN import train_model
from EVAL import evaluate
from DATASET import get_dataloaders
import torch

if __name__ == "__main__":
    args = get_args()
    args = adjust_paths_for_colab(args)
    splits, _ = load_splits(args.split_dir)
    if args.mode == "train":
        train_model(args, splits)
    elif args.mode == "eval":
        _, val_loader, test_loader = get_dataloaders(args, splits)
        loader = test_loader if args.rhythm else val_loader  # Or switch
        table, macro_f1 = evaluate(args.model_path, loader, args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), args.rhythm)
        print(f"Macro F1: {macro_f1}")