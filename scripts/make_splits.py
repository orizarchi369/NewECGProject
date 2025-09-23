# scripts/make_splits.py
# group-stratified (by record), 70/15/15, using rhythm from filename

import os, re, argparse, json
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

RHYTHMS = ['AF','AFIB','AT','SB','SI','SR','ST','VT']  # ordered; edit if needed

def parse_record_and_rhythm(fname: str):
    # Expect: AF0001_ii_008.npz  -> record_id=AF0001, rhythm='AF'
    base = os.path.basename(fname)
    if not base.endswith('.npz') or '_ii_' not in base:
        return None, None
    rec = base.split('_ii_')[0]                # 'AF0001'
    m = re.match(r'^([A-Z]+)\d+', rec)         # 'AF'
    if not m:
        return None, None
    rhythm = m.group(1)
    return rec, rhythm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments_dir', type=str, required=True,
                    help='Folder with *.npz beat segments (e.g., .../lead_ii_segments)')
    ap.add_argument('--split_dir', type=str, required=True,
                    help='Output folder for train/val/test_records.txt')
    ap.add_argument('--train_ratio', type=float, default=0.70)
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.split_dir, exist_ok=True)

    # Collect records and their rhythms (groups)
    files = [os.path.join(args.segments_dir, f) for f in os.listdir(args.segments_dir) if f.endswith('.npz')]
    groups = defaultdict(list)  # record_id -> list of files
    record_rhythm = {}          # record_id -> rhythm string

    for f in files:
        rec, rhythm = parse_record_and_rhythm(f)
        if rec is None: 
            continue
        groups[rec].append(f)
        if rec in record_rhythm:
            if record_rhythm[rec] != rhythm:
                raise ValueError(f"Inconsistent rhythm for {rec}: {record_rhythm[rec]} vs {rhythm}")
        else:
            record_rhythm[rec] = rhythm

    records = sorted(groups.keys())
    rhythms = [record_rhythm[r] for r in records]

    # Map rhythms to IDs (drop any unknowns)
    known = set(RHYTHMS)
    filtered_records, filtered_rhythms = [], []
    for r, rh in zip(records, rhythms):
        if rh in known:
            filtered_records.append(r)
            filtered_rhythms.append(rh)
    records, rhythms = filtered_records, filtered_rhythms

    # First split: train vs (val+test)
    rec_train, rec_rest, y_train, y_rest = train_test_split(
        records, rhythms, train_size=args.train_ratio, random_state=args.seed, stratify=rhythms
    )

    # Second split: val vs test (proportional from the rest)
    val_portion = args.val_ratio / (args.val_ratio + args.test_ratio)
    rec_val, rec_test, _, _ = train_test_split(
        rec_rest, y_rest, train_size=val_portion, random_state=args.seed, stratify=y_rest
    )

    # Save record IDs
    def write_list(path, lst):
        with open(path, 'w') as f:
            for x in sorted(lst):
                f.write(x + '\n')

    write_list(os.path.join(args.split_dir, 'train_records.txt'), rec_train)
    write_list(os.path.join(args.split_dir, 'val_records.txt'),   rec_val)
    write_list(os.path.join(args.split_dir, 'test_records.txt'),  rec_test)

    # Small summary
    def summarize(name, recs):
        cnt = Counter(record_rhythm[r] for r in recs)
        return dict(sorted(cnt.items()))

    summary = {
        'counts': {
            'train': summarize('train', rec_train),
            'val':   summarize('val',   rec_val),
            'test':  summarize('test',  rec_test)
        },
        'num_records': {'train': len(rec_train), 'val': len(rec_val), 'test': len(rec_test)},
        'rhythm_order': RHYTHMS
    }
    with open(os.path.join(args.split_dir, 'split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Done. Wrote splits to', args.split_dir)

if __name__ == '__main__':
    main()
