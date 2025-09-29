import numpy as np
from CONFIG import FS, CLASS_BG, CLASS_P, CLASS_QRS, CLASS_T, BOUNDARY_TYPES

def _runs_of_class(labels):
    L = len(labels)
    runs = []
    if L == 0: return runs
    s = 0; cur = labels[0]
    for i in range(1, L):
        if labels[i] != cur:
            runs.append((s, i, cur))
            s = i; cur = labels[i]
    runs.append((s, L, cur))
    return runs

def _apply_short_blob_fix(y, min_len):
    L = len(y)
    if L == 0: return y
    runs = _runs_of_class(y)
    for (s, e, c) in runs:
        if c == CLASS_BG: continue
        if e - s < min_len:
            prev_c = y[s-1] if s > 0 else None
            next_c = y[e] if e < L else None
            if prev_c is not None and next_c is not None and prev_c == next_c and prev_c != c:
                y[s:e] = prev_c
            else:
                y[s:e] = CLASS_BG
    return y

def _enforce_one_P_T_between_QRS(y):
    runs = _runs_of_class(y)
    qrs_runs = [ (s,e,c) for (s,e,c) in runs if c == CLASS_QRS ]
    if len(qrs_runs) < 2: return y
    for i in range(len(qrs_runs)-1):
        left_end = qrs_runs[i][1]
        right_start = qrs_runs[i+1][0]
        if right_start <= left_end: continue
        seg_runs = _runs_of_class(y[left_end:right_start])
        seg_runs = [(left_end + s, left_end + e, c) for (s,e,c) in seg_runs]
        # Longest P
        p_runs = [(s,e) for (s,e,c) in seg_runs if c == CLASS_P]
        if len(p_runs) > 1:
            lengths = [e - s for (s,e) in p_runs]
            keep_idx = np.argmax(lengths)
            for j, (s,e) in enumerate(p_runs):
                if j != keep_idx:
                    y[s:e] = CLASS_BG
        # Longest T
        t_runs = [(s,e) for (s,e,c) in seg_runs if c == CLASS_T]
        if len(t_runs) > 1:
            lengths = [e - s for (s,e) in t_runs]
            keep_idx = np.argmax(lengths)
            for j, (s,e) in enumerate(t_runs):
                if j != keep_idx:
                    y[s:e] = CLASS_BG
    return y

def post_process(y_pred_1d, min_len_ms):
    y = y_pred_1d.copy()
    min_len = max(1, int(round(min_len_ms * FS / 1000.0)))
    y = _apply_short_blob_fix(y, min_len)
    y = _enforce_one_P_T_between_QRS(y)
    return y

def extract_boundaries(labels):
    d = {bt[0]: [] for bt in BOUNDARY_TYPES}
    L = len(labels)
    for cname, c, side in BOUNDARY_TYPES:
        if side == "on":
            idxs = []
            prev = CLASS_BG
            for i in range(L):
                cur = labels[i]
                if cur == c and prev != c:
                    idxs.append(i)
                prev = cur
            d[cname] = idxs
        else:
            idxs = []
            for i in range(L-1, -1, -1):
                cur = labels[i]
                if cur == c and (i == L-1 or labels[i+1] != c):
                    idxs.append(i)
            idxs = idxs[::-1]  # make ascending
            d[cname] = idxs
    return d

def match_events(gt_idxs, pr_idxs, tol_samples):
    gt_idxs = list(gt_idxs); pr_idxs = list(pr_idxs)
    gt_taken = [False] * len(gt_idxs)
    TP = 0; errors = []
    for p in pr_idxs:
        best_j = -1; best_dist = None
        for j, g in enumerate(gt_idxs):
            if gt_taken[j]: continue
            dist = abs(p - g)
            if dist <= tol_samples and (best_dist is None or dist < best_dist):
                best_dist = dist; best_j = j
        if best_j >= 0:
            TP += 1
            gt_taken[best_j] = True
            errors.append(p - gt_idxs[best_j])
    FP = len(pr_idxs) - TP
    FN = len(gt_idxs) - TP
    return TP, FP, FN, errors