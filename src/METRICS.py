import numpy as np
from CONFIG import FS, RHYTHMS, BOUNDARY_TYPES

class BoundaryStats:
    def __init__(self):
        self.stats = {r: {bt[0]: {'TP': 0, 'FP': 0, 'FN': 0, 'errs': []} for bt in BOUNDARY_TYPES}
                      for r in range(len(RHYTHMS))}

    def add(self, rhythm_id, boundary_name, TP, FP, FN, errs):
        s = self.stats[rhythm_id][boundary_name]
        s['TP'] += TP; s['FP'] += FP; s['FN'] += FN; s['errs'].extend(errs)

    def finalize(self, tol_ms):
        def _row_from_counts(d):
            TP, FP, FN = d['TP'], d['FP'], d['FN']
            se = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            f1 = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0
            if len(d['errs']) > 0:
                errs_ms = np.array(d['errs']) * (1000.0 / FS)
                mu = float(np.mean(errs_ms)); sd = float(np.std(errs_ms))
            else:
                mu = sd = 0.0
            return se, ppv, f1, mu, sd

        out = {}
        for r in range(len(RHYTHMS)):
            out[RHYTHMS[r]] = {bn: _row_from_counts(self.stats[r][bn]) for (bn, _, _) in BOUNDARY_TYPES}
        # Macro across rhythms per boundary
        out['macro'] = {}
        for (bn, _, _) in BOUNDARY_TYPES:
            vals = [out[RHYTHMS[r]][bn] for r in range(len(RHYTHMS))]
            se = float(np.mean([v[0] for v in vals]))
            ppv = float(np.mean([v[1] for v in vals]))
            f1 = float(np.mean([v[2] for v in vals]))
            mu = float(np.mean([v[3] for v in vals]))
            sd = float(np.mean([v[4] for v in vals]))
            out['macro'][bn] = (se, ppv, f1, mu, sd)
        return out