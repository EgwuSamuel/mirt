"""
Cost profile of MIRT (paper, section on computational cost) and the cheaper-ESDA setting of mirt_fast.py.

For the first training fold (repeat 0, fold 0, standardised exactly as in the benchmark) of every
dataset, run sequentially in one process:
  * wall-clock time of SMOTE, MIRT and MIRT-fast (median of REPEATS interleaved runs);
  * MIRT's time split into ESDA feature weighting, ESDA subgroup search, the rest of ESDA,
    difficulty typing and synthesis;
  * agreement of the two similarity matrices on that fold (Pearson over ordered class pairs);
  * ESDA validity of the cheaper estimators on the whole dataset, computed as in
    esda_validity_corrected.py (Pearson correlation with cross-validated CART confusion).
Output: results_ext/tables/T_cost_profile.csv
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines                     # noqa: E402
import benchmark_datasets as bd      # noqa: E402
import mirtplus                      # noqa: E402
import mirt_fast                     # noqa: E402

REPEATS = 5
STAGES = {'_feature_weights': 'esda_rf', '_class_subgroups': 'esda_kmeans',
          'compute_enhanced_similarity': 'esda_total', 'difficulty_types': 'typing'}
clock = {}


def _timed(fn, key):
    def wrap(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            clock[key] = clock.get(key, 0.0) + time.perf_counter() - t
    return wrap


def _run(name, X, y):
    s = baselines.make(name, 0, y)
    t = time.perf_counter()
    s.fit_resample(X, y)
    return time.perf_counter() - t, s


def _offdiag(mu, classes):
    return np.array([mu[a][b] for a in classes for b in classes if a != b])


def profile(dname):
    X, y = bd.load(dname, bd.EVAL_DATASETS)
    tr, _ = next(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y))
    Xtr, ytr = StandardScaler().fit(X[tr]).transform(X[tr]), y[tr]
    row = {'dataset': dname, 'n_train': len(ytr), 'd': X.shape[1], 'K': len(np.unique(y))}

    saved = {f: getattr(mirtplus, f) for f in STAGES}
    runs = []
    for _ in range(REPEATS):              # interleaved, so the three methods share background load
        smote_t = _run('SMOTE', Xtr, ytr)[0]
        clock.clear()
        for f, key in STAGES.items():
            setattr(mirtplus, f, _timed(saved[f], key))
        try:
            total, s_full = _run('MIRT-v1', Xtr, ytr)
        finally:
            for f, fn in saved.items():
                setattr(mirtplus, f, fn)
        fast_t, s_fast = _run('MIRT-fast', Xtr, ytr)
        c = dict(clock)
        runs.append({'smote_s': smote_t, 'mirt_s': total, 'fast_s': fast_t,
                     'esda_rf_s': c['esda_rf'], 'esda_kmeans_s': c['esda_kmeans'],
                     'esda_other_s': c['esda_total'] - c['esda_rf'] - c['esda_kmeans'],
                     'typing_s': c['typing'], 'synthesis_s': total - c['esda_total'] - c['typing']})
    row.update(pd.DataFrame(runs).median().to_dict())
    classes = sorted(np.unique(ytr).tolist())
    row['mu_agreement_r'] = pearsonr(_offdiag(s_full.mu, classes), _offdiag(s_fast.mu, classes))[0]

    mu, cl, _ = mirt_fast.compute_enhanced_similarity_fast(X, y)
    row['fast_validity_r'] = mirtplus.similarity_vs_confusability(X, y, mu, cl)[0]
    return row


def main():
    rows = []
    for d in bd.EVAL_DATASETS:
        row = profile(d)
        rows.append(row)
        print(f"  {d:<20s} SMOTE {row['smote_s']:.3f}s  MIRT {row['mirt_s']:.2f}s "
              f"(RF {row['esda_rf_s']:.2f}, k-means {row['esda_kmeans_s']:.2f})  fast {row['fast_s']:.2f}s  "
              f"agreement r={row['mu_agreement_r']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, 'results_ext', 'tables', 'T_cost_profile.csv')
    df.round(4).to_csv(out, index=False)
    print('->', out)


if __name__ == '__main__':
    main()
