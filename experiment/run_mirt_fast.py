"""
Accuracy of the cheaper-ESDA setting (mirt_fast.py) on the benchmark protocol: same 32 datasets,
5x2 folds, seeds, classifiers and metrics as run_extended_benchmark.py (run_task is reused).
Output: results_fast/ (one CSV per fold, ALL_RESULTS.csv). Compare with MIRT-v1 in results_ext/.
  python run_mirt_fast.py --jobs 3
"""
import argparse
import os
import sys

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchmark_datasets as bd       # noqa: E402
import run_extended_benchmark as R    # noqa: E402

METHODS = {'MIRT-fast': ('MIRT-fast', None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--out', default=os.path.join(HERE, 'results_fast'))
    a = ap.parse_args()
    tasks = []
    for d in bd.EVAL_DATASETS:
        X, y = bd.load(d, bd.EVAL_DATASETS)
        rskf = RepeatedStratifiedKFold(n_splits=R.N_SPLITS, n_repeats=R.N_REPEATS, random_state=42)
        for i, (tr, te) in enumerate(rskf.split(X, y)):
            tasks.append((d, i // R.N_SPLITS, i % R.N_SPLITS, tr, te))
    tasks.sort(key=lambda t: -len(t[3]))
    Parallel(n_jobs=a.jobs, verbose=5)(delayed(R.run_task)(d, r, f, tr, te, METHODS, a.out, bd.EVAL_DATASETS)
                                       for d, r, f, tr, te in tasks)
    files = sorted(os.path.join(a.out, f) for f in os.listdir(a.out) if f.endswith('.csv') and '__' in f)
    pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(a.out, 'ALL_RESULTS.csv'), index=False)
    print(f'merged {len(files)}/{len(tasks)} folds')


if __name__ == '__main__':
    main()
