"""
MIRT ablation on the 32 corrected EVAL datasets (paper Section 4.7).

Same component switches as the original run_ablation.py (mirtplus_ablation.MIRTPlusAblate),
but the corrected dataset registry, 5 classifiers and the metrics of the main benchmark
(run_extended_benchmark.run_task). 5-fold stratified CV x 1 repeat.
  python run_ablation_corrected.py --jobs 3
"""
import argparse
import os
import time
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import benchmark_datasets as bd
import run_extended_benchmark as R

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {
    'Full': {},
    '-FeatWeights': dict(feat_weights=False),
    '-Subgroups': dict(subgroups=False),
    '-Mahalanobis': dict(mahalanobis=False),
    '-NoiseGate': dict(noise_gate=False),
    '-ConfSteer': dict(conf_steer=False),
}
METHODS = {v: ('MIRT-v1-ablate', kw) for v, kw in VARIANTS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=3)
    a = ap.parse_args()
    out = os.path.join(HERE, 'results_ablation')
    jobs = []
    for d in bd.EVAL_DATASETS:
        X, y = bd.load(d, bd.EVAL_DATASETS)
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
            jobs.append(delayed(R.run_task)(d, 0, f, tr, te, METHODS, out, bd.EVAL_DATASETS))
    print('ablation', len(jobs), 'folds', flush=True)
    t0 = time.time()
    for i, _ in enumerate(Parallel(n_jobs=a.jobs, return_as='generator_unordered')(jobs), 1):
        if i % 20 == 0 or i == len(jobs):
            print(f'  {i}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min', flush=True)
    fs = [os.path.join(out, f) for f in os.listdir(out) if '__' in f and f.endswith('.csv')]
    pd.concat(map(pd.read_csv, fs)).to_csv(os.path.join(out, 'ALL_RESULTS.csv'), index=False)
    print('ABLATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
