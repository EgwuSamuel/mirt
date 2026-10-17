"""
Reproduction check: re-run selected folds from scratch and compare them with the committed results.

For each (dataset, repeat, fold) the fold is recomputed with run_extended_benchmark.run_task into a
temporary folder and every metric is compared with results_ext/ (or results_ablation/). Optionally,
point SCIKIT_LEARN_DATA at an empty folder first so the datasets are re-downloaded from OpenML.

It also checks that no resampler modifies the training data passed to it (all methods of a fold share
the same standardized training array, so an in-place change would silently affect later methods).

  python reproduce_check.py                         # default sample of folds covering every method
  python reproduce_check.py --folds glass:0:0,car:1:3 --jobs 2
"""
import argparse
import os
import shutil
import sys
import tempfile
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines                              # noqa: E402
import benchmark_datasets as bd               # noqa: E402
import run_extended_benchmark as R            # noqa: E402

DEFAULT = 'glass:0:0,car:1:3,cardiotocography:0:0,page-blocks:0:2,kr-vs-k:1:1,splice:0:4'
METRICS = ['gmean', 'gmean_ovr', 'bacc', 'f1', 'mcc', 'min_recall', 'auc']


def split_for(dname, rep, fold, ablation):
    X, y = bd.load(dname, bd.EVAL_DATASETS)
    if ablation:                                   # run_ablation_corrected.py: 5-fold x 1
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
        return splits[fold]
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y))
    return splits[rep * 5 + fold]


def compare(new, old):
    key = ['method', 'clf']
    m = old.merge(new, on=key, suffixes=('_old', '_new'), how='outer', indicator=True)
    missing = int((m['_merge'] != 'both').sum())
    diffs = {k: float(np.nanmax(np.abs(m[f'{k}_old'] - m[f'{k}_new']))) for k in METRICS}
    same_fail = bool((m['failed_old'] == m['failed_new']).all())
    same_n = bool((m['n_train_res_old'] == m['n_train_res_new']).all())
    return missing, diffs, same_fail, same_n


def inplace_check(dname, rep, fold):
    tr, _ = split_for(dname, rep, fold, False)
    X, y = bd.load(dname, bd.EVAL_DATASETS)
    Xtr = StandardScaler().fit(X[tr]).transform(X[tr]); ytr = y[tr].copy()
    bad = []
    for mname, (base, params) in R.EVAL_METHODS.items():
        s = baselines.make(base, 1000 * rep + fold, ytr, params)
        if s is None:
            continue
        X0, y0 = Xtr.copy(), ytr.copy()
        try:
            s.fit_resample(Xtr, ytr)
        except Exception:
            pass
        if not (np.array_equal(X0, Xtr) and np.array_equal(y0, ytr)):
            bad.append(mname)
            Xtr, ytr = X0, y0
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default=DEFAULT, help='dataset:repeat:fold,...')
    ap.add_argument('--ablation-folds', default='glass:0:0,yeast:0:3')
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='mirt_repro_')
    print('OpenML cache:', os.environ.get('SCIKIT_LEARN_DATA', '(default)'), '| temp:', tmp, flush=True)

    jobs, plan = [], []
    for kind, spec, methods, ref in [
            ('main', a.folds, R.EVAL_METHODS, 'results_ext'),
            ('ablation', a.ablation_folds,
             {v: ('MIRT-v1-ablate', kw) for v, kw in __import__('run_ablation_corrected').VARIANTS.items()},
             'results_ablation')]:
        for item in filter(None, spec.split(',')):
            d, r, f = item.split(':'); r, f = int(r), int(f)
            tr, te = split_for(d, r, f, kind == 'ablation')
            out = os.path.join(tmp, kind)
            jobs.append(delayed(R.run_task)(d, r, f, tr, te, methods, out, bd.EVAL_DATASETS))
            plan.append((kind, d, r, f, os.path.join(out, f'{d}__r{r}f{f}.csv'),
                         os.path.join(HERE, ref, f'{d}__r{r}f{f}.csv')))
    Parallel(n_jobs=a.jobs)(jobs)

    ok = True
    print(f"\n{'kind':9s} {'fold':28s} {'rows':>5s} {'max |diff| over all metrics':>28s}  failed  n_res")
    for kind, d, r, f, new_p, old_p in plan:
        new, old = pd.read_csv(new_p), pd.read_csv(old_p)
        missing, diffs, same_fail, same_n = compare(new, old)
        worst = max(diffs.values())
        good = missing == 0 and worst <= 1e-9 and same_fail and same_n
        ok &= good
        print(f"{kind:9s} {f'{d} r{r}f{f}':28s} {len(new):5d} {worst:28.2e}  {str(same_fail):6s}  {same_n}"
              f"  {'IDENTICAL' if good else 'DIFFERENT: ' + str({k: v for k, v in diffs.items() if v > 1e-9})}")

    d0, r0, f0 = a.folds.split(',')[0].split(':')
    bad = inplace_check(d0, int(r0), int(f0))
    print('\nresamplers that modify their input in place:', bad or 'none')
    ok &= not bad
    shutil.rmtree(tmp, ignore_errors=True)
    print('\nREPRODUCTION CHECK', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
