"""
Extended benchmark runner (revision experiments).

Protocol
  * 5-fold stratified CV x 2 repeats on every dataset
  * features standardized on the training fold; resampling on the training fold only
  * 5 classifiers: kNN (k=5), CART, SVM-RBF, Random Forest (100), LightGBM (200)
  * metrics: multiclass G-mean (geometric mean of class recalls; primary),
    one-vs-rest G-mean (the metric used in submission v1), balanced accuracy,
    macro-F1, MCC, macro one-vs-rest AUC
  * a resampler that raises is recorded with failed=1 and the fold falls back to
    the original training data (failure counts are reported in the paper)

Results are checkpointed per (dataset, repeat, fold) as CSV in --out, so the run
can be interrupted and resumed.

  python run_extended_benchmark.py --split eval --out results_ext --jobs 3
"""
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score, matthews_corrcoef,
                             roc_auc_score, recall_score)

import baselines
import benchmark_datasets as bd

N_SPLITS, N_REPEATS = 5, 2

# The 17 baselines and MIRT (registered as 'MIRT-v1', i.e. mirtplus.MIRTPlusResampler).
EVAL_METHODS = {name: (name, None) for name in baselines.BASELINES}
EVAL_METHODS['MIRT-v1'] = ('MIRT-v1', None)


def classifiers(seed):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    return {
        'KNN': KNeighborsClassifier(5),
        'CART': DecisionTreeClassifier(random_state=seed),
        'SVM': SVC(C=1.0, gamma='scale', random_state=seed),
        'RF': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        'LGBM': LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed,
                               n_jobs=1, verbose=-1),
    }


def _scores(clf, X):
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)
    s = clf.decision_function(X)
    return s if s.ndim == 2 else np.c_[-s, s]


def metrics(y_true, y_pred, S, classes):
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    from imblearn.metrics import geometric_mean_score
    out = {
        'gmean': float(np.prod(rec) ** (1.0 / len(rec))),
        'gmean_ovr': geometric_mean_score(y_true, y_pred, average='macro'),
        'bacc': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'min_recall': float(rec.min()),
    }
    aucs = []
    for i, c in enumerate(classes):
        yb = (y_true == c).astype(int)
        if 0 < yb.sum() < len(yb):
            aucs.append(roc_auc_score(yb, S[:, i]))
    out['auc'] = float(np.mean(aucs)) if aucs else np.nan
    return out


def run_task(dname, rep, fold, tr, te, methods, out_dir, registry, deadline=None, label_noise=0.0,
             classifier_names=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{dname}__r{rep}f{fold}.csv')
    if os.path.exists(path):
        return path
    if deadline is not None and time.time() > deadline:   # time budget spent: leave for next session
        return None
    warnings.filterwarnings('ignore')
    X, y = bd.load(dname, registry)
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr].copy(), y[te]
    classes = np.unique(y)
    seed = 1000 * rep + fold
    if label_noise > 0:                    # flip a fraction of TRAINING labels to another class
        nrng = np.random.default_rng(seed + 7)
        flip = nrng.random(len(ytr)) < label_noise
        for i in np.where(flip)[0]:
            ytr[i] = nrng.choice(classes[classes != ytr[i]])
    rows = []
    for mname, (base, params) in methods.items():
        failed, t0 = 0, time.perf_counter()
        try:
            s = baselines.make(base, seed, ytr, params)
            Xr, yr = (Xtr, ytr) if s is None else s.fit_resample(Xtr, ytr)
            Xr, yr = np.asarray(Xr, dtype=float), np.asarray(yr).astype(int)
            if not np.isfinite(Xr).all() or len(np.unique(yr)) < len(classes):
                raise ValueError('degenerate resample')
        except Exception as e:                     # recorded, fall back to original data
            failed, Xr, yr = 1, Xtr, ytr
        t_res = time.perf_counter() - t0
        for cname, clf in classifiers(seed).items():
            if classifier_names and cname not in classifier_names:
                continue
            try:
                clf.fit(Xr, yr)
                yp = clf.predict(Xte)
                S = _scores(clf, Xte)
                m = metrics(yte, yp, S, classes)
            except Exception:
                m = {k: np.nan for k in ('gmean', 'gmean_ovr', 'bacc', 'f1', 'mcc', 'min_recall', 'auc')}
            rows.append({'dataset': dname, 'rep': rep, 'fold': fold, 'method': mname,
                         'clf': cname, 'failed': failed, 'resample_s': t_res, 'label_noise': label_noise,
                         'n_train_res': len(yr), **m})
    tmp = path + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['eval', 'dev'], default='eval')
    ap.add_argument('--repeats', type=int, default=N_REPEATS)
    ap.add_argument('--only', default=None, help='comma-separated method subset')
    ap.add_argument('--datasets', default=None, help='comma-separated dataset subset')
    ap.add_argument('--out', default=os.path.join(_HERE, 'results_ext'))
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--max-hours', type=float, default=None,
                    help='stop starting new folds after this many hours (e.g. 11 on Kaggle)')
    a = ap.parse_args()
    deadline = time.time() + 3600 * a.max_hours if a.max_hours else None

    registry = bd.EVAL_DATASETS if a.split == 'eval' else bd.DEV_DATASETS
    methods = dict(EVAL_METHODS)
    if a.only:
        keep = [s.strip() for s in a.only.split(',')]
        methods = {k: v for k, v in methods.items() if k in keep}
    names = list(registry) if not a.datasets else [s.strip() for s in a.datasets.split(',')]
    os.makedirs(a.out, exist_ok=True)

    tasks = []
    for dname in names:
        X, y = bd.load(dname, registry)
        rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=a.repeats, random_state=42)
        for i, (tr, te) in enumerate(rskf.split(X, y)):
            tasks.append((dname, i // N_SPLITS, i % N_SPLITS, tr, te))
    # largest datasets first so the pool stays busy at the end
    tasks.sort(key=lambda t: -len(t[3]))
    print(f'{len(names)} datasets, {len(methods)} methods, {len(tasks)} folds -> {a.out}', flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    done = 0
    for _ in Parallel(n_jobs=a.jobs, return_as='generator_unordered', verbose=0)(
            delayed(run_task)(d, r, f, tr, te, methods, a.out, registry, deadline)
            for d, r, f, tr, te in tasks):
        done += 1
        if done % 10 == 0 or done == len(tasks):
            el = time.time() - t0
            print(f'  {done}/{len(tasks)} folds  {el/60:.1f} min elapsed, '
                  f'~{el/done*(len(tasks)-done)/60:.0f} min left', flush=True)

    files = [os.path.join(a.out, f) for f in os.listdir(a.out) if f.endswith('.csv') and '__' in f]
    pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(a.out, 'ALL_RESULTS.csv'), index=False)
    left = len(tasks) - len(files)
    print(f'merged {len(files)}/{len(tasks)} folds ->', os.path.join(a.out, 'ALL_RESULTS.csv'))
    print('RUN COMPLETE' if left == 0 else f'INCOMPLETE: {left} folds left - rerun to resume', flush=True)


if __name__ == '__main__':
    main()
