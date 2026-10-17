"""
Dataset registry for the extended MIRT benchmark.

Every dataset is fetched from OpenML by its numeric id (names are ambiguous on
OpenML: e.g. data_id=40 is *sonar*, not new-thyroid, and data_id=179 is
*adult*). All characteristics reported in the paper (Table 1) must be produced
by `describe()` from the data actually loaded, never typed in by hand.

Preprocessing (identical for every dataset):
  * nominal attributes -> one-hot; numeric attributes -> median-imputed
  * classes with fewer than MIN_CLASS (=12) examples are dropped so that
    5-fold stratified CV and k-NN synthesis are well defined
  * datasets larger than MAX_N (=4000) are stratified-downsampled with a floor
    of FLOOR examples per class (computational budget; documented in the paper)

Two disjoint splits:
  DEV_DATASETS  - used ONLY to design / select the MIRT v2 configuration
  EVAL_DATASETS - used ONLY for the reported benchmark
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Optional offline cache of the *preprocessed* datasets (one .npz per dataset),
# e.g. for Kaggle/Colab runs without internet: set MIRT_DATA_DIR to its folder.
DATA_DIR = os.environ.get('MIRT_DATA_DIR')
from sklearn.preprocessing import LabelEncoder

MIN_CLASS = 12
MAX_N = 4000
FLOOR = 30

# name -> (openml data_id, special-handling key or None)
EVAL_DATASETS = {
    # carried over from the first submission (ids corrected)
    'new-thyroid':        (40682, None),   # paper v1 wrongly used id 40 (= sonar)
    'hayes-roth':         (329, None),
    'balance-scale':      (11, None),
    'car':                (21, None),
    'cmc':                (23, None),
    'ecoli':              (39, None),
    'glass':              (41, None),
    'wine':               (187, None),
    'dermatology':        (35, None),
    'yeast':              (181, None),
    'winequality-red':    (40691, None),
    'winequality-white':  (40498, None),
    'cleveland':          (194, None),     # paper v1 actually loaded heart-h (id 1565)
    'heart-h':            (1565, None),    # kept, correctly named
    'thyroid-allbp':      (40474, None),   # paper v1 called this "ann-thyroid"
    'abalone':            (183, 'abalone3'),
    'flare':              (46174, None),
    'satimage':           (182, None),
    'splice':             (46, None),
    # added in the revision
    'page-blocks':        (30, None),
    'shuttle':            (40685, None),
    'steel-plates-fault': (40982, None),
    'cardiotocography':   (1466, 'ctg_noleak'),  # V26-V35 = one-hot copy of the class label
    'wall-robot-nav':     (1497, None),
    'autos':              (9, None),
    'fo-theorem-proving': (1475, None),
    'hypothyroid':        (57, None),
    'nursery':            (26, None),
    'kr-vs-k':            (184, None),
    'user-knowledge':     (1508, None),
    'soybean':            (42, None),
    'anneal':             (2, None),
}

DEV_DATASETS = {
    'vertebra-column':    (1523, None),
    'eucalyptus':         (188, None),
    'authorship':         (458, None),
    'flags':              (285, None),
    'baseball':           (185, None),
    'primary-tumor':      (171, None),
    'ESL':                (1027, None),
}

_CACHE = {}


def _target_series(d):
    t = d.target
    if getattr(t, 'ndim', 1) > 1:
        t = t.iloc[:, 0]
    return pd.Series(np.asarray(t).ravel()).astype(str)


def _encode(df):
    df = df.copy()
    num = df.select_dtypes(include=[np.number]).columns
    cat = [c for c in df.columns if c not in num]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.0)
    if cat:
        df[cat] = df[cat].astype(str).replace({'nan': 'missing', 'None': 'missing'})
        df = pd.get_dummies(df, columns=cat, dtype=float)
    X = df.values.astype(float)
    keep = X.std(axis=0) > 0                       # drop constant columns
    return X[:, keep]


def _downsample(X, y, rng):
    n = len(y)
    if n <= MAX_N:
        return X, y
    classes, counts = np.unique(y, return_counts=True)
    # find scale s so that sum(min(n_c, max(FLOOR, s*n_c))) == MAX_N
    lo, hi = 0.0, 1.0
    for _ in range(60):
        s = (lo + hi) / 2
        tot = np.minimum(counts, np.maximum(FLOOR, s * counts)).sum()
        lo, hi = (s, hi) if tot < MAX_N else (lo, s)
    quota = np.minimum(counts, np.maximum(FLOOR, np.floor(lo * counts))).astype(int)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], q, replace=False)
                          for c, q in zip(classes, quota)])
    idx.sort()
    return X[idx], y[idx]


def load(name, registry=None):
    """Return (X, y) as float array / int labels 0..K-1, preprocessed."""
    if name in _CACHE:
        return _CACHE[name]
    if DATA_DIR and os.path.exists(os.path.join(DATA_DIR, f'{name}.npz')):
        z = np.load(os.path.join(DATA_DIR, f'{name}.npz'))
        _CACHE[name] = (z['X'], z['y'])
        return _CACHE[name]
    reg = registry or {**EVAL_DATASETS, **DEV_DATASETS}
    did, special = reg[name]
    d = fetch_openml(data_id=did, as_frame=True, parser='auto')
    y_raw = _target_series(d)
    if special == 'abalone3':          # rings -> 3 age groups, as in the literature
        r = y_raw.astype(int).values
        y_raw = pd.Series(np.where(r <= 8, 'young', np.where(r <= 11, 'adult', 'old')))
    data = d.data
    if special == 'ctg_noleak':        # OpenML 1466 ships the 10 class-indicator columns (A ... SUSP)
        data = data.drop(columns=[f'V{i}' for i in range(26, 36)])
    X = _encode(data)
    vc = y_raw.value_counts()
    mask = y_raw.isin(vc[vc >= MIN_CLASS].index).values
    X, y = X[mask], LabelEncoder().fit_transform(y_raw[mask])
    X, y = _downsample(X, y, np.random.default_rng(0))
    _CACHE[name] = (X, y)
    return X, y


def describe(names, registry=None):
    rows = []
    for n in names:
        X, y = load(n, registry)
        cnt = np.bincount(y)
        rows.append({'dataset': n, 'openml_id': (registry or {**EVAL_DATASETS, **DEV_DATASETS})[n][0],
                     'n': len(y), 'd': X.shape[1], 'K': len(cnt),
                     'IR': round(cnt.max() / cnt.min(), 1),
                     'min_class': int(cnt.min())})
    df = pd.DataFrame(rows).sort_values('IR').reset_index(drop=True)
    df['hard'] = (df['IR'] >= 8) & (df['K'] >= 5)
    return df


def export(folder):
    """Write every preprocessed dataset to <folder>/<name>.npz (offline runs)."""
    os.makedirs(folder, exist_ok=True)
    for n in {**EVAL_DATASETS, **DEV_DATASETS}:
        X, y = load(n)
        np.savez_compressed(os.path.join(folder, f'{n}.npz'), X=X, y=y)


if __name__ == '__main__':
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    if len(sys.argv) > 2 and sys.argv[1] == '--export':
        export(sys.argv[2]); print('exported to', sys.argv[2]); sys.exit()
    print('EVALUATION'); print(describe(EVAL_DATASETS).to_string())
    print('\nDEVELOPMENT'); print(describe(DEV_DATASETS).to_string())
