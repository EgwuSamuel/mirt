"""
Worked example (paper, worked-example section): what ESDA tells a practitioner about a dataset.

For satimage and ecoli this script reports
  * the ESDA similarity matrix and the 5-fold cross-validated CART confusion matrix, computed exactly
    as in esda_validity_corrected.py (so the Pearson r equals T_esda_confusability.csv);
  * the ESDA feature weights (for satimage aggregated by spectral band);
  * the difficulty profile of every class (safe / borderline / rare / outlier, k = 5, in ESDA's
    feature-weighted space, as MIRT computes it);
  * the per-class recall of no resampling, SMOTE and MIRT on the benchmark protocol (same 5x2 folds,
    seeds and classifiers as run_extended_benchmark.py; the G-mean recomputed from these recalls is
    checked against results_ext/ALL_RESULTS.csv).
Output: results_ext/tables/T_worked_<dataset>_{esda,confusion,classes,features}.csv
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines                       # noqa: E402
import benchmark_datasets as bd        # noqa: E402
import run_extended_benchmark as R     # noqa: E402
from mirtplus import compute_enhanced_similarity, difficulty_types, similarity_vs_confusability  # noqa: E402

OUT = os.path.join(HERE, 'results_ext', 'tables')
# Class descriptions from the UCI documentation of the two datasets.
CLASSES = {
    'satimage': {'1.': 'red soil', '2.': 'cotton crop', '3.': 'grey soil', '4.': 'damp grey soil',
                 '5.': 'vegetation stubble', '7.': 'very damp grey soil'},
    'ecoli': {'cp': 'cytoplasm', 'im': 'inner membrane', 'imU': 'inner membrane, uncleavable signal',
              'om': 'outer membrane', 'pp': 'periplasm'},
}
SAT_BANDS = ['green', 'red', 'near-infrared 1', 'near-infrared 2']   # Landsat MSS bands 1-4
METHODS = ['NoResample', 'SMOTE', 'MIRT-v1']


def class_labels(dname):
    """Raw OpenML labels in the order of benchmark_datasets.load (LabelEncoder over kept classes)."""
    did, _ = bd.EVAL_DATASETS[dname]
    y = bd._target_series(fetch_openml(data_id=did, as_frame=True, parser='auto')).astype(str)
    vc = y.value_counts()
    return sorted(vc[vc >= bd.MIN_CLASS].index)


def per_class_recall(dname, X, y):
    classes = np.unique(y)
    rec, gm = {}, []
    rskf = RepeatedStratifiedKFold(n_splits=R.N_SPLITS, n_repeats=R.N_REPEATS, random_state=42)
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        rep, fold = divmod(i, R.N_SPLITS)
        seed = 1000 * rep + fold
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        for m in METHODS:
            s = baselines.make(m, seed, ytr)
            Xr, yr = (Xtr, ytr) if s is None else s.fit_resample(Xtr, ytr)
            for cname, clf in R.classifiers(seed).items():
                r = recall_score(yte, clf.fit(Xr, yr).predict(Xte), labels=classes, average=None,
                                 zero_division=0)
                rec.setdefault(m, []).append(r)
                gm.append({'rep': rep, 'fold': fold, 'method': m, 'clf': cname,
                           'gmean': float(np.prod(r) ** (1 / len(r)))})
    # same numbers as the benchmark?
    ref = pd.read_csv(os.path.join(HERE, 'results_ext', 'ALL_RESULTS.csv'))
    ref = ref[(ref.dataset == dname) & ref.method.isin(METHODS)]
    chk = ref.merge(pd.DataFrame(gm), on=['rep', 'fold', 'method', 'clf'], suffixes=('', '_new'))
    assert len(chk) == len(gm) and np.allclose(chk.gmean, chk.gmean_new, atol=1e-12), 'protocol mismatch'
    return {m: np.mean(v, axis=0) for m, v in rec.items()}


def main():
    for dname in CLASSES:
        X, y = bd.load(dname, bd.EVAL_DATASETS)
        raw = class_labels(dname)
        assert set(raw) == set(CLASSES[dname]), raw
        names = [CLASSES[dname][r] for r in raw]
        classes = list(range(len(raw)))

        mu, cl, w = compute_enhanced_similarity(X, y)
        cm = confusion_matrix(y, cross_val_predict(DecisionTreeClassifier(random_state=42), X, y, cv=5),
                              labels=cl).astype(float)
        conf = cm / cm.sum(axis=1, keepdims=True)
        r_p, _ = similarity_vs_confusability(X, y, mu, cl)
        ref = pd.read_csv(os.path.join(OUT, 'T_esda_confusability.csv')).set_index('dataset')
        assert abs(r_p - ref.loc[dname, 'pearson_r']) < 1e-3, (r_p, ref.loc[dname, 'pearson_r'])

        pd.DataFrame([[mu[a][b] for b in cl] for a in cl], index=names, columns=names).round(4) \
            .to_csv(os.path.join(OUT, f'T_worked_{dname}_esda.csv'))
        pd.DataFrame(conf, index=names, columns=names).round(4) \
            .to_csv(os.path.join(OUT, f'T_worked_{dname}_confusion.csv'))

        Xw = StandardScaler().fit_transform(X) * np.sqrt(w)
        types = difficulty_types(Xw, y, k=5)[0]
        rec = per_class_recall(dname, X, y)
        rows = []
        for c, name in zip(classes, names):
            t = types[y == c]
            rows.append({'class': name, 'label': raw[c], 'n': int((y == c).sum()),
                         **{f'share_{k}': float(np.mean(t == k)) for k in 'SBRO'},
                         **{f'recall_{m}': float(rec[m][c]) for m in METHODS}})
        pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, f'T_worked_{dname}_classes.csv'), index=False)

        if dname == 'satimage':
            # UCI order: 9 pixels x 4 bands, pixel-major. Check it: the same band in adjacent pixels is
            # more correlated than adjacent bands of one pixel.
            C = np.corrcoef(X, rowvar=False)
            same_band = np.mean([C[i, i + 4] for i in range(32)])
            same_pixel = np.mean([C[i, i + 1] for i in range(36) if i % 4 != 3])
            assert same_band > same_pixel, (same_band, same_pixel)
            feat = pd.DataFrame({'feature': SAT_BANDS, 'weight': [w[b::4].sum() for b in range(4)]})
        else:
            did, _ = bd.EVAL_DATASETS[dname]
            cols = list(fetch_openml(data_id=did, as_frame=True, parser='auto').data.columns)
            feat = pd.DataFrame({'feature': cols, 'weight': w})
        feat.sort_values('weight', ascending=False).round(4) \
            .to_csv(os.path.join(OUT, f'T_worked_{dname}_features.csv'), index=False)
        print(f'{dname}: r={r_p:.3f}, outputs written', flush=True)


if __name__ == '__main__':
    main()
