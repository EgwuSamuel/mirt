"""
MIRT with a cheaper ESDA (paper, section on computational cost). mirtplus.py is NOT modified.

Profiling (cost_profile.py) shows that ESDA's random forest and its k-means/silhouette subgroup
search take most of MIRT's time, while difficulty typing and synthesis are cheap. This variant
changes only those two estimators:
  * feature weights from a random forest with N_TREES trees instead of 100;
  * subgroup search with one k-means initialisation instead of ten, the silhouette computed on at
    most SIL_SAMPLE points, and the selected clustering reused instead of being fitted again.
Everything else (distances, aggregation, shrinkage, typing, synthesis, seeds) is MIRT v1.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

import mirtplus

N_TREES = 20
SIL_SAMPLE = 1000


def _feature_weights_fast(X, y, seed=42):
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    return imp / imp.sum() if imp.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]


def _class_subgroups_fast(Xc, m_max=3, seed=42, min_per_cluster=10):
    n = len(Xc)
    max_k = max(1, min(m_max, n // min_per_cluster))
    best_k, best_score, best_km = 1, -1.0, None
    if max_k >= 2 and n >= 2 * min_per_cluster:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init=1).fit(Xc)
            if len(np.unique(km.labels_)) < 2:
                continue
            try:
                score = silhouette_score(Xc, km.labels_,
                                         sample_size=SIL_SAMPLE if n > SIL_SAMPLE else None,
                                         random_state=seed)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score, best_k, best_km = score, k, km
    if best_k == 1:
        return [Xc.mean(axis=0)], [n], [np.cov(Xc, rowvar=False)]
    centroids, sizes, covs = [], [], []
    for lab in range(best_k):
        sub = Xc[best_km.labels_ == lab]
        centroids.append(sub.mean(axis=0))
        sizes.append(len(sub))
        covs.append(np.cov(sub, rowvar=False) if len(sub) > 1 else np.eye(Xc.shape[1]))
    return centroids, sizes, covs


class _FastESDA:
    """Swap ESDA's two estimators in mirtplus for the duration of a call (process-local)."""
    def __enter__(self):
        self.saved = mirtplus._feature_weights, mirtplus._class_subgroups
        mirtplus._feature_weights, mirtplus._class_subgroups = _feature_weights_fast, _class_subgroups_fast

    def __exit__(self, *exc):
        mirtplus._feature_weights, mirtplus._class_subgroups = self.saved


def compute_enhanced_similarity_fast(X, y, **kw):
    with _FastESDA():
        return mirtplus.compute_enhanced_similarity(X, y, **kw)


class MIRTFast(mirtplus.MIRTPlusResampler):
    def fit_resample(self, X, y):
        with _FastESDA():
            return super().fit_resample(X, y)
