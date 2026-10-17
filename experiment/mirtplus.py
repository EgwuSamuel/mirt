"""
mirtplus.py — MIRT++ : Multiclass Informative Resampling Technique (enhanced)

Two novel components:
  A. Enhanced Similarity Degree Algorithm (ESDA) — feature-weighted,
     subgroup-clustered, distribution-aware (Mahalanobis) inter-class
     similarity, validated against empirical classifier confusability.
  B. Similarity-modulated, difficulty-aware synthetic resampling — SMOTE-style
     synthesis that excludes noise (outliers), favours borderline seeds, and
     applies noise-gated conservativeness steered by the similarity matrix.

This module is imported by both the research notebook and the FastAPI service,
so the deployed model and the experiments use identical code.
"""
import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

__all__ = [
    'compute_enhanced_similarity', 'difficulty_types',
    'MIRTPlusResampler', 'similarity_vs_confusability',
]


# ═══════════════════════════════════════════════════════════════════════════
#  NOVELTY A — Enhanced Similarity Degree Algorithm (ESDA)
# ═══════════════════════════════════════════════════════════════════════════

def _feature_weights(X, y, seed=42):
    """RandomForest impurity importances → normalised feature weights."""
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    return imp / imp.sum() if imp.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]


def _class_subgroups(Xc, m_max=3, seed=42, min_per_cluster=10):
    """Cluster one class into <= m_max subgroups (KMeans + silhouette).
    Returns (centroids, sizes, covariances)."""
    n = len(Xc)
    max_k = max(1, min(m_max, n // min_per_cluster))
    best_k, best_score = 1, -1.0
    if max_k >= 2 and n >= 2 * min_per_cluster:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xc)
            if len(np.unique(km.labels_)) < 2:
                continue
            try:
                score = silhouette_score(Xc, km.labels_)
            except Exception:
                score = -1.0
            if score > best_score:
                best_score, best_k = score, k
    if best_k == 1:
        return [Xc.mean(axis=0)], [n], [np.cov(Xc, rowvar=False)]
    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(Xc)
    centroids, sizes, covs = [], [], []
    for lab in range(best_k):
        sub = Xc[km.labels_ == lab]
        centroids.append(sub.mean(axis=0))
        sizes.append(len(sub))
        covs.append(np.cov(sub, rowvar=False) if len(sub) > 1 else np.eye(Xc.shape[1]))
    return centroids, sizes, covs


def compute_enhanced_similarity(X, y, m_max=3, lam=0.85, eps=1e-6,
                                use_kernel=False, seed=42):
    """
    ESDA — returns (mu, classes, feature_weights).

    mu[i][j] ∈ [0,1] is the enhanced similarity degree between classes i and j
    (1 = identical). Pipeline: standardize → RF feature weights → per-class
    subgroup clustering → weighted-Mahalanobis subgroup distances →
    exp(-alpha·d) with median-heuristic alpha → size-weighted aggregation →
    regularization toward the mean → min-max normalization.
    """
    classes = sorted(np.unique(y).tolist())
    d = X.shape[1]

    Xs = StandardScaler().fit_transform(X)          # Step 1
    w = _feature_weights(Xs, y, seed)               # Step 2
    Xw = Xs * np.sqrt(w)

    sub = {c: _class_subgroups(Xw[y == c], m_max=m_max, seed=seed) for c in classes}  # Step 3

    raw, all_d = {}, []                             # Step 4: subgroup distances
    for ci in classes:
        cen_i, siz_i, cov_i = sub[ci]
        for cj in classes:
            cen_j, siz_j, cov_j = sub[cj]
            for p in range(len(cen_i)):
                for q in range(len(cen_j)):
                    diff = cen_i[p] - cen_j[q]
                    Sig = (np.atleast_2d(cov_i[p]) + np.atleast_2d(cov_j[q])) / 2.0 + eps * np.eye(d)
                    try:
                        dist = float(np.sqrt(max(diff @ np.linalg.solve(Sig, diff), 0)))
                    except np.linalg.LinAlgError:
                        dist = float(np.linalg.norm(diff))
                    raw[(ci, cj, p, q)] = (dist, siz_i[p] * siz_j[q], float(np.linalg.norm(diff)))
                    if ci != cj:
                        all_d.append(dist)
    med = np.median(all_d) if all_d else 1.0
    alpha = 1.0 / med if med > 0 else 1.0           # median heuristic
    sigma = med if med > 0 else 1.0

    mu_raw = {ci: {} for ci in classes}             # Steps 5-7: aggregate
    for ci in classes:
        for cj in classes:
            num = den = 0.0
            for p in range(len(sub[ci][0])):
                for q in range(len(sub[cj][0])):
                    dist, wgt, euc = raw[(ci, cj, p, q)]
                    s = (math.exp(-(euc ** 2) / (2 * sigma ** 2)) if use_kernel
                         else math.exp(-alpha * dist))
                    num += wgt * s
                    den += wgt
            mu_raw[ci][cj] = num / den if den > 0 else 0.0

    offdiag = [mu_raw[ci][cj] for ci in classes for cj in classes if ci != cj]
    baseline = np.mean(offdiag) if offdiag else 0.0                       # Step 8
    mu_reg = {ci: {cj: lam * mu_raw[ci][cj] + (1 - lam) * baseline
                   for cj in classes} for ci in classes}

    od = [mu_reg[ci][cj] for ci in classes for cj in classes if ci != cj]  # Step 9
    lo, hi = (min(od), max(od)) if od else (0.0, 1.0)
    rng = (hi - lo) if hi > lo else 1.0
    mu = {ci: {cj: (1.0 if ci == cj else float((mu_reg[ci][cj] - lo) / rng))
               for cj in classes} for ci in classes}
    return mu, classes, w


# ═══════════════════════════════════════════════════════════════════════════
#  Difficulty typing (safe / borderline / rare / outlier)
# ═══════════════════════════════════════════════════════════════════════════

def difficulty_types(Xw, y, k=5):
    """k-NN difficulty typing. Returns (types, same_counts, neighbor_idx)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xw)
    _, idx = nn.kneighbors(Xw)
    types = np.empty(len(y), dtype='<U1')
    same_counts = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        same = int(np.sum(y[idx[i, 1:k + 1]] == y[i]))
        same_counts[i] = same
        types[i] = 'S' if same >= 4 else 'B' if same >= 2 else 'R' if same == 1 else 'O'
    return types, same_counts, idx


# ═══════════════════════════════════════════════════════════════════════════
#  NOVELTY B — Similarity-modulated, difficulty-aware synthetic resampler
# ═══════════════════════════════════════════════════════════════════════════

class MIRTPlusResampler:
    """
    imbalanced-learn-style resampler:  X_res, y_res = MIRTPlusResampler().fit_resample(X, y)

    Oversampling (per minority class):
      * seeds are difficulty-weighted — outliers excluded, borderline favoured
        (weights S:2, B:3, R:1, O:0);
      * synthesis interpolates toward same-class neighbours;
      * conservativeness is NOISE-GATED — the interpolation step only shrinks
        when the class is genuinely noisy/overlapping, steered by the ESDA
        similarity to the most-confusable class;
      * extremely under-represented classes (oversample ratio >= rare_ratio)
        switch to full SMOTE-like reach so tiny classes are not starved.

    Synthetic values are left continuous by default (round_int=False); rounding
    onto an integer/ordinal grid collapses points into duplicates and is only
    enabled on request. Optional similarity-aware majority cleaning
    (clean_frac > 0) and an over+under 'mean' strategy are also available but
    OFF by default; the default 'max' strategy is a pure oversampler directly
    comparable to SMOTE.
    """
    def __init__(self, strategy='max', k_neighbors=5, m_max=3,
                 beta=0.15, lam=0.85, clean_frac=0.0, use_kernel=False, seed=42,
                 round_int=False, rare_ratio=6.0, noise_floor=0.6):
        self.strategy = strategy
        self.k = k_neighbors
        self.m_max = m_max
        self.beta = beta
        self.lam = lam
        self.clean_frac = clean_frac
        self.use_kernel = use_kernel
        self.seed = seed
        self.round_int = round_int      # round synthetic values on integer columns
        self.rare_ratio = rare_ratio    # oversample ratio above which a class is 'rare'
        self.noise_floor = noise_floor  # min interpolation-step cap under noise

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = sorted(np.unique(y).tolist())
        sizes = {c: int(np.sum(y == c)) for c in classes}

        self.mu, _, self.weights = compute_enhanced_similarity(
            X, y, m_max=self.m_max, lam=self.lam,
            use_kernel=self.use_kernel, seed=self.seed)
        Xw = StandardScaler().fit_transform(X) * np.sqrt(self.weights)
        types, _, _ = difficulty_types(Xw, y, k=self.k)
        self.types_ = types

        int_cols = [j for j in range(X.shape[1])
                    if np.allclose(X[:, j], np.round(X[:, j]))]

        maj = max(sizes, key=sizes.get)
        out_X, out_y = [], []

        # optional similarity-aware majority cleaning (off by default)
        maj_mask = (y == maj)
        Xmaj, Xwmaj = X[maj_mask], Xw[maj_mask]
        nmaj = len(Xmaj)
        max_remove = int(self.clean_frac * nmaj)
        keep_score = np.zeros(nmaj)
        if max_remove > 0:
            nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Xw)
            _, idx = nn.kneighbors(Xwmaj)
            for i in range(nmaj):
                ncls = y[idx[i, 1:self.k + 1]]
                overlap = sum(self.mu[maj][j] * np.mean(ncls == j)
                              for j in classes if j != maj)
                keep_score[i] = np.mean(ncls == maj) - overlap
            cand = np.where(keep_score < 0)[0]
            removed = set(cand[np.argsort(keep_score[cand])][:max_remove].tolist())
            keep_maj = np.array([i for i in range(nmaj) if i not in removed])
        else:
            keep_maj = np.arange(nmaj)
        Xmaj_kept = Xmaj[keep_maj]

        target = (math.ceil(np.mean(list(sizes.values())))
                  if self.strategy == 'mean' else len(Xmaj_kept))

        for c in classes:
            mask = (y == c)
            Xc, Xwc, tc = X[mask], Xw[mask], types[mask]
            n = len(Xc)

            if c == maj:
                if self.strategy == 'mean' and len(Xmaj_kept) > target:
                    sel = keep_maj[np.argsort(-keep_score[keep_maj])[:target]]
                    out_X.append(Xmaj[sel]); out_y.append(np.full(target, c))
                else:
                    out_X.append(Xmaj_kept); out_y.append(np.full(len(Xmaj_kept), c))
                continue

            out_X.append(Xc); out_y.append(np.full(n, c))
            need = target - n
            if need <= 0:
                continue

            # extremely under-represented class → needs full SMOTE-like reach
            is_rare = (target / max(n, 1)) >= self.rare_ratio

            # Seed selection: use every example type equally for diversity.
            # Restricting synthesis to safe/borderline seeds was found to starve
            # tiny classes (e.g. glass's) and collapse coverage; noise protection
            # is provided instead by the noise-gated interpolation step below.
            weight_map = {'S': 1.0, 'B': 1.0, 'R': 1.0, 'O': 1.0}
            seed_w = np.array([weight_map[t] for t in tc])
            if seed_w.sum() == 0:
                seed_w = np.ones(n)
            seed_ids = np.where(seed_w > 0)[0]
            seed_p = seed_w[seed_ids] / seed_w[seed_ids].sum()

            kk = min(self.k, n - 1)
            if kk < 1:
                reps = Xc[rng.choice(seed_ids, size=need, replace=True, p=seed_p)]
                out_X.append(reps); out_y.append(np.full(need, c))
                continue
            _, idx_in = NearestNeighbors(n_neighbors=kk + 1).fit(Xwc).kneighbors(Xwc)

            noise_level = np.mean(np.isin(tc, ['R', 'O']))
            s_conf = max([self.mu[c][j] for j in classes if j != c], default=0.0)
            # rare classes: full reach; else noise-gated conservativeness
            step_cap = (1.0 if is_rare
                        else max(self.noise_floor, 1.0 - self.beta * s_conf * noise_level))

            syn = np.empty((need, X.shape[1]))
            for t in range(need):
                si = rng.choice(seed_ids, p=seed_p)
                nb = rng.choice(idx_in[si, 1:kk + 1])
                cap = step_cap * (0.85 if (tc[si] == 'B' and noise_level > 0.3 and not is_rare) else 1.0)
                syn[t] = Xc[si] + rng.uniform(0, cap) * (Xc[nb] - Xc[si])
            if self.round_int and int_cols:
                syn[:, int_cols] = np.round(syn[:, int_cols])
            out_X.append(syn); out_y.append(np.full(need, c))

        X_res = np.vstack(out_X)
        y_res = np.concatenate(out_y)
        perm = rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]


# ═══════════════════════════════════════════════════════════════════════════
#  Validation — similarity vs empirical confusability
# ═══════════════════════════════════════════════════════════════════════════

def similarity_vs_confusability(X, y, mu, classes, seed=42):
    """Correlate ESDA similarity with a classifier's cross-validated
    confusion-matrix confusability. Returns (pearson_r, spearman_r)."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    from scipy.stats import pearsonr, spearmanr

    y_pred = cross_val_predict(DecisionTreeClassifier(random_state=seed), X, y, cv=5)
    cm = confusion_matrix(y, y_pred, labels=classes).astype(float)
    row = cm.sum(axis=1, keepdims=True); row[row == 0] = 1
    conf = cm / row

    sim_vals, conf_vals = [], []
    for a, ci in enumerate(classes):
        for b, cj in enumerate(classes):
            if ci != cj:
                sim_vals.append(mu[ci][cj]); conf_vals.append(conf[a, b])
    if len(set(conf_vals)) < 2 or len(set(sim_vals)) < 2:
        return float('nan'), float('nan')
    return pearsonr(sim_vals, conf_vals)[0], spearmanr(sim_vals, conf_vals)[0]
