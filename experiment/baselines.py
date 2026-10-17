"""
Baseline resamplers for the extended benchmark, behind one interface:

    make(name, seed) -> object with fit_resample(X, y) -> (X_res, y_res)

Sources
  imbalanced-learn : SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek, KMeans-SMOTE
  smote-variants   : MWMOTE, ProWSyn, Safe-Level-SMOTE   (Kovacs 2019 reference code,
                     multiclass via its MulticlassOversampling wrapper)
  multi-imbalance  : SOUP, MDO, Static-SMOTE              (vendored, MIT licence)
  authors' code    : MC-CCR (Koziarski et al. 2020)       third_party/mc_ccr_algorithms.py
  knnor package    : KNNOR (Islam et al., Applied Soft Computing 2022), c-vs-rest adaptation
  re-implemented   : MC-RBO (Krawczyk et al. 2020) - vectorised version of the authors'
                     code, same algorithm and defaults (see validate_mcrbo.py)
                     SWIM-Maha (Bellinger et al. 2020) - following the paper
  ours             : MIRT (mirtplus.py); ablation variants (mirtplus_ablation.py)

Input X is assumed standardized on the training fold (done by the runner).
MC-CCR / MC-RBO min-max scale internally, as in the authors' experiments.
"""
import os
import sys
import logging
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'third_party'))
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)


# --------------------------------------------------------------------------- utils
def _min_k(y):
    return int(np.bincount(np.unique(y, return_inverse=True)[1]).min())


class _MinMaxWrap:
    """Resample in [0,1]-scaled space, map back to the caller's space."""
    def __init__(self, inner):
        self.inner = inner

    def fit_resample(self, X, y):
        sc = MinMaxScaler().fit(X)
        Xr, yr = self.inner(sc.transform(X), y)
        return sc.inverse_transform(np.asarray(Xr, dtype=float)), np.asarray(yr)


# --------------------------------------------------------------------------- MC-CCR
def _mc_ccr(seed, energy=0.25):
    from mc_ccr_algorithms import MultiClassCCR

    def run(X, y):
        np.random.seed(seed)
        classes, yi = np.unique(y, return_inverse=True)
        Xr, yr = MultiClassCCR(energy=energy).fit_sample(X, yi)
        return Xr, classes[np.asarray(yr).astype(int)]
    return _MinMaxWrap(run)


# --------------------------------------------------------------------------- MC-RBO
def _rbo_vectorised(X, y, minority_class, n, rng, gamma=0.05, step_size=0.001,
                    n_steps=500, n_nearest_neighbors=25):
    """Vectorised equivalent of RBO.fit_sample (authors' code, approximate_potential=True).

    For every synthetic point the authors run a greedy coordinate search: random
    (dimension, sign) moves of size step_size are tried in random order without
    replacement; a move is accepted when it lowers |mutual class potential|, after
    which the direction list is regenerated without the reverse move. The search
    stops after n_steps tries or when no direction is left. Here all synthetic
    points advance in lock-step; each keeps its own direction permutation.
    """
    min_idx = np.where(y == minority_class)[0]
    d = X.shape[1]
    if n <= 0 or len(min_idx) == 0:
        return np.empty((0, d))
    # neighbours of each minority seed within X (the seed itself is included, as in the original)
    nn = NearestNeighbors(n_neighbors=min(n_nearest_neighbors + 1, len(X)), metric='manhattan').fit(X)
    nbr = nn.kneighbors(X[min_idx], return_distance=False)
    sign_of = np.where(y[nbr] == minority_class, -1.0, 1.0)      # +1 majority, -1 minority

    seeds = rng.choice(len(min_idx), size=n)                     # uniform allocation
    P = X[min_idx[seeds]]
    NB = X[nbr[seeds]]                                           # (n, m, d)
    SG = sign_of[seeds]                                          # (n, m)

    def potential(points):
        dist = np.abs(points[:, None, :] - NB_act).sum(axis=2)
        return (SG_act * np.exp(-(dist / gamma) ** 2)).sum(axis=1)

    T = np.zeros_like(P)
    D2 = 2 * d
    perm = np.argsort(rng.random((n, D2)), axis=1)               # direction ids 0..2d-1
    ptr = np.zeros(n, dtype=int)
    avail = np.full(n, D2)
    NB_act, SG_act = NB, SG
    pot = np.abs(potential(P))
    for _ in range(n_steps):
        act = np.where(ptr < avail)[0]
        if len(act) == 0:
            break
        dir_id = perm[act, ptr[act]]
        ptr[act] += 1
        dim, sgn = dir_id // 2, np.where(dir_id % 2 == 0, -1.0, 1.0)
        cand = T[act].copy()
        cand[np.arange(len(act)), dim] += sgn * step_size
        NB_act, SG_act = NB[act], SG[act]
        new_pot = np.abs(potential(P[act] + cand))
        better = new_pot < pot[act]
        if better.any():
            ia = act[better]
            T[ia] = cand[better]
            pot[ia] = new_pot[better]
            # regenerate direction list without the reverse move
            rev = dim[better] * 2 + (sgn[better] < 0).astype(int)   # opposite sign id
            newp = np.argsort(rng.random((len(ia), D2)), axis=1)
            pos = np.argmax(newp == rev[:, None], axis=1)
            newp[np.arange(len(ia)), pos] = newp[:, -1]
            newp[:, -1] = rev
            perm[ia] = newp
            ptr[ia] = 0
            avail[ia] = D2 - 1
    return P + T


def _mc_rbo(seed):
    def run(X, y):
        rng = np.random.default_rng(seed)
        classes, yi = np.unique(y, return_inverse=True)
        sizes = np.bincount(yi)
        order = np.argsort(sizes)[::-1]
        obs = [X[yi == c] for c in order]
        n_max = len(obs[0])
        for i in range(1, len(order)):          # 'sampling' scheme of MultiClassRBO
            n = n_max - len(obs[i])
            Xs, ys = [obs[i]], [np.full(len(obs[i]), order[i])]
            for j in range(i):
                ids = rng.choice(len(obs[j]), int(n_max / i))
                Xs.append(obs[j][ids]); ys.append(np.full(len(ids), order[j]))
            app = _rbo_vectorised(np.vstack(Xs), np.concatenate(ys), order[i], n, rng)
            if len(app):
                obs[i] = np.vstack([obs[i], app])
        Xr = np.vstack(obs)
        yr = np.concatenate([np.full(len(o), c) for o, c in zip(obs, order)])
        return Xr, classes[yr]
    return _MinMaxWrap(run)


# --------------------------------------------------------------------------- SWIM
class SWIMMaha:
    """SWIM with Mahalanobis distance (Sharma et al. ICDM 2018; Bellinger et al. KAIS 2020),
    multiclass: every non-majority class is oversampled w.r.t. the density of the
    largest class. A small ridge keeps the majority covariance invertible."""
    def __init__(self, sd=0.25, ridge=1e-6, seed=0):
        self.sd, self.ridge, self.seed = sd, ridge, seed

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.seed)
        classes, cnt = np.unique(y, return_counts=True)
        maj = classes[np.argmax(cnt)]
        Xm = X[y == maj]
        mu = Xm.mean(axis=0)
        C = np.cov(Xm - mu, rowvar=False) + self.ridge * np.eye(X.shape[1])
        L, V = np.linalg.eigh(C)
        L = np.maximum(L, self.ridge)
        M = V @ np.diag(L ** -0.5) @ V.T              # whitening
        M_inv = V @ np.diag(L ** 0.5) @ V.T
        out_X, out_y = [X], [y]
        for c, n_c in zip(classes, cnt):
            need = cnt.max() - n_c
            if c == maj or need <= 0:
                continue
            W = (X[y == c] - mu) @ M.T                 # centred on the majority, whitened
            sd = W.std(axis=0)
            base = W[rng.integers(0, len(W), size=need)]
            new = base + rng.uniform(-self.sd, self.sd, size=base.shape) * sd
            nb = np.linalg.norm(base, axis=1, keepdims=True)
            nn_ = np.linalg.norm(new, axis=1, keepdims=True)
            new = new * np.where(nn_ > 0, nb / np.maximum(nn_, 1e-12), 1.0)  # keep Mahalanobis dist
            out_X.append(new @ M_inv.T + mu)
            out_y.append(np.full(need, c))
        return np.vstack(out_X), np.concatenate(out_y)


# --------------------------------------------------------------------------- KNNOR
class KNNORMulti:
    """KNNOR (Islam et al., Applied Soft Computing 115, 2022, 108288; `knnor` package,
    MIT licence) adapted to multiclass: each non-majority class c is run as
    c-vs-rest so KNNOR's validity check sees every other class, and a random
    n_max - n_c of the generated points are kept."""
    def __init__(self, seed=0):
        self.seed = seed

    def fit_resample(self, X, y):
        import random
        from knnor import data_augment
        rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)               # knnor draws its interpolation weights from `random`
        classes, cnt = np.unique(y, return_counts=True)
        out_X, out_y = [X], [y]
        for c, n_c in zip(classes, cnt):
            need = cnt.max() - n_c
            if need <= 0:
                continue
            yb = (y == c).astype(int)
            _, _, Xn, _ = data_augment.KNNOR().fit_resample(X, yb)
            Xn = np.asarray(Xn, dtype=float).reshape(-1, X.shape[1])
            if len(Xn) > need:
                Xn = Xn[rng.choice(len(Xn), need, replace=False)]
            if len(Xn):
                out_X.append(Xn); out_y.append(np.full(len(Xn), c))
        return np.vstack(out_X), np.concatenate(out_y)


# --------------------------------------------------------------------------- multi-imbalance
class _MI:
    def __init__(self, cls, rs, **kw):
        self.cls, self.rs, self.kw = cls, rs, kw

    def fit_resample(self, X, y):
        np.random.seed(self.rs)
        classes, yi = np.unique(y, return_inverse=True)
        Xr, yr = self.cls(**self.kw)._fit_resample(X.copy(), yi.copy())
        return np.asarray(Xr, dtype=float), classes[np.asarray(yr).astype(int)]


# --------------------------------------------------------------------------- smote-variants
class _SV:
    def __init__(self, name, seed, k):
        self.name, self.seed, self.k = name, seed, k

    def fit_resample(self, X, y):
        import smote_variants as sv
        params = {'random_state': self.seed}
        if self.name in ('ProWSyn', 'Safe_Level_SMOTE'):
            params['n_neighbors'] = self.k
        o = sv.MulticlassOversampling(oversampler=self.name, oversampler_params=params)
        classes, yi = np.unique(y, return_inverse=True)
        Xr, yr = o.sample(X, yi)
        return np.asarray(Xr, dtype=float), classes[np.asarray(yr).astype(int)]


# --------------------------------------------------------------------------- registry
BASELINES = ['NoResample', 'SMOTE', 'Borderline-SMOTE', 'ADASYN', 'SMOTE-ENN', 'SMOTE-Tomek',
             'KMeans-SMOTE', 'Safe-Level-SMOTE', 'MWMOTE', 'ProWSyn',
             'Static-SMOTE', 'SOUP', 'MDO', 'MC-CCR', 'MC-RBO', 'SWIM', 'KNNOR']


def make(name, seed, y=None, params=None):
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
    k = 5 if y is None else max(1, min(5, _min_k(y) - 1))
    if name == 'NoResample':
        return None
    if name == 'SMOTE':
        return SMOTE(k_neighbors=k, random_state=seed)
    if name == 'Borderline-SMOTE':
        return BorderlineSMOTE(k_neighbors=k, random_state=seed)
    if name == 'ADASYN':
        return ADASYN(n_neighbors=k, random_state=seed)
    if name == 'SMOTE-ENN':
        return SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
    if name == 'SMOTE-Tomek':
        return SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
    if name == 'KMeans-SMOTE':
        return KMeansSMOTE(k_neighbors=k, random_state=seed, cluster_balance_threshold=0.0)
    if name == 'Safe-Level-SMOTE':
        return _SV('Safe_Level_SMOTE', seed, k)
    if name == 'MWMOTE':
        return _SV('MWMOTE', seed, k)
    if name == 'ProWSyn':
        return _SV('ProWSyn', seed, k)
    if name == 'Static-SMOTE':
        from multi_imbalance.resampling.static_smote import StaticSMOTE
        return _MI(StaticSMOTE, seed)
    if name == 'SOUP':
        from multi_imbalance.resampling.soup import SOUP
        return _MI(SOUP, seed, k=7)
    if name == 'MDO':
        from multi_imbalance.resampling.mdo import MDO
        return _MI(MDO, seed, k=5, seed=seed)
    if name == 'MC-CCR':
        return _mc_ccr(seed)
    if name == 'MC-RBO':
        return _mc_rbo(seed)
    if name == 'SWIM':
        return SWIMMaha(seed=seed)
    if name == 'KNNOR':
        return KNNORMulti(seed=seed)
    if name == 'MIRT-v1':
        from mirtplus import MIRTPlusResampler
        return MIRTPlusResampler(seed=seed, k_neighbors=k)
    if name == 'MIRT-fast':                           # cheaper ESDA estimators, mirt_fast.py
        from mirt_fast import MIRTFast
        return MIRTFast(seed=seed, k_neighbors=k)
    if name == 'MIRT-v1-ablate':                     # component switches, mirtplus_ablation.py
        from mirtplus_ablation import MIRTPlusAblate
        return MIRTPlusAblate(seed=seed, k_neighbors=k, **(params or {}))
    raise ValueError(name)
