"""
Ablation variants of MIRT++. Production mirtplus.py is left untouched.
Each flag disables one component so its contribution can be isolated:
  feat_weights  : RF feature weighting        (off -> uniform weights)
  subgroups     : per-class subgroup clustering(off -> one subgroup per class)
  mahalanobis   : distribution-aware distance  (off -> Euclidean)
  noise_gate    : Eq.(18) conservativeness     (off -> full interpolation reach)
  conf_steer    : most-confusable steering     (off -> gate on noise only)
"""
import math, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _feature_weights(X, y, seed=42):
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = rf.feature_importances_
    return imp / imp.sum() if imp.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]


def _class_subgroups(Xc, m_max=3, seed=42, min_per_cluster=10):
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
        centroids.append(sub.mean(axis=0)); sizes.append(len(sub))
        covs.append(np.cov(sub, rowvar=False) if len(sub) > 1 else np.eye(Xc.shape[1]))
    return centroids, sizes, covs


def compute_similarity_ablate(X, y, m_max=3, lam=0.85, eps=1e-6, seed=42,
                              feat_weights=True, subgroups=True, mahalanobis=True):
    classes = sorted(np.unique(y).tolist())
    d = X.shape[1]
    Xs = StandardScaler().fit_transform(X)
    w = _feature_weights(Xs, y, seed) if feat_weights else np.ones(d) / d
    Xw = Xs * np.sqrt(w)
    eff_m = m_max if subgroups else 1
    sub = {c: _class_subgroups(Xw[y == c], m_max=eff_m, seed=seed) for c in classes}

    raw, all_d = {}, []
    for ci in classes:
        cen_i, siz_i, cov_i = sub[ci]
        for cj in classes:
            cen_j, siz_j, cov_j = sub[cj]
            for p in range(len(cen_i)):
                for q in range(len(cen_j)):
                    diff = cen_i[p] - cen_j[q]
                    if mahalanobis:
                        Sig = (np.atleast_2d(cov_i[p]) + np.atleast_2d(cov_j[q]))/2.0 + eps*np.eye(d)
                        try:
                            dist = float(np.sqrt(max(diff @ np.linalg.solve(Sig, diff), 0)))
                        except np.linalg.LinAlgError:
                            dist = float(np.linalg.norm(diff))
                    else:
                        dist = float(np.linalg.norm(diff))
                    raw[(ci, cj, p, q)] = (dist, siz_i[p]*siz_j[q])
                    if ci != cj:
                        all_d.append(dist)
    med = np.median(all_d) if all_d else 1.0
    alpha = 1.0/med if med > 0 else 1.0

    mu_raw = {ci: {} for ci in classes}
    for ci in classes:
        for cj in classes:
            num = den = 0.0
            for p in range(len(sub[ci][0])):
                for q in range(len(sub[cj][0])):
                    dist, wgt = raw[(ci, cj, p, q)]
                    s = math.exp(-alpha*dist)
                    num += wgt*s; den += wgt
            mu_raw[ci][cj] = num/den if den > 0 else 0.0
    offdiag = [mu_raw[ci][cj] for ci in classes for cj in classes if ci != cj]
    baseline = np.mean(offdiag) if offdiag else 0.0
    mu_reg = {ci: {cj: lam*mu_raw[ci][cj] + (1-lam)*baseline for cj in classes} for ci in classes}
    od = [mu_reg[ci][cj] for ci in classes for cj in classes if ci != cj]
    lo, hi = (min(od), max(od)) if od else (0.0, 1.0)
    rng = (hi-lo) if hi > lo else 1.0
    mu = {ci: {cj: (1.0 if ci == cj else float((mu_reg[ci][cj]-lo)/rng)) for cj in classes} for ci in classes}
    return mu, classes, w


def _difficulty_types(Xw, y, k=5):
    nn = NearestNeighbors(n_neighbors=k+1).fit(Xw)
    _, idx = nn.kneighbors(Xw)
    types = np.empty(len(y), dtype='<U1')
    for i in range(len(y)):
        same = int(np.sum(y[idx[i, 1:k+1]] == y[i]))
        types[i] = 'S' if same >= 4 else 'B' if same >= 2 else 'R' if same == 1 else 'O'
    return types


class MIRTPlusAblate:
    """MIRT++ with component switches. Defaults reproduce the full method."""
    def __init__(self, k_neighbors=5, m_max=3, beta=0.15, lam=0.85, seed=42,
                 rare_ratio=6.0, noise_floor=0.6,
                 feat_weights=True, subgroups=True, mahalanobis=True,
                 noise_gate=True, conf_steer=True):
        self.k = k_neighbors; self.m_max = m_max; self.beta = beta; self.lam = lam
        self.seed = seed; self.rare_ratio = rare_ratio; self.noise_floor = noise_floor
        self.feat_weights = feat_weights; self.subgroups = subgroups
        self.mahalanobis = mahalanobis; self.noise_gate = noise_gate; self.conf_steer = conf_steer

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        classes = sorted(np.unique(y).tolist())
        sizes = {c: int(np.sum(y == c)) for c in classes}
        mu, _, weights = compute_similarity_ablate(
            X, y, m_max=self.m_max, lam=self.lam, seed=self.seed,
            feat_weights=self.feat_weights, subgroups=self.subgroups,
            mahalanobis=self.mahalanobis)
        Xw = StandardScaler().fit_transform(X) * np.sqrt(weights)
        types = _difficulty_types(Xw, y, k=self.k)
        maj = max(sizes, key=sizes.get)
        target = sizes[maj]
        out_X, out_y = [], []
        for c in classes:
            mask = (y == c); Xc, Xwc, tc = X[mask], Xw[mask], types[mask]
            n = len(Xc)
            out_X.append(Xc); out_y.append(np.full(n, c))
            if c == maj:
                continue
            need = target - n
            if need <= 0:
                continue
            is_rare = (target/max(n, 1)) >= self.rare_ratio
            kk = min(self.k, n-1)
            if kk < 1:
                reps = Xc[rng.integers(0, n, size=need)]
                out_X.append(reps); out_y.append(np.full(need, c)); continue
            _, idx_in = NearestNeighbors(n_neighbors=kk+1).fit(Xwc).kneighbors(Xwc)
            noise_level = np.mean(np.isin(tc, ['R', 'O']))
            s_conf = max([mu[c][j] for j in classes if j != c], default=0.0)
            if not self.conf_steer:
                s_conf = 1.0                        # gate on noise only
            if is_rare or not self.noise_gate:
                step_cap = 1.0                      # full reach
            else:
                step_cap = max(self.noise_floor, 1.0 - self.beta*s_conf*noise_level)
            syn = np.empty((need, X.shape[1]))
            for t in range(need):
                si = rng.integers(0, n)
                nb = rng.choice(idx_in[si, 1:kk+1])
                cap = step_cap*(0.85 if (tc[si] == 'B' and noise_level > 0.3 and not is_rare) else 1.0)
                syn[t] = Xc[si] + rng.uniform(0, cap)*(Xc[nb]-Xc[si])
            out_X.append(syn); out_y.append(np.full(need, c))
        X_res = np.vstack(out_X); y_res = np.concatenate(out_y)
        perm = rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]
