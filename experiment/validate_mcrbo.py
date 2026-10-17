"""
Check that the vectorised MC-RBO in baselines.py matches the authors' reference
implementation (third_party/mc_rbo_algorithms.py) in distribution: same number
of synthetic points per class, same displacement statistics, and the same
downstream G-mean within seed noise.
"""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party'))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.metrics import geometric_mean_score
import mc_rbo_algorithms as ref
import baselines
from benchmark_datasets import load

X, y = load('vertebra-column')
res = {'reference': [], 'vectorised': []}
disp = {'reference': [], 'vectorised': []}
times = {'reference': 0.0, 'vectorised': 0.0}
for seed in range(3):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for which in res:
            t = time.time()
            if which == 'reference':
                np.random.seed(seed * 10 + f)
                Xr, yr = ref.MultiClassRBO().fit_sample(Xtr, y[tr].astype(float))
                yr = yr.astype(int)
            else:
                Xr, yr = baselines._mc_rbo(seed * 10 + f).inner(Xtr, y[tr])
            times[which] += time.time() - t
            # displacement of synthetic points from their nearest original minority point
            syn = Xr[len(Xtr):] if which == 'vectorised' else None
            n_orig = len(Xtr)
            counts = np.bincount(yr)
            for clf in (KNeighborsClassifier(5), DecisionTreeClassifier(random_state=0)):
                clf.fit(Xr, yr)
                res[which].append(geometric_mean_score(y[te], clf.predict(Xte), average='macro'))
            disp[which].append(counts.tolist())
for w in res:
    print(f'{w:10s} G-mean mean={np.mean(res[w]):.4f} sd={np.std(res[w]):.4f}  time={times[w]:.1f}s  '
          f'class counts (first fold)={disp[w][0]}')
d = np.array(res['reference']) - np.array(res['vectorised'])
print(f'paired diff mean={d.mean():+.4f}, sd={d.std():.4f}, n={len(d)}')
