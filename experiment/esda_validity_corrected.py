"""
ESDA validity (paper Section 4.6) on the 32 corrected EVAL datasets.

Identical procedure to compute_confusability.py (original Table 6): the MIRT
similarity matrix from mirtplus.compute_enhanced_similarity is correlated with the
5-fold cross-validated CART confusion matrix (mirtplus.similarity_vs_confusability).
Only the dataset registry differs (benchmark_datasets.py, correct OpenML ids).
Output: results_ext/tables/T_esda_confusability.csv
"""
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from mirtplus import compute_enhanced_similarity, similarity_vs_confusability
import benchmark_datasets as bd

rows = []
for name in bd.EVAL_DATASETS:
    X, y = bd.load(name, bd.EVAL_DATASETS)
    mu, classes, _ = compute_enhanced_similarity(X, y)
    r_p, r_s = similarity_vs_confusability(X, y, mu, classes)
    rows.append({'dataset': name, 'K': len(classes), 'pearson_r': r_p, 'spearman_r': r_s})
    print(f'  {name:<20s} K={len(classes):<2d} pearson={r_p:+.3f} spearman={r_s:+.3f}', flush=True)

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_ext', 'tables')
os.makedirs(out, exist_ok=True)
df.round(4).to_csv(os.path.join(out, 'T_esda_confusability.csv'), index=False)
v = df[df['pearson_r'].notna()]
print(f'\n{len(v)} datasets (all K >= 3): Pearson mean {v.pearson_r.mean():+.3f} '
      f'median {v.pearson_r.median():+.3f} | Spearman mean {v.spearman_r.mean():+.3f} '
      f'median {v.spearman_r.median():+.3f} | positive {int((v.pearson_r > 0).sum())}/{len(v)} '
      f'| r >= 0.3: {int((v.pearson_r >= 0.3).sum())}/{len(v)}')
