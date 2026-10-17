"""
Turn results_ext/ALL_RESULTS.csv into the revision's tables.

All inferential statistics use the DATASET as the unit (scores are averaged over
the 10 CV folds first), via dataset_level_stats.analyse:
Friedman / Iman-Davenport, Nemenyi CD, Holm-corrected one-sided Wilcoxon,
and the Bayesian signed-rank test (ROPE = 0.01).

  python analyse_extended.py --res results_ext --proposed MIRT-v1
Outputs go to <res>/tables/.
"""
import argparse
import os
import numpy as np
import pandas as pd

import benchmark_datasets as bd
from dataset_level_stats import analyse


def per_dataset(df, metric, clf=None):
    d = df if clf is None else df[df['clf'] == clf]
    return d.groupby(['dataset', 'method'])[metric].mean().unstack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results_ext')
    ap.add_argument('--proposed', default='MIRT-v1')
    ap.add_argument('--metric', default='gmean')
    ap.add_argument('--exclude', default='', help='methods to leave out of the comparison')
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.res, 'ALL_RESULTS.csv'))
    if a.exclude:
        df = df[~df['method'].isin(a.exclude.split(','))]
    out = os.path.join(a.res, 'tables')
    os.makedirs(out, exist_ok=True)

    meta = bd.describe(sorted(df['dataset'].unique()), bd.EVAL_DATASETS)
    meta.to_csv(os.path.join(out, 'T1_datasets.csv'), index=False)
    hard = meta.loc[meta['hard'], 'dataset'].tolist()
    print(f'{df.dataset.nunique()} datasets ({len(hard)} hard), {df.method.nunique()} methods, '
          f'{df.clf.nunique()} classifiers, {len(df)} rows\n')

    # mean performance (dataset-averaged, so every dataset weighs the same)
    summ = {}
    for m in ['gmean', 'bacc', 'f1', 'mcc', 'auc', 'gmean_ovr']:
        summ[m] = per_dataset(df, m).mean()
    summ = pd.DataFrame(summ).sort_values(a.metric, ascending=False)
    summ.round(4).to_csv(os.path.join(out, 'T2_mean_performance.csv'))
    print('Mean over datasets (all classifiers pooled):')
    print(summ.round(4).to_string(), '\n')

    # pooled + per-classifier + hard regime statistics
    blocks = [('ALL', None, None)] + [(c, c, None) for c in sorted(df.clf.unique())] + [('HARD', None, hard)]
    rank_tab = {}
    for label, clf, subset in blocks:
        P = per_dataset(df, a.metric, clf)
        if subset is not None:
            P = P.loc[P.index.intersection(subset)]
        P = P.dropna()
        print(f'===== {label} ({a.metric}) =====')
        ranks, pw = analyse(P, a.proposed)
        rank_tab[label] = ranks
        pw.round(4).to_csv(os.path.join(out, f'T_pairwise_{a.metric}_{label}.csv'), index=False)
        P.round(4).to_csv(os.path.join(out, f'T_perdataset_{a.metric}_{label}.csv'))
        print()
    pd.DataFrame(rank_tab).round(2).to_csv(os.path.join(out, f'T_ranks_{a.metric}.csv'))

    # failures and cost
    fails = df.groupby('method')['failed'].mean().mul(100).round(2).rename('failed_%')
    t = df.drop_duplicates(['dataset', 'rep', 'fold', 'method']).groupby('method')['resample_s']
    cost = pd.DataFrame({'mean_s': t.mean(), 'median_s': t.median()})
    cost['x_SMOTE'] = cost['mean_s'] / cost.loc['SMOTE', 'mean_s'] if 'SMOTE' in cost.index else np.nan
    cost = cost.join(fails).sort_values('mean_s')
    cost.round(4).to_csv(os.path.join(out, 'T_cost_failures.csv'))
    print('Resampling cost and failure rate:')
    print(cost.round(3).to_string())


if __name__ == '__main__':
    main()
