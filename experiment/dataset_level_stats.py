"""
Dataset-level statistical comparison (Demšar 2006; Garcia & Herrera 2008;
Benavoli et al. 2017).

The original analysis ran Friedman/Wilcoxon over (dataset, fold, classifier)
rows, which treats folds of the same dataset as independent samples and
inflates significance. Here the unit of analysis is the dataset: each method
is summarised by its mean score per dataset (N = number of datasets).

Reports, for one per-dataset score table:
  * average ranks and Friedman / Iman-Davenport test
  * Nemenyi critical difference
  * one-sided Wilcoxon signed-rank of the proposed method vs each baseline,
    with Holm correction for the family of comparisons
  * per-dataset win/tie/loss (|diff| <= tie_tol counts as a tie)
  * Bayesian signed-rank test (baycomp) with a region of practical
    equivalence (ROPE), giving P(proposed better), P(equivalent), P(worse)

Usage:
  python dataset_level_stats.py table_T3_per_dataset_gmean.csv
  python dataset_level_stats.py table_T3_per_dataset_gmean.csv --subset cleveland,ecoli,flare,glass,winequality-red,winequality-white,yeast
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nemenyi_cd(k, n, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def analyse(df, proposed, rope=0.01, tie_tol=0.002, alpha=0.05):
    n, k = df.shape
    ranks = df.rank(axis=1, ascending=False).mean().sort_values()
    chi2, p_f = friedmanchisquare(*[df[c].values for c in df.columns])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    print(f'N datasets = {n}, k methods = {k}')
    print(f'Friedman chi2 = {chi2:.2f}, p = {p_f:.2e}; Iman-Davenport F = {ff:.2f}')
    print(f'Nemenyi CD (alpha={alpha}) = {nemenyi_cd(k, n, alpha):.3f}\n')
    print('Average ranks (lower is better):')
    print(ranks.round(2).to_string(), '\n')

    try:
        import baycomp
    except ImportError:
        baycomp = None

    rows = []
    for c in df.columns:
        if c == proposed:
            continue
        diff = df[proposed] - df[c]
        w = int((diff > tie_tol).sum())
        l = int((diff < -tie_tol).sum())
        t = n - w - l
        try:
            p = wilcoxon(df[proposed], df[c], alternative='greater',
                         zero_method='zsplit').pvalue
        except ValueError:
            p = 1.0
        row = {'baseline': c, 'mean_diff': diff.mean(), 'W/T/L': f'{w}/{t}/{l}',
               'wilcoxon_p': p}
        if baycomp is not None:
            # fixed seed and more Monte-Carlo samples: the posterior is estimated by sampling
            p_better, p_rope, p_worse = baycomp.two_on_multiple(
                df[proposed].values, df[c].values, rope=rope, nsamples=200000, random_state=0)
            row.update({'P(better)': p_better, 'P(rope)': p_rope, 'P(worse)': p_worse})
        rows.append(row)
    out = pd.DataFrame(rows)
    out['holm_p'] = holm(out['wilcoxon_p'])
    out['sig(Holm)'] = np.where(out['holm_p'] < alpha, 'yes', 'no')
    print(f'{proposed} vs baselines (one-sided Wilcoxon, Holm-adjusted; '
          f'Bayesian signed-rank with ROPE = {rope}):')
    print(out.round(4).to_string(index=False))
    return ranks, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('table', help='CSV: rows = datasets, columns = methods')
    ap.add_argument('--proposed', default=None, help='column of the proposed method')
    ap.add_argument('--subset', default=None, help='comma-separated dataset names')
    ap.add_argument('--rope', type=float, default=0.01)
    ap.add_argument('--out', default=None, help='optional CSV for the pairwise table')
    a = ap.parse_args()

    df = pd.read_csv(a.table, index_col=0)
    df = df[[c for c in df.columns if not str(c).startswith('Unnamed')]].dropna()
    if a.subset:
        keep = [s.strip() for s in a.subset.split(',')]
        df = df.loc[df.index.intersection(keep)]
    proposed = a.proposed or next(c for c in df.columns if 'MIRT' in c)
    _, out = analyse(df, proposed, rope=a.rope)
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
