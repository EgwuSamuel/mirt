"""
Generate every table (LaTeX), results figure (EPS + PDF + 600-dpi PNG) and quoted number of the JIIS
manuscript directly from the experiment outputs. No number in the paper is typed by hand, and every
qualitative statement in the text is checked against the data (the script stops if one fails).

Inputs  : results_ext/ALL_RESULTS.csv (main benchmark), results_ablation/ALL_RESULTS.csv,
          results_fast/ALL_RESULTS.csv, results_ext/tables/T_*.csv
Outputs : tab_*.tex, Fig2-Fig6 (.eps/.pdf/.png), numbers.tex (LaTeX macros), numbers.json
          -> experiment/paper_assets/ when run from the repository's experiment folder.
  python make_assets.py
"""
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, 'results_ext')):          # repository: experiment/make_assets.py
    EXP, OUT = HERE, os.path.join(HERE, 'paper_assets')
else:                                                         # authors' manuscript folder
    EXP = os.path.join(HERE, '..', 'mirtplus-api', 'experiment')
    OUT = HERE
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, EXP)
import benchmark_datasets as bd                      # noqa: E402
from dataset_level_stats import analyse, holm          # noqa: E402

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
BLUE, GREY, INK, MUTED = '#2a78d6', '#a9a8a3', '#0b0b0b', '#52514e'

V2 = ['MIRT-v2', 'MIRT-v2-Lite', 'MIRT-v2[uniform-mu]', 'MIRT-v2[crisp-clean]', 'MIRT-v2[no-clean]']
RENAME = {'MIRT-v1': 'MIRT', 'NoResample': 'No resampling'}
df = pd.read_csv(os.path.join(EXP, 'results_ext', 'ALL_RESULTS.csv'))
df = df[~df.method.isin(V2)].copy()
df['method'] = df['method'].replace(RENAME)
METHODS = sorted(df.method.unique())
CLFS = ['KNN', 'CART', 'SVM', 'RF', 'LGBM']
CLF_LABEL = {'KNN': 'k-NN', 'CART': 'CART', 'SVM': 'SVM', 'RF': 'RF', 'LGBM': 'LightGBM'}
meta = bd.describe(sorted(df.dataset.unique()), bd.EVAL_DATASETS)
HARD = meta.loc[meta.hard, 'dataset'].tolist()
num = {}                                              # numbers quoted in the text


def per_ds(metric, clf=None, subset=None):
    d = df if clf is None else df[df.clf == clf]
    P = d.groupby(['dataset', 'method'])[metric].mean().unstack()
    return P.loc[subset] if subset is not None else P


def ranks(P):
    return P.rank(axis=1, ascending=False).mean()


def fmt(x, nd=3):
    return f'{x:.{nd}f}'


def tex_escape(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def write(name, body):
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write(body)


# ---------------------------------------------------------------- Table: datasets
m = meta.sort_values('IR').reset_index(drop=True)
rows = [f"{tex_escape(r.dataset)} & {r.openml_id} & {r.n} & {r.d} & {r.K} & {r.IR:.1f} & "
        f"{'$\\checkmark$' if r.hard else ''} \\\\" for r in m.itertuples()]
half = (len(rows) + 1) // 2
L, R = rows[:half], rows[half:] + [r'\\'] * (2 * half - len(rows))
body = '\n'.join(f"{a[:-3]} & {b[:-3] if b != chr(92)*2 else '& & & & & &'} \\\\" for a, b in zip(L, R))
write('tab_datasets.tex', r"""\begin{table}[t]
\caption{The 32 benchmark datasets (OpenML identifiers; characteristics computed from the
preprocessed data). $n$: examples, $d$: features after encoding, $K$: classes, IR: imbalance
ratio, H: hard regime (IR $\geq 8$ and $K \geq 5$)}\label{tab:datasets}
\centering\scriptsize\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}lrrrrrc lrrrrrc@{}}
\toprule
Dataset & ID & $n$ & $d$ & $K$ & IR & H & Dataset & ID & $n$ & $d$ & $K$ & IR & H\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")
num['n_datasets'] = len(meta); num['n_hard'] = len(HARD)
num['IR_range'] = [float(meta.IR.min()), float(meta.IR.max())]
num['K_range'] = [int(meta.K.min()), int(meta.K.max())]
num['n_range'] = [int(meta.n.min()), int(meta.n.max())]

# ---------------------------------------------------------------- overall statistics
G = per_ds('gmean')
r_all, pw_all = analyse(G, 'MIRT')
pw_all = pw_all.set_index('baseline')
Ghard = per_ds('gmean', subset=HARD)
r_hard, pw_hard = analyse(Ghard, 'MIRT')
pw_hard = pw_hard.set_index('baseline')
rank_tab = pd.DataFrame({c: ranks(per_ds('gmean', c)) for c in CLFS})
rank_tab['All'] = r_all
rank_tab['Hard'] = r_hard
order = rank_tab['All'].sort_values().index.tolist()
means = pd.DataFrame({k: per_ds(k).mean() for k in ['gmean', 'bacc', 'f1', 'mcc', 'auc']})

# ---------------------------------------------------------------- Table: main results
lines = []
best = {c: rank_tab[c].min() for c in rank_tab}
bestm = {c: means[c].max() for c in means}
for mth in order:
    cells = [tex_escape(mth) if mth != 'MIRT' else r'\textbf{MIRT}']
    for c in ['All'] + CLFS + ['Hard']:
        v = fmt(rank_tab.loc[mth, c], 2)
        cells.append(r'\textbf{' + v + '}' if np.isclose(rank_tab.loc[mth, c], best[c]) else v)
    for c in ['gmean', 'bacc', 'f1', 'mcc']:
        v = fmt(means.loc[mth, c])
        cells.append(r'\textbf{' + v + '}' if np.isclose(means.loc[mth, c], bestm[c]) else v)
    lines.append(' & '.join(cells) + r' \\')
write('tab_main.tex', r"""\begin{table}[t]
\caption{Average ranks on G-mean (lower is better; datasets are the unit) pooled over the five
classifiers (All), per classifier (LGBM: LightGBM), and on the 11 hard-regime datasets (Hard), followed by the mean
over datasets of G-mean, balanced accuracy (BAcc), macro-F1 and MCC. Methods are ordered by the
pooled rank; the best value in each column is in bold}\label{tab:main}
\centering\scriptsize\setlength{\tabcolsep}{1.2pt}
\begin{tabular}{@{}l rrrrrrr rrrr@{}}
\toprule
 & \multicolumn{7}{c}{Average rank (G-mean)} & \multicolumn{4}{c}{Mean over datasets}\\
\cmidrule(lr){2-8}\cmidrule(l){9-12}
Method & All & k-NN & CART & SVM & RF & LGBM & Hard & G-mean & BAcc & F1 & MCC\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: pairwise MIRT vs baselines
from scipy.stats import friedmanchisquare     # noqa: E402
lines = []
for b in [x for x in order if x != 'MIRT']:
    a, h = pw_all.loc[b], pw_hard.loc[b]
    sig = lambda p: r'$^{\ast}$' if p < 0.05 else ''
    lines.append(f"{tex_escape(b)} & {a['mean_diff']:+.3f} & {a['W/T/L']} & {a['holm_p']:.3f}{sig(a['holm_p'])} & "
                 f"{a['P(better)']:.2f} & {a['P(rope)']:.2f} & {a['P(worse)']:.2f} & "
                 f"{h['W/T/L']} & {h['holm_p']:.3f}{sig(h['holm_p'])} \\\\")
write('tab_pairwise.tex', r"""\begin{table}[t]
\caption{MIRT versus each baseline on G-mean (one row per baseline; datasets are the unit).
$\Delta$: mean difference MIRT $-$ baseline; W/T/L: datasets won, tied ($|\Delta| \leq 0.002$)
and lost by MIRT; $p_{\mathrm{Holm}}$: one-sided Wilcoxon signed-rank test with Holm correction
over the 17 comparisons ($^{\ast}$: $p<0.05$); Bayesian signed-rank posterior probabilities
that MIRT is better, practically equivalent (ROPE $\pm 0.01$) or worse}\label{tab:pairwise}
\centering\scriptsize\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}l rrr rrr rr@{}}
\toprule
 & \multicolumn{6}{c}{All 32 datasets} & \multicolumn{2}{c}{Hard regime (11)}\\
\cmidrule(lr){2-7}\cmidrule(l){8-9}
Baseline & $\Delta$ & W/T/L & $p_{\mathrm{Holm}}$ & P(better) & P(equiv.) & P(worse) & W/T/L & $p_{\mathrm{Holm}}$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
chi2, pF = friedmanchisquare(*[G[c] for c in G.columns])
num.update({
    'friedman_chi2': float(chi2), 'friedman_p': float(pF),
    'mirt_rank_all': float(r_all['MIRT']), 'mirt_position_all': int(order.index('MIRT') + 1),
    'best_all': order[0], 'best_rank_all': float(r_all[order[0]]),
    'mirt_rank_hard': float(r_hard['MIRT']),
    'mirt_position_hard': int(r_hard.sort_values().index.tolist().index('MIRT') + 1),
    'holm_sig_all': sorted(pw_all.index[pw_all.holm_p < 0.05].tolist()),
    'holm_sig_hard': sorted(pw_hard.index[pw_hard.holm_p < 0.05].tolist()),
    'rope_smote': float(pw_all.loc['SMOTE', 'P(rope)']),
    'mirt_means': means.loc['MIRT'].round(4).to_dict(),
    'means_by_method': means.round(4).to_dict(orient='index'),
    'ranks': rank_tab.round(2).to_dict(orient='index'),
    'per_clf_sig': {c: sorted(analyse(per_ds('gmean', c), 'MIRT')[1].query('holm_p < 0.05').baseline.tolist())
                    for c in CLFS},
})

# ---------------------------------------------------------------- other metrics: rank + two-sided Holm
from scipy.stats import wilcoxon               # noqa: E402
other = {}
for met in ['gmean', 'bacc', 'f1', 'mcc', 'auc']:
    P = per_ds(met)
    rr = ranks(P).sort_values()
    bl = [b for b in P.columns if b != 'MIRT']
    p_better = holm([wilcoxon(P['MIRT'], P[b], alternative='greater', zero_method='zsplit').pvalue for b in bl])
    p_worse = holm([wilcoxon(P[b], P['MIRT'], alternative='greater', zero_method='zsplit').pvalue for b in bl])
    other[met] = {'rank': float(rr['MIRT']), 'position': int(list(rr.index).index('MIRT') + 1),
                  'top3': [(k, round(float(v), 2)) for k, v in rr.head(3).items()],
                  'mirt_better_than': sorted(b for b, p in zip(bl, p_better) if p < 0.05),
                  'better_than_mirt': sorted(b for b, p in zip(bl, p_worse) if p < 0.05)}
num['other_metrics'] = other

# ---------------------------------------------------------------- Table: ablation
A = pd.read_csv(os.path.join(EXP, 'results_ablation', 'ALL_RESULTS.csv'))
lab = {'Full': 'MIRT (full)', '-FeatWeights': 'w/o RF feature weighting', '-Subgroups': 'w/o subgroup clustering',
       '-Mahalanobis': 'w/o Mahalanobis (Euclidean)', '-NoiseGate': 'w/o noise gate (full reach)',
       '-ConfSteer': 'w/o confusability steering'}
PA = A.groupby(['dataset', 'method'])['gmean'].mean().unstack()
ra, pa = analyse(PA, 'Full'); pa = pa.set_index('baseline')
PAh = PA.loc[HARD]; rah, pah = analyse(PAh, 'Full'); pah = pah.set_index('baseline')
chiA, pA = friedmanchisquare(*[PA[c] for c in PA.columns])
lines = []
for v in ['Full', '-FeatWeights', '-Subgroups', '-Mahalanobis', '-NoiseGate', '-ConfSteer']:
    if v == 'Full':
        lines.append(f"{lab[v]} & {PA[v].mean():.4f} & {ra[v]:.2f} & -- & -- & {PAh[v].mean():.4f} & {rah[v]:.2f} & -- \\\\")
    else:
        lines.append(f"{lab[v]} & {PA[v].mean():.4f} & {ra[v]:.2f} & {pa.loc[v, 'holm_p']:.3f} & {pa.loc[v, 'P(rope)']:.2f} & "
                     f"{PAh[v].mean():.4f} & {rah[v]:.2f} & {pah.loc[v, 'holm_p']:.3f} \\\\")
write('tab_ablation.tex', r"""\begin{table}[t]
\caption{Ablation of MIRT (G-mean, 5-fold cross-validation, five classifiers). Each variant
disables one component. $p_{\mathrm{Holm}}$: one-sided Wilcoxon test that the full method is better
(Holm-corrected); P(equiv.): Bayesian probability of practical equivalence (ROPE $\pm 0.01$)}\label{tab:ablation}
\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l rrrr rrr@{}}
\toprule
 & \multicolumn{4}{c}{All 32 datasets} & \multicolumn{3}{c}{Hard regime (11)}\\
\cmidrule(lr){2-5}\cmidrule(l){6-8}
Variant & G-mean & Rank & $p_{\mathrm{Holm}}$ & P(equiv.) & G-mean & Rank & $p_{\mathrm{Holm}}$\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
num['ablation_friedman'] = [float(chiA), float(pA)]

# ---------------------------------------------------------------- ESDA validity
E = pd.read_csv(os.path.join(EXP, 'results_ext', 'tables', 'T_esda_confusability.csv'))
num['esda'] = {'median_pearson': float(E.pearson_r.median()), 'mean_pearson': float(E.pearson_r.mean()),
               'median_spearman': float(E.spearman_r.median()), 'positive': int((E.pearson_r > 0).sum()),
               'ge03': int((E.pearson_r >= 0.3).sum()), 'n': int(E.pearson_r.notna().sum()),
               'negative': E.loc[E.pearson_r < 0, 'dataset'].tolist()}

# ---------------------------------------------------------------- cost
t = df.drop_duplicates(['dataset', 'rep', 'fold', 'method']).groupby('method')['resample_s']
cost = pd.DataFrame({'mean': t.mean(), 'median': t.median()})
cost['x_smote'] = cost['mean'] / cost.loc['SMOTE', 'mean']
fail = df.groupby('method')['failed'].mean() * 100
num['cost'] = cost.round(3).to_dict(orient='index'); num['fail_pct'] = fail.round(2).to_dict()

# ---------------------------------------------------------------- cost profile and cheaper ESDA
TAB = os.path.join(EXP, 'results_ext', 'tables')
CP = pd.read_csv(os.path.join(TAB, 'T_cost_profile.csv')).set_index('dataset')
STAGE = {'esda_rf_s': 'ESDA: random-forest feature weights', 'esda_kmeans_s': 'ESDA: subgroup search ($k$-means, silhouette)',
         'esda_other_s': 'ESDA: distances, aggregation, normalisation', 'typing_s': 'Difficulty typing ($k$-NN)',
         'synthesis_s': 'Synthesis'}
share = CP[list(STAGE)].div(CP.mirt_s, axis=0)
CP['speedup'] = CP.mirt_s / CP.fast_s
F = pd.read_csv(os.path.join(EXP, 'results_fast', 'ALL_RESULTS.csv'))
assert len(F) == 32 * 10 * 5 and F.failed.sum() == 0 and F.gmean.notna().all()
Gf = F.groupby('dataset').gmean.mean()
GF = pd.DataFrame({'MIRT': G['MIRT'], 'MIRT-fast': Gf.loc[G.index]})
import baycomp                             # noqa: E402  (same settings as dataset_level_stats.analyse)
from scipy.stats import wilcoxon as _wx   # noqa: E402
p_fast_two = float(_wx(GF['MIRT'], GF['MIRT-fast'], zero_method='zsplit').pvalue)
_dF = GF['MIRT-fast'] - GF['MIRT']
pw_fast = {'P(rope)': baycomp.two_on_multiple(GF['MIRT'].values, GF['MIRT-fast'].values, rope=0.01,
                                              nsamples=200000, random_state=0)[1],
           'W/T/L': f"{int((_dF > 0.002).sum())}/{int((_dF.abs() <= 0.002).sum())}/{int((_dF < -0.002).sum())}"}
Gsub = G.drop(columns='MIRT').assign(**{'MIRT-fast': Gf.loc[G.index]})
r_sub = ranks(Gsub).sort_values()
low_agree = CP.index[CP.mu_agreement_r < 0.8].tolist()
num['cost_profile'] = {'mirt_median_s': float(CP.mirt_s.median()), 'share_median': share.median().to_dict(),
                       'fast_speedup_median': float(CP.speedup.median()), 'agree_median': float(CP.mu_agreement_r.median()),
                       'fast_validity_median': float(CP.fast_validity_r.median()), 'low_agree': low_agree,
                       'fast_rank_substituted': float(r_sub['MIRT-fast']), 'fast_p_two_sided': p_fast_two,
                       'fast_rope': float(pw_fast['P(rope)'])}
Esd = E.set_index('dataset').pearson_r
lines = [f"{lab_} & {CP[c].median():.3f} & {100 * share[c].median():.0f} \\\\" for c, lab_ in STAGE.items()]
write('tab_cost.tex', r"""\begin{table}[t]
\caption{Where MIRT spends its time, and the cheaper ESDA setting. Top: time of each stage on the first
training fold of every dataset (median over """ + str(len(CP)) + r""" datasets of the median of five runs) and its
median share of MIRT's time. Bottom: MIRT with the default ESDA and with the cheaper ESDA estimators
(Sect.~\ref{sec:res-cost})}\label{tab:cost}
\centering\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
Stage of MIRT & Time (s) & Share (\%)\\
\midrule
""" + '\n'.join(lines) + r"""
\midrule
Total MIRT (SMOTE: """ + f"{CP.smote_s.median():.3f}" + r""" s) & """ + f"{CP.mirt_s.median():.3f}" + r""" & 100 \\
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}lrr@{}}
\toprule
 & Default ESDA & Cheaper ESDA\\
\midrule
Median time per training fold (s) & """ + f"{CP.mirt_s.median():.3f} & {CP.fast_s.median():.3f}" + r""" \\
Median speed-up & -- & """ + f"{CP.speedup.median():.1f}$\\times$ ({CP.speedup.min():.1f}--{CP.speedup.max():.1f})" + r""" \\
Median agreement of $\mu$ with the default (Pearson $r$) & -- & """ + f"{CP.mu_agreement_r.median():.2f}" + r""" \\
ESDA validity: median $r$ with CV confusion & """ + f"{Esd.median():.2f} & {CP.fast_validity_r.median():.2f}" + r""" \\
ESDA validity: datasets with $r>0$ & """ + f"{int((Esd > 0).sum())} & {int((CP.fast_validity_r > 0).sum())}" + r""" \\
Mean G-mean (32 datasets, 5 classifiers) & """ + f"{G['MIRT'].mean():.4f} & {Gf.mean():.4f}" + r""" \\
Average rank among the 18 methods & """ + f"{r_all['MIRT']:.2f} & {r_sub['MIRT-fast']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- worked example (satimage, ecoli)
WORKED = ['satimage', 'ecoli']
WE = {d: {k: pd.read_csv(os.path.join(TAB, f'T_worked_{d}_{k}.csv'), **({'index_col': 0} if k in ('esda', 'confusion') else {}))
          for k in ('esda', 'confusion', 'classes', 'features')} for d in WORKED}


def pairs(d):
    """Unordered class pairs with ESDA similarity, mean confusion in both directions and their ranks."""
    S, C = WE[d]['esda'], WE[d]['confusion']
    out = [{'a': a, 'b': b, 'sim': S.loc[a, b], 'conf': (C.loc[a, b] + C.loc[b, a]) / 2}
           for i, a in enumerate(S.index) for b in S.index[i + 1:]]
    out = pd.DataFrame(out)
    out['sim_rank'] = out.sim.rank(ascending=False, method='min').astype(int)
    out['conf_rank'] = out.conf.rank(ascending=False, method='min').astype(int)
    return out.sort_values('sim', ascending=False).reset_index(drop=True)


PW = {d: pairs(d) for d in WORKED}
lines = []
for d in WORKED:
    T = WE[d]['classes']
    lines.append(r'\multicolumn{9}{@{}l}{\emph{' + d + r'}}\\')
    for _, r in T.iterrows():
        lines.append(f"\\quad {tex_escape(r['class'])} & {r['n']} & {100 * r['share_S']:.0f} & {100 * r['share_B']:.0f} & "
                     f"{100 * r['share_R']:.0f} & {100 * r['share_O']:.0f} & {r['recall_NoResample']:.3f} & "
                     f"{r['recall_SMOTE']:.3f} & {r['recall_MIRT-v1']:.3f} \\\\")
write('tab_worked.tex', r"""\begin{table}[t]
\caption{Class profile produced by MIRT for the two worked examples: share (\%) of safe (S), borderline
(B), rare (R) and outlier (O) examples in ESDA's feature-weighted space, and the recall of each class
without resampling, with SMOTE and with MIRT (mean over the 5$\times$2 folds and five classifiers of
the benchmark)}\label{tab:worked}
\centering\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
 & & \multicolumn{4}{c}{Example types (\%)} & \multicolumn{3}{c}{Recall}\\
\cmidrule(lr){3-6}\cmidrule(l){7-9}
Class & $n$ & S & B & R & O & None & SMOTE & MIRT\\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
num['worked'] = {d: PW[d].head(4).to_dict(orient='records') for d in WORKED}


# ================================================================= figures
def save(fig, name):
    for ext, kw in [('eps', {}), ('pdf', {}), ('png', {'dpi': 600})]:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', facecolor='white', **kw)
    plt.close(fig)


# Fig 2: critical-difference diagram (Nemenyi, pooled)
from scipy.stats import studentized_range   # noqa: E402
k, N = G.shape[1], G.shape[0]
cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6.0 * N))
num['nemenyi_cd'] = float(cd)
rs = r_all.sort_values()
fig, ax = plt.subplots(figsize=(6.85, 2.9))
lo, hi = 1, int(np.ceil(rs.max()))
ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0, 1); ax.axis('off')
ax.plot([lo, hi], [0.86, 0.86], color=INK, lw=0.8)
for t_ in range(lo, hi + 1):
    ax.plot([t_, t_], [0.86, 0.885], color=INK, lw=0.8)
    ax.text(t_, 0.91, str(t_), ha='center', va='bottom', fontsize=7)
ax.plot([lo, lo + cd], [0.98, 0.98], color=INK, lw=1.2)
ax.text(lo + cd / 2, 1.0, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=7)
names = rs.index.tolist(); half = (len(names) + 1) // 2
for i, nm in enumerate(names):
    left = i < half
    y = 0.78 - (i if left else len(names) - 1 - i) * (0.72 / half)
    x_end = lo - 0.2 if left else hi + 0.2
    col = BLUE if nm == 'MIRT' else INK
    ax.plot([rs[nm], rs[nm], x_end], [0.86, y, y], color=col, lw=1.4 if nm == 'MIRT' else 0.7)
    ax.text(x_end + (-0.05 if left else 0.05), y, f'{nm} ({rs[nm]:.2f})', ha='right' if left else 'left',
            va='center', fontsize=7, color=col, fontweight='bold' if nm == 'MIRT' else 'normal')
# cliques: maximal groups whose rank span < CD
cliques, vals = [], rs.values
for i in range(len(vals)):
    j = max(jj for jj in range(i, len(vals)) if vals[jj] - vals[i] < cd)
    if j > i and not any(a <= i and j <= b for a, b in cliques):
        cliques.append((i, j))
for c_i, (a, b) in enumerate(cliques):
    yy = 0.83 - 0.025 * c_i
    ax.plot([vals[a] - 0.04, vals[b] + 0.04], [yy, yy], color=MUTED, lw=2.0, solid_capstyle='round')
save(fig, 'Fig2')

# Fig 3: average rank per classifier (dot plot, MIRT highlighted)
fig, ax = plt.subplots(figsize=(6.85, 3.6))
cols = ['All'] + CLFS
for j, c in enumerate(cols):
    for mth in order:
        v = rank_tab.loc[mth, c]
        is_m = mth == 'MIRT'
        ax.scatter(j, v, s=36 if is_m else 14, color=BLUE if is_m else GREY, zorder=3 if is_m else 2,
                   edgecolor='white', linewidth=0.6)
    ax.annotate(f"{rank_tab.loc['MIRT', c]:.2f}", (j, rank_tab.loc['MIRT', c]), xytext=(7, 0),
                textcoords='offset points', va='center', fontsize=7, color=INK)
ax.set_xticks(range(len(cols)), ['Pooled'] + [CLF_LABEL[c] for c in CLFS])
ax.invert_yaxis(); ax.set_ylabel('Average rank on G-mean (1 = best)')
ax.grid(axis='y', color='#e4e3df', lw=0.5); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.scatter([], [], s=36, color=BLUE, label='MIRT'); ax.scatter([], [], s=14, color=GREY, label='17 baselines')
ax.legend(frameon=False, loc='lower left', fontsize=7)
save(fig, 'Fig3')

# Fig 4: ESDA similarity vs confusability, per dataset
Es = E.sort_values('pearson_r')
fig, ax = plt.subplots(figsize=(6.85, 2.6))
ax.bar(range(len(Es)), Es.pearson_r, color=[GREY if v < 0 else BLUE for v in Es.pearson_r], width=0.72)
ax.axhline(0, color=INK, lw=0.6)
ax.axhline(Es.pearson_r.median(), color=MUTED, lw=0.8, ls='--')
ax.text(0, Es.pearson_r.median() + 0.04, f'median r = {Es.pearson_r.median():.2f}',
        ha='left', va='bottom', fontsize=7, color=MUTED)
ax.set_xticks(range(len(Es)), Es.dataset, rotation=70, ha='right', fontsize=6.5)
ax.set_ylabel('Pearson r (ESDA vs. CV confusion)'); ax.set_ylim(-1, 1)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
save(fig, 'Fig4')

# Fig 5: worked example - ESDA similarity next to the measured confusion
fig, axes = plt.subplots(2, 2, figsize=(6.85, 6.3), gridspec_kw={'hspace': 0.75, 'wspace': 0.55})
for row, d in enumerate(WORKED):
    for col, (key, title, cmap) in enumerate([('esda', 'ESDA similarity $\\mu_{ij}$', 'Blues'),
                                              ('confusion', 'CV confusion $\\gamma_{ij}$ (row: true class)', 'Greys')]):
        M = WE[d][key].copy()
        if d == 'ecoli':                              # long class names -> the dataset's own class codes
            M.index = M.columns = WE[d]['classes']['label'].tolist()
        V = M.values.astype(float).copy(); np.fill_diagonal(V, np.nan)
        ax = axes[row, col]
        ax.imshow(V, cmap=cmap, vmin=0, vmax=np.nanmax(V))
        for i in range(len(M)):
            for j in range(len(M)):
                if i != j:
                    ax.text(j, i, f'{V[i, j]:.2f}', ha='center', va='center', fontsize=5.8,
                            color='white' if V[i, j] > 0.6 * np.nanmax(V) else INK)
        ax.set_xticks(range(len(M)), M.columns, rotation=40, ha='right', fontsize=6.3)
        ax.set_yticks(range(len(M)), M.index, fontsize=6.3)
        ax.set_title(f'{d}: {title}', fontsize=7.5, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
save(fig, 'Fig5')

# Fig 6: cost vs. accuracy (one y-axis: rank; x: resampling time, log)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for mth in order:
    is_m = mth == 'MIRT'
    if mth == 'No resampling':
        continue
    x, y = cost.loc[mth, 'mean'], r_all[mth]
    ax.scatter(x, y, s=40 if is_m else 16, color=BLUE if is_m else GREY, zorder=3, edgecolor='white', lw=0.6)
    ax.annotate(mth, (x, y), xytext=(4, 2), textcoords='offset points', fontsize=6.2,
                color=INK if is_m else MUTED, fontweight='bold' if is_m else 'normal')
ax.set_xscale('log'); ax.invert_yaxis()
ax.set_xlabel('Mean resampling time per fold (s, log scale)'); ax.set_ylabel('Average rank on G-mean (1 = best)')
ax.grid(color='#e4e3df', lw=0.5); ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
save(fig, 'Fig6')

# ================================================================= numbers.tex (macros quoted in the text)
WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
         10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
         17: 'seventeen'}
ORD = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
       9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth'}
RECENT = ['MDO', 'SOUP', 'MC-CCR', 'MC-RBO', 'SWIM', 'KNNOR', 'Static-SMOTE']


def nice(names):
    names = [('no resampling' if n == 'No resampling' else n) for n in names]
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def order_by(names, key):
    return sorted(names, key=key)


mac = {}
r_hard_sorted = r_hard.sort_values()
sig_all = pw_all.index[pw_all.holm_p < 0.05].tolist()
sig_hard = pw_hard.index[pw_hard.holm_p < 0.05].tolist()
equiv = pw_all.index[pw_all['P(rope)'] >= 0.95].tolist()
mac.update({
    'NDatasets': len(meta), 'NHard': len(HARD), 'NDown': int((meta.n >= 3990).sum()),
    'IRmin': f'{meta.IR.min():.1f}', 'IRmax': f'{meta.IR.max():.1f}',
    'Kmin': int(meta.K.min()), 'Kmax': int(meta.K.max()),
    'FriedmanChi': f'{chi2:.1f}', 'FriedmanExp': int(np.floor(np.log10(pF))) + 1, 'CD': f'{cd:.2f}',
    'MirtRank': f"{r_all['MIRT']:.2f}", 'MirtPos': ORD[order.index('MIRT') + 1],
    'FirstName': order[0], 'FirstRank': f'{r_all[order[0]]:.2f}',
    'SecondName': order[1], 'SecondRank': f'{r_all[order[1]]:.2f}',
    'ThirdName': order[2], 'ThirdRank': f'{r_all[order[2]]:.2f}',
    'NSigAll': WORDS[len(sig_all)], 'SigAllList': nice(order_by(sig_all, lambda b: -pw_all.loc[b, 'mean_diff'])[::-1]),
    'NNotSigAll': WORDS[17 - len(sig_all)],
    'SigRecentList': nice([b for b in RECENT if b in sig_all]) if any(b in sig_all for b in RECENT) else 'none',
    'EquivList': nice(order_by(equiv, lambda b: r_all[b])), 'EquivMin': f"{pw_all.loc[equiv, 'P(rope)'].min():.2f}" if equiv else '--',
    'MccrEquiv': f"{pw_all.loc['MC-CCR', 'P(rope)']:.2f}",
    'MdoWins': pw_all.loc['MDO', 'W/T/L'].split('/')[0], 'McrboWins': pw_all.loc['MC-RBO', 'W/T/L'].split('/')[0],
    'SwimWins': pw_all.loc['SWIM', 'W/T/L'].split('/')[0],
    'MirtBacc': f"{means.loc['MIRT', 'bacc']:.3f}", 'MirtFone': f"{means.loc['MIRT', 'f1']:.3f}",
    'MirtMcc': f"{means.loc['MIRT', 'mcc']:.3f}", 'SmoteBacc': f"{means.loc['SMOTE', 'bacc']:.3f}",
    'SmoteFone': f"{means.loc['SMOTE', 'f1']:.3f}", 'SmoteMcc': f"{means.loc['SMOTE', 'mcc']:.3f}",
    'SoupG': f"{means.loc['SOUP', 'gmean']:.3f}", 'SoupFone': f"{means.loc['SOUP', 'f1']:.3f}",
    'SoupMcc': f"{means.loc['SOUP', 'mcc']:.3f}", 'BestMeanG': means['gmean'].idxmax(),
    'BestMeanGval': f"{means['gmean'].max():.3f}",
    'HardMirtRank': f"{r_hard['MIRT']:.2f}", 'HardMirtPos': ORD[list(r_hard_sorted.index).index('MIRT') + 1],
    'HardAhead': nice([f'{m} ({r_hard[m]:.2f})' for m in r_hard_sorted.index[:list(r_hard_sorted.index).index('MIRT')]]),
    'SigHardList': nice(order_by(sig_hard, lambda b: b.lower())),
    'MccrHardWTL': pw_hard.loc['MC-CCR', 'W/T/L'], 'MccrHardP': f"{pw_hard.loc['MC-CCR', 'holm_p']:.3f}",
    'EsdaPos': num['esda']['positive'], 'EsdaMed': f"{num['esda']['median_pearson']:.2f}",
    'EsdaMean': f"{num['esda']['mean_pearson']:.2f}", 'EsdaSpear': f"{num['esda']['median_spearman']:.2f}",
    'EsdaGe': num['esda']['ge03'],
    'EsdaNeg': nice([f"{d} ($r={E.set_index('dataset').loc[d, 'pearson_r']:.2f}$)"
                     for d in E.sort_values('pearson_r').query('pearson_r < 0').dataset]),
    'AblChi': f'{chiA:.2f}', 'AblP': f'{pA:.2f}',
    'AblHardEquivMin': f"{pah['P(rope)'].min():.2f}",
    'CostMirt': f"{cost.loc['MIRT', 'mean']:.2f}", 'CostSmote': f"{cost.loc['SMOTE', 'mean']:.3f}",
    'CostRatio': f"{cost.loc['MIRT', 'x_smote']:.0f}", 'CostMcrbo': f"{cost.loc['MC-RBO', 'mean']:.1f}",
    'CostMwmote': f"{cost.loc['MWMOTE', 'mean']:.1f}",
    'FailAdasyn': f"{fail['ADASYN']:.0f}", 'FailKmeans': f"{fail['KMeans-SMOTE']:.0f}",
})
for c, nm in [('KNN', 'Knn'), ('SVM', 'Svm'), ('CART', 'Cart'), ('RF', 'Rf'), ('LGBM', 'Lgbm')]:
    Pc = per_ds('gmean', c)
    rc = ranks(Pc).sort_values()
    mac[f'NSig{nm}'] = WORDS.get(len(num['per_clf_sig'][c]), str(len(num['per_clf_sig'][c])))
    mac[f'Mirt{nm}Rank'] = f"{rc['MIRT']:.2f}"
    mac[f'Mirt{nm}Pos'] = ORD[list(rc.index).index('MIRT') + 1]
    mac[f'Best{nm}'] = rc.index[0]
    mac[f'Best{nm}Rank'] = f'{rc.iloc[0]:.2f}'
    mac[f'Fried{nm}Exp'] = int(np.floor(np.log10(friedmanchisquare(*[Pc[m] for m in Pc.columns])[1]))) + 1
    k_sig = len(num['per_clf_sig'][c])
    mac[f'SigText{nm}'] = (f'is significantly better than {WORDS[k_sig]} baseline' + ('s' if k_sig > 1 else '')
                           if k_sig else 'is not significantly better than any baseline')
for met, nm in [('bacc', 'Bacc'), ('f1', 'Fone'), ('mcc', 'Mcc'), ('auc', 'Auc')]:
    o = num['other_metrics'][met]
    mac[f'Pos{nm}'] = ORD[o['position']]
    mac[f'Above{nm}'] = nice(o['better_than_mirt']) if o['better_than_mirt'] else 'none'
ab, af = num['other_metrics']['bacc']['better_than_mirt'], num['other_metrics']['f1']['better_than_mirt']
mac['AboveBaccFoneText'] = ('no baseline is significantly better than MIRT on either metric' if not (ab or af) else
                            f"significantly better than MIRT are {nice(ab) if ab else 'none'} on balanced accuracy "
                            f"and {nice(af) if af else 'none'} on macro-F1")
mac['AblEquivMin'] = f"{min(pa['P(rope)'].min(), pah['P(rope)'].min()):.2f}"

# ---- worked example
def wclass(d, name):
    return WE[d]['classes'].set_index('class').loc[name]


def recall_shift(d):
    """Recall change of every class from no resampling to MIRT."""
    T = WE[d]['classes'].set_index('class')
    return (T['recall_MIRT-v1'] - T['recall_NoResample']).sort_values()


def esda_neighbours(d, name, k):
    S = WE[d]['esda'].loc[name].drop(name)
    return S.sort_values(ascending=False).index[:k].tolist()


def top_pairs(d, key, k):
    P = PW[d].sort_values(key, ascending=False).head(k)
    return {frozenset((a, b)) for a, b in zip(P.a, P.b)}


Esd_all = E.set_index('dataset')
sat_d, eco_i = wclass('satimage', 'damp grey soil'), wclass('ecoli', 'inner membrane, uncleavable signal')
sat_vd, sat_g = wclass('satimage', 'very damp grey soil'), wclass('satimage', 'grey soil')
eco_im = wclass('ecoli', 'inner membrane')
sat_other_safe = WE['satimage']['classes'].set_index('class').drop('damp grey soil')['share_S']
mac.update({
    'SatR': f"{Esd_all.loc['satimage', 'pearson_r']:.2f}", 'EcoR': f"{Esd_all.loc['ecoli', 'pearson_r']:.2f}",
    'SatPairConf': f"{PW['satimage'].conf.iloc[0]:.2f}",
    'SatSafeDamp': f"{100 * sat_d.share_S:.0f}", 'SatSafeOtherMin': f"{100 * sat_other_safe.min():.0f}",
    'SatDampNone': f"{sat_d.recall_NoResample:.3f}", 'SatDampSmote': f"{sat_d.recall_SMOTE:.3f}",
    'SatDampMirt': f"{sat_d['recall_MIRT-v1']:.3f}",
    'SatVdNone': f"{sat_vd.recall_NoResample:.3f}", 'SatVdMirt': f"{sat_vd['recall_MIRT-v1']:.3f}",
    'SatGreyNone': f"{sat_g.recall_NoResample:.3f}", 'SatGreyMirt': f"{sat_g['recall_MIRT-v1']:.3f}",
    'EcoPairConf': f"{PW['ecoli'].conf.iloc[0]:.2f}",
    'EcoSafeImu': f"{100 * eco_i.share_S:.0f}", 'EcoBorderImu': f"{100 * eco_i.share_B:.0f}",
    'EcoImuNone': f"{eco_i.recall_NoResample:.3f}", 'EcoImuSmote': f"{eco_i.recall_SMOTE:.3f}",
    'EcoImuMirt': f"{eco_i['recall_MIRT-v1']:.3f}",
    'EcoImNone': f"{eco_im.recall_NoResample:.3f}", 'EcoImMirt': f"{eco_im['recall_MIRT-v1']:.3f}",
    'EcoTopFeat': nice(WE['ecoli']['features'].feature.head(3).tolist()),
    'EcoZeroFeat': nice(WE['ecoli']['features'].query('weight < 0.01').feature.tolist()),
})

# ---- cost profile and cheaper ESDA
cpn = num['cost_profile']
mac.update({
    'ProfMirt': f"{CP.mirt_s.median():.2f}", 'ProfMirtMin': f"{CP.mirt_s.min():.2f}",
    'ProfMirtMax': f"{CP.mirt_s.max():.1f}", 'ProfSmote': f"{CP.smote_s.median():.3f}",
    'ProfRatio': f"{(CP.mirt_s / CP.smote_s).median():.0f}",
    'ShareRf': f"{100 * share['esda_rf_s'].median():.0f}", 'ShareKm': f"{100 * share['esda_kmeans_s'].median():.0f}",
    'ShareSyn': f"{100 * share['synthesis_s'].median():.0f}", 'ShareTyp': f"{100 * share['typing_s'].median():.0f}",
    'ShareEsda': f"{100 * (share['esda_rf_s'] + share['esda_kmeans_s'] + share['esda_other_s']).median():.0f}",
    'FastTrees': 20, 'FastSil': 1000,
    'FastSpeed': f"{CP.speedup.median():.1f}", 'FastSpeedMin': f"{CP.speedup.min():.1f}",
    'FastSpeedMax': f"{CP.speedup.max():.1f}", 'FastTime': f"{CP.fast_s.median():.2f}",
    'FastAgree': f"{CP.mu_agreement_r.median():.2f}",
    'FastLowAgree': nice([f"{d} ($r={CP.loc[d, 'mu_agreement_r']:.2f}$)"
                          for d in CP.sort_values('mu_agreement_r').index if d in low_agree]),
    'NFastLowAgree': WORDS[len(low_agree)],
    'FastValid': f"{CP.fast_validity_r.median():.2f}", 'FastValidPos': int((CP.fast_validity_r > 0).sum()),
    'FastG': f"{Gf.mean():.4f}", 'MirtG': f"{G['MIRT'].mean():.4f}",
    'FastRank': f"{r_sub['MIRT-fast']:.2f}", 'FastPos': ORD[list(r_sub.index).index('MIRT-fast') + 1],
    'FastP': f"{p_fast_two:.2f}", 'FastRope': f"{pw_fast['P(rope)']:.2f}", 'FastWTL': pw_fast['W/T/L'],
})
mac['BestTreeText'] = (f"{mac['BestRf']} ranks first with both" if mac['BestRf'] == mac['BestLgbm'] else
                       f"best: {mac['BestRf']} and {mac['BestLgbm']}, respectively")

# ---- every qualitative statement in the text, checked against the data (build fails if one is false)
claims = {
    'top methods within Nemenyi CD of the best': r_all['MIRT'] - r_all.min() < cd,
    'no baseline significantly better than MIRT on G-mean': not num['other_metrics']['gmean']['better_than_mirt'],
    'SOUP has the highest mean G-mean': means['gmean'].idxmax() == 'SOUP',
    'SOUP macro-F1 and MCC below SMOTE': (means.loc['SOUP', 'f1'] < means.loc['SMOTE', 'f1'])
                                         and (means.loc['SOUP', 'mcc'] < means.loc['SMOTE', 'mcc']),
    'no resampling ranks last on G-mean': order[-1] == 'No resampling',
    'no resampling has the best MCC': means['mcc'].idxmax() == 'No resampling',
    'MIRT mid-table on MCC and AUC': all(6 <= num['other_metrics'][m]['position'] <= 13 for m in ['mcc', 'auc']),
    'MIRT not first in the hard regime': r_hard.idxmin() != 'MIRT',
    'ablation: no variant significant': (pa['holm_p'] >= 0.05).all() and (pah['holm_p'] >= 0.05).all(),
    'MC-RBO and MWMOTE slower than MIRT': cost.loc['MC-RBO', 'mean'] > cost.loc['MIRT', 'mean'] < cost.loc['MWMOTE', 'mean'],
    'MIRT, SMOTE and multiclass methods never failed': all(fail[m] == 0 for m in
                                                          ['MIRT', 'SMOTE', 'SOUP', 'MDO', 'MC-CCR', 'MC-RBO', 'SWIM', 'KNNOR', 'Static-SMOTE']),
    'k-NN and SVM: large Friedman differences': mac['FriedKnnExp'] <= -10 and mac['FriedSvmExp'] <= -10,
    'splice is among the negative ESDA correlations': 'splice' in num['esda']['negative'],
    # worked example
    'satimage: top-3 ESDA pairs are the top-3 confused pairs': top_pairs('satimage', 'sim', 3) == top_pairs('satimage', 'conf', 3),
    'ecoli: top-4 ESDA pairs are the top-4 confused pairs': top_pairs('ecoli', 'sim', 4) == top_pairs('ecoli', 'conf', 4),
    'satimage: top-3 ESDA pairs all involve grey-soil classes':
        all(any('grey' in c for c in p) for p in top_pairs('satimage', 'sim', 3)),
    'satimage: the most similar pair has the highest confusion': PW['satimage'].conf.iloc[0] == PW['satimage'].conf.max(),
    'satimage: most other pairs confused below 3%': (PW['satimage'].conf.iloc[3:] < 0.03).mean() > 0.5,
    'ecoli: the most similar pair has the highest confusion': PW['ecoli'].conf.iloc[0] == PW['ecoli'].conf.max(),
    'cheaper ESDA: low agreement only where ESDA validity is below its median':
        all(Esd_all.loc[d, 'pearson_r'] < Esd_all.pearson_r.median() for d in low_agree),
    'satimage: damp grey soil has the fewest safe examples': WE['satimage']['classes'].set_index('class').share_S.idxmin() == 'damp grey soil',
    'satimage: damp grey soil gains most recall with MIRT': recall_shift('satimage').index[-1] == 'damp grey soil',
    'satimage: the two largest losses are its two ESDA neighbours':
        set(recall_shift('satimage').index[:2]) == set(esda_neighbours('satimage', 'damp grey soil', 2)),
    'satimage: SMOTE also raises damp grey soil recall': sat_d.recall_SMOTE > sat_d.recall_NoResample,
    'ecoli: imU has the fewest safe and most borderline examples':
        WE['ecoli']['classes'].set_index('class').share_S.idxmin() == 'inner membrane, uncleavable signal'
        and WE['ecoli']['classes'].set_index('class').share_B.idxmax() == 'inner membrane, uncleavable signal',
    'ecoli: imU gains most recall with MIRT': recall_shift('ecoli').index[-1] == 'inner membrane, uncleavable signal',
    'ecoli: MIRT recall on imU above SMOTE': eco_i['recall_MIRT-v1'] > eco_i.recall_SMOTE,
    'ecoli: largest loss is on im, the ESDA neighbour of imU':
        recall_shift('ecoli').index[0] == esda_neighbours('ecoli', 'inner membrane, uncleavable signal', 1)[0] == 'inner membrane',
    # cost
    'RF and k-means are the two most expensive stages': set(share.median().sort_values().index[-2:]) == {'esda_rf_s', 'esda_kmeans_s'},
    'typing and synthesis are cheap': share['typing_s'].median() < 0.1 and share['synthesis_s'].median() < 0.2,
    'cheaper ESDA always faster': (CP.speedup > 1).all(),
    'cheaper ESDA validity practically unchanged': abs(CP.fast_validity_r.median() - Esd.median()) < 0.02
        and abs(int((CP.fast_validity_r > 0).sum()) - int((Esd > 0).sum())) <= 1,
    'cheaper ESDA practically equivalent in accuracy': pw_fast['P(rope)'] >= 0.95,
    'MIRT takes about a second per training set': 0.5 <= CP.mirt_s.median() <= 1.5,
    'cheaper ESDA not significantly different (two-sided)': p_fast_two >= 0.05,
}
bad = [k for k, v in claims.items() if not v]
assert not bad, f'claims in the text no longer hold: {bad}'
num['claims_checked'] = list(claims)

with open(os.path.join(OUT, 'numbers.tex'), 'w', encoding='utf-8') as f:
    f.write('% generated by make_assets.py from the result files - do not edit\n')
    for k, v in mac.items():
        f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
num['macros'] = mac

json.dump(num, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1, default=float)
print('tables: tab_datasets, tab_main, tab_pairwise, tab_ablation, tab_worked, tab_cost | figures: Fig2-Fig6 | numbers.tex')
print(json.dumps({k: num[k] for k in ['mirt_rank_all', 'mirt_position_all', 'best_all', 'holm_sig_all',
                                      'holm_sig_hard', 'rope_smote', 'mirt_rank_hard', 'mirt_position_hard',
                                      'ablation_friedman', 'esda', 'nemenyi_cd', 'per_clf_sig']}, indent=1, default=float))
