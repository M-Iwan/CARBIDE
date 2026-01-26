from typing import List, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import wilcoxon


def matched_pairs_rank_biserial(x: np.ndarray, y: np.ndarray):
    """
    Effect size for Wilcoxon Signed-Rank test.

    Parameters
    ----------
    x: np.ndarray
    y: np.ndarray

    Returns
    -------
    r: float
        Value between -1 and 1 describing effect's strength
    """

    differences = x - y
    differences = differences[differences != 0]

    ranks = stats.rankdata(np.abs(differences))

    r_plus = np.sum(ranks[differences > 0])
    r_minus = np.sum(ranks[differences < 0])

    r = (r_plus - r_minus) / (r_plus + r_minus)

    if (r_abs := np.abs(r)) < 0.1:
        desc = 'Negligible'
    elif r_abs < 0.3:
        desc = 'Small'
    elif r_abs < 0.5:
        desc = 'Medium'
    else:
        desc = 'Large'

    return r, desc


def compare_pairs(df: pl.DataFrame, group_col: str,  pairs: List[Tuple], index_cols: List[str],
                  sign_level: float = 0.01):
    """
    Perform pairwise Wilcoxon signed-rank tests with matched-pairs rank-biserial correlation.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format DataFrame containing the values to compare.
    group_col : str
        Column name containing the grouping variable whose levels will be
        compared (e.g., 'DPA' with levels 'PRR', 'ROR', 'IC').
    pairs : list of tuple
        Pairs of group levels to compare.
        Example: [('PRR', 'ROR'), ('PRR', 'IC'), ('ROR', 'IC')]
    index_cols : list of str
        Column names that uniquely identify matched observations across groups.
        Example: ['Model', 'Descriptors', 'Fold', 'Dataset', 'PT']
    sign_level : float, optional
        Significance level for hypothesis testing (default: 0.01).

    Returns
    -------
    visualise : pl.DataFrame
        Outcome of the analyses
    """

    df = (df.with_columns(
        pl.Series('Idx', ["_".join(row) for row in df[index_cols].iter_rows()])
        )
        .drop(index_cols)
    )

    df = df.pivot(on=group_col, index='Idx')

    results = []
    for pair in pairs:
        null_h = f"{pair[0]} == {pair[1]}"

        x = df[pair[0]].to_numpy()
        y = df[pair[1]].to_numpy()

        w_out = wilcoxon(
            x=x,
            y=y,
            zero_method='wilcox',
            correction=True,  # practically irrelevant with our sample sizes (810)
            alternative='two-sided'
        )

        w_stat, p_value = w_out.statistic, w_out.pvalue

        r_stat, effect_size = matched_pairs_rank_biserial(x, y)

        if discernible := p_value < sign_level:
            if r_stat > 0:
                outcome = f"{pair[0]} > {pair[1]}"
            elif r_stat < 0:
                outcome = f"{pair[0]} < {pair[1]}"
            else:
                outcome = f"{pair[0]} == {pair[1]}"
        else:
            outcome = f"{pair[0]} == {pair[1]}"

        res_df = pl.DataFrame({
            'null_hypothesis': null_h,
            'null_rejected': discernible,
            'outcome': outcome,
            'effect_size': effect_size,
            'w_stat': w_stat,
            'p_value': p_value,
            'r_stat': r_stat,
        })

        results.append(res_df)

    results = pl.concat(results)
    return results

