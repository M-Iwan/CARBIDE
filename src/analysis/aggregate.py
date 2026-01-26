"""
Module with functions for aggregating training visualise
"""

from typing import Tuple

import pandas as pd
import optuna

from src.data.utils import round_to_significant


def scores_to_df(scores: dict):
    """
    Aggregate scores from a single trained model

    Parameters
    ----------
    scores: Dict[str, Dict] (an overly complicated nested dictionary)

    Returns
    -------
    pd.DataFrame
        A DataFrame with scores in a readable format
    """

    def series_to_df(sub_scores: dict, series: str):

        def subseries_to_df(sub_sub_scores: dict, sign: Tuple):
            subseries_df = pd.DataFrame({key: [value] for key, value in sub_sub_scores.items()})
            subseries_df['Kind'] = ' | '.join(sign)
            return subseries_df

        sub_dfs = [
            subseries_to_df(sub_scores['Total'], ('Total', 'Total'))
        ]

        for key, ss_scores in sub_scores['Sign'].items():
            sub_dfs.append(subseries_to_df(ss_scores, key))

        sub_df = pd.concat(sub_dfs).reset_index(drop=True)
        sub_df['Series'] = series
        return sub_df

    weighted_df = series_to_df(scores['Weighted'], 'Weighted')
    unweighted_df = series_to_df(scores['Unweighted'], 'Unweighted')

    return pd.concat([weighted_df, unweighted_df])


def study_to_df(study: optuna.study.Study):
    """
    Aggregate information from Optuna study
    """

    trial_dfs = []

    for trial in study.trials:
        df = pd.DataFrame({
            'Params': [' | '.join([f'{key}: {round_to_significant(value, 7)}' for key, value in trial.params.items()])],
            'Value': [trial.value],
            'Number': [trial.number],

        })
        trial_dfs.append(df)

    study_df = pd.concat(trial_dfs).reset_index(drop=True)

    return study_df
