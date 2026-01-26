"""
Module with functions for Disproportionality Analysis
"""
from typing import Union, List, Optional, Dict

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from pandas.api.types import CategoricalDtype

from src.data.utils import is_valid_float


def binarize_reactions(df: pd.DataFrame, pt_idx: List[int], strat_col: Optional[Union[str, List[str]]] = None, smiles_col: str = 'smi_enc',
                       reaction_col: str = 'reac_enc', count_col: str = 'count'):
    """
    Combine reactions based on passed pt_idx to obtain a binary classification.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame with smi_enc : reac_enc counts
    pt_idx: List[int]
        List of reaction encoding indices to merge
    strat_col: Union[None, str, List[str]]
        List of columns to stratify on, e.g. ['sex', 'age', 'weight'].
        The smiles_col and reaction_col will be appended to the passed list.
        Default is None.
    smiles_col: str, optional
        Name of the column with drug/SMILES description. Default is 'smi_enc'
    reaction_col: str, optional
        Name of the column with event description. Default is 'reac_enc'
    count_col: str, optional
        Name of the column with pair counts. Default is 'count'

    Returns
    -------
    df: pd.DataFrame
        Pandas DataFrame with binarized reaction column
    """

    df = df.reset_index(drop=True)
    df.loc[:, reaction_col] = df[reaction_col].apply(lambda reaction: int(reaction in pt_idx))

    if strat_col is None:
        strat = [smiles_col, reaction_col]
    elif isinstance(strat_col, str):
        strat = [strat_col, smiles_col, reaction_col]
    elif isinstance(strat_col, list):
        strat = strat_col + [smiles_col, reaction_col]
    else:
        raise ValueError(f'Expected strat_col to be one of < None, int, list >. Received < {type(strat_col)} > instead')

    df = df.groupby(strat)[count_col].sum().reset_index(name=count_col)
    return df


def dpa_matrix(df: pd.DataFrame, smi_idx: int, pt_idx: Union[int, List[int]],
               smiles_col: str = 'StSMILES_enc', reaction_col: str = 'reactions_enc'):
    """
    Calculate the contingency matrix for a given SMILES index : PT index pair(s).

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to be used. Must include columns corresponding to drug and event in 'exploded' form.
    smi_idx: int
        SMILES index to use.
    pt_idx: Union[int, List[int]]
        PT index or list of indices to use.
    smiles_col: str, optional
        Name of the column with drug/SMILES description.
    reaction_col: str, optional
        Name of the column with event description.

    Returns
    -------
    cont_matrix: dict
        Dictionary with the visualise.
    """

    smiles_mask = df[smiles_col] == smi_idx

    reaction_mask = df[reaction_col].isin(pt_idx) if isinstance(pt_idx, list) else (df[reaction_col] == pt_idx)

    a = np.sum(smiles_mask & reaction_mask)  # pairs of drug AND event
    b = np.sum(smiles_mask & ~reaction_mask)  # pairs of drug AND NOT event
    c = np.sum(~smiles_mask & reaction_mask)  # pairs of NOT drug AND event
    d = np.sum(~smiles_mask & ~reaction_mask)  # pairs of NOT drug AND NOT event

    cont_matrix = {smiles_col: [smi_idx], reaction_col: [pt_idx], "a": a, "b": b, "c": c, "d": d}

    return cont_matrix


"""
Step 6
Prepare the drug-reaction counts stratified by Demo Factors.
"""

def step_6_1(df: pd.DataFrame):
    """
    In step 6.1 the following operations are performed:
    - data stratification by various combinations of demographic factors
    - counting of individual entries (i.e. instead of having 100 rows with the same data,
    there will be a single row with its count)

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe from step 5.

    Returns
    -------
    df: pd.DataFrame
    """

    strats = [(['Sex'], 's'), (['Age'], 'a'), (['Weight'], 'w'), (['Sex', 'Age'], 'sa'), (['Sex', 'Weight'], 'sw'),
              (['Age', 'Weight'], 'aw'), (['Sex', 'Age', 'Weight'], 'saw')]

    df = df.explode('reac_enc').explode('smi_enc').reset_index(drop=True)
    df = df.astype({'reac_enc': 'UInt16', 'smi_enc': 'UInt16'})

    dfs = [df.groupby(['smi_enc', 'reac_enc']).size().reset_index(name='count').assign(strat='n')]

    for strat, name in strats:
        sub_df = df.groupby(strat)[df.columns].apply(lambda group: group.groupby(['smi_enc', 'reac_enc']).size()).reset_index(name='count').assign(strat=name)
        dfs.append(sub_df)

    df = pd.concat(dfs).reset_index(drop=True)

    return df


def step_6_2(dfs: pd.DataFrame):
    """
    In step 6.2 the previously-split dataframes are combined

    Parameters
    ----------
    dfs: pd.DataFrame
        A dataframe from step 6.1

    Returns
    -------
    pd.DataFrame
    """

    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    pr_dfs = [dfs[dfs['strat'] == 'n'].groupby(['smi_enc', 'reac_enc'])['count'].sum().reset_index(name='count').assign(strat='n')]

    for name, strat in strats.items():
        sub_df = dfs[dfs['strat'] == name]
        gb = sub_df.groupby(strat)[sub_df.columns].apply(lambda group: group.groupby(['smi_enc', 'reac_enc'])['count'].sum().reset_index(name='count')).reset_index()
        gb = gb.drop(columns=[f'level_{len(name)}']).assign(strat=name)

        pr_dfs.append(gb)

    return pd.concat(pr_dfs, ignore_index=True)


"""
Step 7
"""


def step_7(df: pd.DataFrame, idx_dc: Dict[str, List[int]], strat_type: str):
    """
    In step 7, cardiotoxicity labels (1/0) are assigned to reactions.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe from step 6.2
    idx_dc: Dict[str, List[int]]
        A mapping from cardiotoxicity set (Cred, Card, Cvas) to sets of integers representing cardiotoxic reactions
    strat_type: str
        Stratification type.

    Returns
    -------
    df: pd.DataFrame
    """

    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    dfs = []

    for key, values in idx_dc.items():
        sub_df = df.copy()
        sub_df.loc[:, key] = df['reac_enc'].apply(lambda entry: int(entry in values)).astype('UInt8')

        if strat_type != 'n':
            sub_df = sub_df.groupby(strats.get(strat_type))[sub_df.columns].apply(lambda group: group.groupby(['smi_enc', key])['count'].sum().reset_index(name=f'{key}_ct'))
            sub_df = sub_df.reset_index().drop(columns=f'level_{len(strat_type)}')
        else:
            sub_df = sub_df.groupby(['smi_enc', key])['count'].sum().reset_index(name=f'{key}_ct')
        dfs.append((sub_df, key, strat_type))

    return dfs


"""
Step 8
"""


def dpa_matrix_count(df: pd.DataFrame, smi_idx: int, pt_idx: Union[int, List[int]], smiles_col: str = 'smi_enc',
                     reaction_col: str = 'reac_enc', count_col: str = 'count'):
    """
    Calculate the contingency matrix for a given SMILES index : PT index pair(s) using pre-counted pairs.
    The paris can be obtained using count_pairs function.

    Parameters
    ----------
    df: pd.DataFrame
        Dataset to be used. Must include columns corresponding to drug and event in 'exploded' form.
    smi_idx: int
        SMILES index to use.
    pt_idx: Union[int, List[int]]
        PT index or list of indices to use.
    smiles_col: str, optional
        Name of the column with drug/SMILES description. Default is 'smi_enc'
    reaction_col: str, optional
        Name of the column with event description. Default is 'reac_enc'
    count_col: str, optional
        Name of the column with pair counts. Default is 'count'

    Returns
    -------
    cont_matrix: dict
        Dictionary holding the visualise.
    """

    if isinstance(pt_idx, int):
        reaction_mask = df[reaction_col] == pt_idx
    else:
        reaction_mask = df[reaction_col].isin(pt_idx)

    a = df[(df[smiles_col] == smi_idx) & reaction_mask][count_col].sum()  # pairs of drug AND reaction
    b = df[(df[smiles_col] == smi_idx) & ~reaction_mask][count_col].sum()  # pairs of drug AND NOT reaction
    c = df[(df[smiles_col] != smi_idx) & reaction_mask][count_col].sum()  # pairs of NOT drug and reaction
    d = df[(df[smiles_col] != smi_idx) & ~reaction_mask][count_col].sum()  # pairs of NOT drug and NOT reaction

    return {smiles_col: smi_idx, reaction_col: pt_idx, 'a': a, 'b': b, 'c': c, 'd': d}


def step_8(df: pd.DataFrame, smiles_col: str, reaction_col: str, count_col: str, pt_idx: int, strat_type: str):
    """
    In step 8 DPA contingency matrices are calculated for each data stratification level

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 7.
    smiles_col: str
        Name of a column holding encoded SMILES
    reaction_col: str
        Name of a column holding encoded reactions.
    count_col: str
        Name of a column holding pair counts.
    pt_idx: int
        Whether to calculate cardiotoxicity (1) or non-cardiotoxicity (0) signals
    strat_type: str
        Type of stratification

    Returns
    -------
    df: pd.DataFrame
    """


    strats = {'s': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    if strat_type == 'n':
        entries = [dpa_matrix_count(df, smi_idx, pt_idx, smiles_col, reaction_col, count_col) for smi_idx in df[smiles_col].unique()]
        ct_df = pd.DataFrame(entries)

    else:
        dfs = []
        strat = strats.get(strat_type)
        for group in df.groupby(strat):
            group_df = pd.DataFrame(group[1])
            entries = [dpa_matrix_count(group_df, smi_idx, pt_idx, smiles_col, reaction_col, count_col) for smi_idx in group_df[smiles_col].unique()]
            df_ = pd.DataFrame(entries).assign(**{strat[i]: group[0][i] for i in range(len(strat))})
            dfs.append(df_)
        ct_df = pd.concat(dfs, ignore_index=True)

    ct_df = ct_df.astype({reaction_col: 'UInt8', 'a': 'UInt32', 'b': 'UInt32', 'c': 'UInt32', 'd': 'UInt32'})
    return ct_df


"""
Step 9
"""


def proportional_reporting_rate(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray],
                                d: Union[int, np.ndarray], sign_level: float = 0.01, decimals: int = 5):
    """
    Calculate the Proportional Reporting Ratio (PRR) from a contingency table. Based on  M. Fusaroli 'pvda' R package.
    Divisions by zero errors are caught automatically and return np.nan.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.05
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    prr_value: Union[float, np.ndarray]
        Calculated PRR.
    prr_lower: Union[float, np.ndarray]
        Lower bound of the PRR Confidence Interval.
    prr_upper: Union[float, np.ndarray]
        Upper bound of the PRR Confidence Interval.
    """

    exp_count = ((a + b) * c) / (c + d)
    prr_value = np.round(a / exp_count, decimals)

    s = np.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c + d)))
    z_value = norm.ppf(1 - (sign_level / 2))

    prr_lower = np.round(prr_value * np.exp(-z_value * s), decimals)
    prr_upper = np.round(prr_value * np.exp(z_value * s), decimals)

    return prr_value, prr_lower, prr_upper


def reporting_odds_ratio(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray],
                         d: Union[int, np.ndarray], sign_level: float = 0.01, decimals: int = 5):
    """
    Calculate the Reporting Odds Ratio (ROR) from a contingency table. Based on  M. Fusaroli 'pvda' R package.
    Divisions by zero errors are caught automatically and return np.nan.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.001
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    ror_value: Union[float, np.ndarray]
        Calculated ROR.
    ror_lower: Union[float, np.ndarray]
        Lower bound of the ROR Confidence Interval.
    ror_upper: Union[float, np.ndarray]
        Upper bound of the ROR Confidence Interval.
    """

    ror_value = np.round((a*d) / (b*c), decimals)

    s = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z_value = norm.ppf(1 - (sign_level / 2))

    ror_lower = np.round(ror_value * np.exp(-z_value * s), decimals)
    ror_upper = np.round(ror_value * np.exp(z_value * s), decimals)

    return ror_value, ror_lower, ror_upper


def information_component(a: Union[int, np.ndarray], b: Union[int, np.ndarray], c: Union[int, np.ndarray], d: Union[int, np.ndarray],
                          sign_level: float = 0.01, shrink: float = 0.5, decimals: int = 5):
    """
    Calculate the Information Component (IC) from a contingency table. Based on  M. Fusaroli 'pvda' R package.

    Parameters
    ----------
    a: Union[int, np.ndarray]
        Number of drug AND event pairs
    b: Union[int, np.ndarray]
        Number of drug AND NOT event pairs
    c: Union[int, np.ndarray]
        Number of NOT drug AND event pairs
    d: Union[int, np.ndarray]
        Number of NOT drug AND NOT event pairs
    sign_level: float, optional
        Significance level when calculating the CI. Default is 0.05
    shrink: float, optional
        Shrinkage factor. Default is 0.5.
    decimals: int, optional
        Number of decimals to keep. Default is 5

    Returns
    -------
    ic_value: Union[float, np.ndarray]
        Calculated Information Content.
    ic_lower: Union[float, np.ndarray]
        Lower bound of the IC Confidence Interval.
    ic_upper: Union[float, np.ndarray]
        Upper bound of the IC Confidence Interval.
    """

    obs_count = a
    exp_count = (a + b) * (a + c) / (a + b + c + d)
    # n_drugs * n_events / n_total
    alpha = obs_count + shrink
    beta = exp_count + shrink

    ic_value = np.round(np.log2(alpha / beta), decimals)
    ic_lower = np.round(np.log2(gamma.ppf(sign_level / 2, a=alpha, scale=1/beta)), decimals)
    ic_upper = np.round(np.log2(gamma.ppf(1 - sign_level / 2, a=alpha, scale=1/beta)), decimals)

    return ic_value, ic_lower, ic_upper


def calculate_dpa_metrics(df, a_col='a', b_col='b', c_col='c', d_col='d', sign_level: float = 0.01,
                          shrink: float = 0.5):
    """
    Calculate DPA metrics based on contingency matrix.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 8.
    a_col: str
        Name of a column for drug:reaction counts
    b_col: str
        Name of a column for drug:not reaction counts
    c_col: str
        Name of a column for not drug:reaction counts
    d_col: str
        Name of a column for not drug:not reaction counts
    sign_level: float
        Significance level when calculating the CI. Default is 0.05. In the study we used 0.001.
        Only affects Confidence Intervals.
    shrink: float
        Shrinkage factor for Information Component. Default is 0.5.

    Returns
    -------
    df: pd.DataFrame
    """

    a = df[a_col].to_numpy()
    b = df[b_col].to_numpy()
    c = df[c_col].to_numpy()
    d = df[d_col].to_numpy()

    prr_values, prr_lower, prr_upper = proportional_reporting_rate(a=a, b=b, c=c, d=d, sign_level=sign_level)
    ror_values, ror_lower, ror_upper = reporting_odds_ratio(a=a, b=b, c=c, d=d, sign_level=sign_level)
    ic_values, ic_lower, ic_upper = information_component(a=a, b=b, c=c, d=d, sign_level=sign_level, shrink=shrink)

    df = df.assign(prr=prr_values, prr_lower=prr_lower, prr_upper=prr_upper, ror=ror_values, ror_lower=ror_lower,
                   ror_upper=ror_upper, ic=ic_values, ic_lower=ic_lower, ic_upper=ic_upper)
    return df


def clean_metric_triplet(df: pd.DataFrame, metric: str):
    """
    Check for missing values in CI_lower, Value, CI_upper and remove rows with missing values.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 8.
    metric: str
        One of: PRR, ROR, IC

    Returns
    -------
    df: pd.DataFrame
    """

    lower_col = f'{metric}_lower'
    upper_col = f'{metric}_upper'

    def clean_row(row):
        values = [row[metric], row[lower_col], row[upper_col]]
        if all(is_valid_float(val) for val in values):
            return row
        else:
            row[metric] = pd.NA
            row[lower_col] = pd.NA
            row[upper_col] = pd.NA
            return row

    return df.apply(clean_row, axis=1)


def metric_to_label(row, metric: str, min_records: int = 3):
    """
    Assign cardiotoxicity labels based on thresholds and confidence intervals. Remove entries with less than
    3 records for the drug:reaction pairs.

    Parameters
    ----------
    row: pd.Series
        A row from .iterrows() method
    metric: str
        DPA metric: prr, ror, ic
    min_records: int
        Minimum number of records to keep given entry

    Returns
    -------
    str, Cardiotoxicity label
    """

    thresholds = {'prr': 1.0, 'ror': 1.0, 'ic': 0.0}
    threshold = thresholds.get(metric)

    if row['a'] < min_records:  # we cannot say how toxic a molecule is due to lack of data
        return 'Undefined'

    value_lower, value, value_upper = row[f'{metric}_lower'], row[f'{metric}'], row[f'{metric}_upper']

    if not all(is_valid_float(x) for x in [value, value_lower, value_upper]):  # check if all values are present and not missing
        return 'Ambiguous'

    if threshold <= value_lower:  # i.e. CI to the right of threshold
        return 'High'
    elif value_lower < threshold < value_upper:  # i.e. T within CI
        if threshold < value:
            return 'Moderate'
        elif threshold == value:
            return 'Ambiguous'
        elif value < threshold:
            return 'Low'
        else:
            raise ValueError('Check the setup')
    elif value_upper <= threshold:
        return 'Minimal'
    else:
        raise ValueError('Check the setup')


def conf_score(row, metric: str, dist_power: float = 0.5):
    """
    Calculate confidence score based on the relative position of value and threshold,
    and the width of confidence interval.

    Parameters
    ----------
    row: pd.Series
        A row from .iterrows() method
    metric: str
        DPA metric: prr, ror, ic
    dist_power: float
        Power value for the distance between value and threshold

    Returns
    -------
    confidence_score: float
    """

    thresholds = {'prr': 1.0, 'ror': 1.0, 'ic': 0.0}
    threshold = thresholds.get(metric)

    value_lower, value, value_upper = row[f'{metric}_lower'], row[f'{metric}'], row[f'{metric}_upper']
    label = row[f'{metric}_tox']

    if not all(is_valid_float(x) for x in [value, value_lower, value_upper]):  # check if all values are present and not missing
        return pd.NA

    if label in ['Ambiguous', 'Undefined']:
        return pd.NA
    elif label in ['High', 'Moderate', 'Low', 'Minimal']:
        ci_range = (value_upper - value_lower) + 0.0001
        distance = abs(value - threshold) ** dist_power
        confidence_score = np.round(distance / ci_range, 5)
        return confidence_score
    else:
        raise ValueError('Unknown label')


def mod_sigmoid(x, saturation: float = 2.5):
    """
    Modified sigmoid. Saturation corresponds to the X value at which the function reaches 0.9.

    Parameters
    ----------
    x: float
        Value passed to the sigmoid function
    saturation: float
        Elongation factor

    Returns
    -------
    float
    """


    if not is_valid_float(x):
        return pd.NA

    alpha = -2 / saturation * np.log(9)
    beta = saturation / 2

    value = 1 / (1 + np.exp(alpha * (x - beta)))
    return np.round(value, 5)


def step_9(df: pd.DataFrame, min_records: int = 3, sign_level: float = 0.01, shrink: float = 0.5,
           dist_power: float = 0.5, saturation: float = 2.5):
    """
    In step 9 the following operations are performed:
    - entries with insufficient amount of data to perform DPA are removed
    - the DPA metrics and confidence scores are calculated
    - the cardiotoxicity risk labels are assigned
    - risk labels are binarized

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 8.
    min_records: int
        Minimum number of drug:reaction pairs to keep an entry
    sign_level: float
        Significance level for CI calculation
    shrink: float
        Shrinkage factor for Information Component
    dist_power: float
        Power value for the distance between value and threshold
    saturation: float
        Elongation factor for the modified sigmoid.

    Returns
    -------
    df: pd.DataFrame
    """

    class_map = {
        'High': 1,
        'Moderate': 1,
        'Low': 0,
        'Minimal': 0
    }

    df = calculate_dpa_metrics(df, sign_level=sign_level, shrink=shrink)  # somehow this is vectorized ^_^
    df = df.astype({'prr': 'Float32', 'prr_lower': 'Float32', 'prr_upper': 'Float32',
                    'ror': 'Float32', 'ror_lower': 'Float32', 'ror_upper': 'Float32',
                    'ic': 'Float32', 'ic_lower': 'Float32', 'ic_upper': 'Float32'})

    for metric in ['prr', 'ror', 'ic']:
        df = clean_metric_triplet(df, metric)
        df.loc[:, f'{metric}_tox'] = pd.Series([metric_to_label(row, metric=metric, min_records=min_records) for idx, row in df.iterrows()])
        df.loc[:, f'{metric}_conf'] = pd.Series([conf_score(row, metric=metric, dist_power=dist_power) for idx, row in df.iterrows()])
        df.loc[:, f'{metric}_weight'] = df[f'{metric}_conf'].apply(mod_sigmoid, saturation=saturation)
        df.loc[:, f'{metric}_bin'] = df[f'{metric}_tox'].apply(lambda value: class_map.get(value))
        df = df.astype({f'{metric}_tox': 'string', f'{metric}_conf': 'Float32', f'{metric}_weight': 'Float32'})

    return df


"""
Step 10
"""


def smooth_labels(row, label_col: str, weight_col: str) -> np.ndarray:
    """
    Convert binary labels and confidence scores into regression labels.

    Parameters
    ----------
    row: pd.Series
        A row from .iterrows() method
    label_col: str
        Name of the column with binary labels
    weight_col: str
        Name of the columns with confidence scores

    Returns
    -------
    float
    """

    value = row[label_col]
    weight = row[weight_col]
    return np.round(np.array(weight * value + (1 - weight) * 0.5, dtype=np.float32), 5)


def step_10(df: pd.DataFrame, strat_type: str, pt_type: str, drop_unknown: bool = True, dpa_metric: str = 'ic'):
    """
    In step 10 the visualise from DPA are aggregated and datatypes assigned. Generally, a lot of data manipulation.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 9.
    strat_type: str
        Stratification type
    pt_type: str
        Cardiotoxicity definition type
    drop_unknown: bool
        Flag to remove entries with missing information
    dpa_metric: str
        DPA metric for which the data manipulation is performed

    Returns
    -------
    df: pd.DataFrame
    """

    sex_order = CategoricalDtype(categories=['Male', 'Female', 'Unknown'], ordered=True)
    age_order = CategoricalDtype(categories=['Children', 'Adolescent', 'Adult', 'Elderly', 'Unknown'], ordered=True)
    wgt_order = CategoricalDtype(categories=['Low', 'Average', 'High', 'Unknown'], ordered=True)

    all_metrics = ['prr', 'ror', 'ic']
    all_metrics.pop(all_metrics.index(dpa_metric))

    for metric in all_metrics:
        df = df.drop(columns=[metric, f'{metric}_lower', f'{metric}_upper', f'{metric}_tox'])

    df = df.drop(columns=pt_type)

    strats = {'n': [],
              's': ['Sex'],
              'a': ['Age'],
              'w': ['Weight'],
              'sa': ['Sex', 'Age'],
              'sw': ['Sex', 'Weight'],
              'aw': ['Age', 'Weight'],
              'saw': ['Sex', 'Age', 'Weight']
              }

    df = df[~df[f'{dpa_metric}_tox'].isin(['Undefined', 'Ambiguous'])]

    if drop_unknown and (strat_groups := strats.get(strat_type)) != 'n':
        for group in strat_groups:
            df = df[df[group] != 'Unknown'].copy()

    if 'Sex' not in df.columns:
        df = df.assign(Sex='Unknown')
    if 'Age' not in df.columns:
        df = df.assign(Age='Unknown')
    if 'Weight' not in df.columns:
        df = df.assign(Weight='Unknown')

    df['Sex'] = df['Sex'].replace({'M': 'Male', 'F': 'Female'})

    df['Strat'] = strat_type
    df[f'{dpa_metric}_smooth'] = df.apply(smooth_labels, label_col=f'{dpa_metric}_bin', weight_col=f'{dpa_metric}_weight', axis=1)
    df = df.astype({'Sex': sex_order, 'Age': age_order, 'Weight': wgt_order, 'Strat': 'string',
                    'a': 'UInt64', 'b': 'UInt64', 'c': 'UInt64', 'd': 'UInt64'})

    df = df.loc[:, ['Strat', 'Sex', 'Age', 'Weight', 'smi_enc', dpa_metric,
                    f'{dpa_metric}_lower', f'{dpa_metric}_upper', f'{dpa_metric}_tox',
                    f'{dpa_metric}_bin', f'{dpa_metric}_conf', f'{dpa_metric}_weight',
                    f'{dpa_metric}_smooth', 'a', 'b', 'c', 'd']]
    return df


"""
Step 11
"""


def combine_arrays(row, columns: List[str]) -> np.ndarray:
    """
    Combine arrays from multiple columns.

    Parameters
    ----------
    row: pd.Series
        A row from .iterrows() method
    columns: List[str]
        A list of column names to combine

    Returns
    -------
    np.array
    """

    arrays = np.concatenate([row[col] for col in columns], axis=0)
    return arrays


def sex_to_array(sex) -> np.ndarray:
    """
    Map gender to corresponding numpy ndarray
    """
    mapping = {'Male':    np.array([1, 0], dtype=np.int64),
               'Female':  np.array([0, 1], dtype=np.int64),
               'Unknown': np.array([0, 0], dtype=np.int64)}
    return mapping.get(sex, np.nan)


def age_to_array(age) -> np.ndarray:
    """
    Map age to corresponding numpy ndarray
    """
    mapping = {'Children':   np.array([1, 0, 0, 0], dtype=np.int64),
               'Adolescent': np.array([0, 1, 0, 0], dtype=np.int64),
               'Adult':      np.array([0, 0, 1, 0], dtype=np.int64),
               'Elderly':    np.array([0, 0, 0, 1], dtype=np.int64),
               'Unknown':    np.array([0, 0, 0, 0], dtype=np.int64)}
    return mapping.get(age, np.nan)


def wgt_to_array(weight) -> np.ndarray:
    """
    Map weight to corresponding numpy ndarray
    """

    mapping = {'Low':     np.array([1, 0, 0], dtype=np.int64),
               'Average': np.array([0, 1, 0], dtype=np.int64),
               'High':    np.array([0, 0, 1], dtype=np.int64),
               'Unknown': np.array([0, 0, 0], dtype=np.int64)}
    return mapping.get(weight, np.nan)


def pack_to_tuple(row, cols: List[str]):
    """
    Combine entries from multiple columns into a single tuple
    """
    return tuple([row[col] for col in cols])


def pack_to_array(row, cols: List[str]):
    """
    Combine entries from multiple columns into a single array
    """
    return np.array([row[col] for col in cols])


def step_11(df: pd.DataFrame, smiles_col: str, idx_2_smi: dict, dpa_metric: str) -> pd.DataFrame:
    """
    In step 11 the following operation are performed:
    - decoding of SMILES indices
    - encoding of demographic factors as numpy arrays (yes, again)
    - concatenation of demographic arrays
    - concatenation of demographic factors
    - packing of values from DPA analysis
    - column renaming, removal, and casting

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 10
    smiles_col: str
        Name of a column with encoded SMILES
    idx_2_smi: Dict[int, str]
        Dictionary mapping from index to SMILES
    dpa_metric: str
        DPA metric for which the clean-up is performed

    Returns
    -------
    df: pd.DataFrame
    """

    df.loc[:, 'SMILES'] = [idx_2_smi.get(idx) for idx in df[smiles_col]]
    df['Sex_enc'] = [sex_to_array(value) for value in df['Sex']]
    df['Age_enc'] = [age_to_array(value) for value in df['Age']]
    df['Weight_enc'] = [wgt_to_array(value) for value in df['Weight']]
    df['Demo'] = df.apply(combine_arrays, columns=['Sex_enc', 'Age_enc', 'Weight_enc'], axis=1)

    df['Signature'] = df.apply(pack_to_tuple, cols=['Sex', 'Age', 'Weight'], axis=1)
    df['DPA_confusion'] = df.apply(pack_to_array, cols=['a', 'b', 'c', 'd'], axis=1)
    df['DPA_values'] = df.apply(pack_to_array, cols=[dpa_metric, f'{dpa_metric}_lower', f'{dpa_metric}_upper'], axis=1)
    df['DPA_values'] = df['DPA_values'].apply(lambda array: np.round(array.reshape(-1), 5))

    df = df.rename(columns={
        'Strat': 'Stratification',
        f'{dpa_metric}_tox': 'Risk',
        f'{dpa_metric}_bin': 'Label',
        f'{dpa_metric}_conf': 'Label_confidence',
        f'{dpa_metric}_weight': 'Label_weight',
        f'{dpa_metric}_smooth': 'Label_regression',
        'Demo': 'DemoFP'
    })

    df = df.drop(columns=['Sex', 'Age', 'Weight', 'a', 'b', 'c', 'd', 'smi_enc', 'Sex_enc', 'Age_enc', 'Weight_enc'])

    df = df.astype({
        'Stratification': 'string',
        'Risk': 'string',
        'Label': 'UInt8',
        'Label_confidence': 'Float32',
        'Label_weight': 'Float32',
        'Label_regression': 'Float32',
        'SMILES': 'string',
    })

    df = df.loc[:, ['SMILES', 'Signature', 'Risk', 'Label', 'Label_confidence', 'Label_weight', 'Label_regression',
                    'DPA_confusion', 'DPA_values', 'DemoFP', 'Stratification']]

    return df


def step_12(df: pd.DataFrame, fold_df: pd.DataFrame, smiles_col: str, dataset_type:str, dpa_type: str, pt_set: str) -> pd.DataFrame:
    """
    In step 12 the information about cluster and fold assignment is added, and identifying attributes are given.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame from step 11
    fold_df: pd.DataFrame
        The DataFrame with Cluster and Fold assignment for unique SMILES. Output of src/split/butina_split
    smiles_col: str
        Name of a column with encoded SMILES
    dataset_type: str
        Primary / Secondary
    dpa_type: str
        PRR / ROR / IC
    pt_set: str
        Cred / Card / Cvas

    Returns
    -------
    pd.DataFrame
    """

    df.attrs = {
        'Dataset': dataset_type,
        'Metric': dpa_type,
        'Set': pt_set
    }

    df = df.merge(fold_df, on=smiles_col, how='left')

    if df.Cluster_ID.isna().sum() != 0:
        print(f'Some compounds in {dataset_type}:{dpa_type}:{pt_set} have not been assigned to a cluster')
    if df.Fold.isna().sum() != 0:
        print(f'Some compounds in {dataset_type}:{dpa_type}:{pt_set} have not been assigned to a fold')

    df = df.astype({'SMILES': 'string', 'Fold': 'UInt8', 'Cluster_ID': 'UInt16'})
    return df

