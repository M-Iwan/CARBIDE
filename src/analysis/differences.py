"""
Module with functions for calculating the prediction difference caused by demographic features
"""

import os
import joblib

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.ml.manager import DatasetManager
from src.ml.models import IterEnsemble


def entry_difference(ensemble: IterEnsemble, x_array: np.ndarray, x_base: np.ndarray):
    """
    Calculate impact of demographic features by comparing predictions when a specific
    feature is known vs. predictions with baseline population demographics.

    Parameters
    ----------
    ensemble: IterEnsemble
        The ensemble model to use for the analysis.
    x_array : np.ndarray
        Molecular descriptor array for a single sample.
    x_base: np.ndarray
        Base feature array containing demographic population means.

    Returns
    -------
    np.ndarray
        Array containing the difference in predictions for each demographic feature.
    """

    if x_array.ndim == 1 or x_array.shape[0] == 1:
        x_array = x_array.reshape(1, -1)

    sex_idxs = [0, 1]  # Male, Female
    age_idxs = [2, 3, 4, 5]  # Children, Adolescent, Adult, Elderly
    wgt_idxs = [6, 7, 8]  # Low, Average, High

    demo_res = np.zeros(shape=(1, 9))
    base_res = np.zeros(shape=(1, 9))

    base_pred = ensemble.predict_proba(x_array, x_base)

    for demo_idx in range(9):
        x_demo = x_base.copy()

        # Set all sex features to 0, then target feature to 1
        if demo_idx in sex_idxs:
            for idx in sex_idxs:
                x_demo[0, idx] = 0
            x_demo[0, demo_idx] = 1

        elif demo_idx in age_idxs:
            for idx in age_idxs:
                x_demo[0, idx] = 0
            x_demo[0, demo_idx] = 1

        elif demo_idx in wgt_idxs:
            for idx in wgt_idxs:
                x_demo[0, idx] = 0
            x_demo[0, demo_idx] = 1

        demo_pred = ensemble.predict_proba(x_array, x_demo)

        demo_res[0, demo_idx] = demo_pred
        base_res[0, demo_idx] = base_pred

    return demo_res - base_res


def fold_difference(df: pd.DataFrame, ensemble: IterEnsemble, desc_col: str, smiles_col: str, x_base: np.ndarray):
    """
    Calculate and save attributions for all samples in a specific test fold.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing test samples for the fold.
    ensemble: IterEnsemble
        Trained instance of IterEnsemble.
    desc_col : str
        Name of the column containing molecular descriptors.
    smiles_col : str
        Name of the column containing SMILES strings.
    x_base: np.ndarray
        Base feature array for a single sample.

    Returns
    -------
    dict
        Dictionary containing attribution arrays for the fold.
    """

    smis = df[smiles_col].tolist()
    x_arrays = df[desc_col].to_numpy()

    dc = {}

    for smiles, x_array in tqdm(zip(smis, x_arrays), total=len(smis)):
        dc[smiles] = entry_difference(ensemble=ensemble,
            x_array=x_array,
            x_base=x_base
        )

    return dc


def data_difference(dataset_path: str, desc_path: str, ensemble_dir: str, save_path: str):
    """
    Perform attribution analysis across all folds of a dataset.

    Loads dataset and descriptors, processes each fold, computes attributions for all samples, and aggregates statistics across folds.

    Parameters
    ----------
    dataset_path : str
        Path to the pickled dataset file.
    desc_path : str
        Path to the pickled descriptors file.
    ensemble_dir : str
        Directory containing ensemble model files.
    save_path : str
        Path to save attribution visualise and summaries.

    Returns
    -------
    dict
        Dictionary containing summarized attribution visualise.
    """

    dataset = joblib.load(dataset_path)
    descriptors = joblib.load(desc_path)

    desc_col = ensemble_dir.rstrip('/').split('/')[-1]

    dataset = dataset.merge(descriptors[['SMILES', desc_col]], on='SMILES', how='inner')

    data_res = {}

    for test_fold in dataset['Fold'].unique():
        train_df = dataset[dataset['Fold'] != test_fold]

        train_mgr = DatasetManager(
            df=train_df,
            desc_col=desc_col,
            sign_names=['Sex', 'Age', 'Weight']
        )

        x_base = train_mgr.x_demo.mean(axis=0).reshape(1, -1)

        test_df = dataset[dataset['Fold'] == test_fold].reset_index(drop=True)
        test_df = test_df.drop_duplicates(subset='SMILES', keep='first')

        ensemble_path = os.path.join(ensemble_dir, f'ensemble_tf_{test_fold}.joblib')
        ensemble = joblib.load(ensemble_path)

        res = fold_difference(
            df=test_df,
            ensemble=ensemble,
            desc_col=desc_col,
            smiles_col='SMILES',
            x_base=x_base,
        )

        data_res[test_fold] = res

    joblib.dump(data_res, save_path)

    return data_res
