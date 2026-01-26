"""
Module for DatasetManager class
"""

from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.data.utils import IndexedDict
from src.ml.models import FoldUnit


class DatasetManager:
    """
    Handles dataset preparation, including descriptor processing, demo features, labels,
    sample weights, cross-validation splits, and signature metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing all necessary data columns.
    desc_col : str
        Name of the column containing molecular descriptors.
    demo_col : str, optional
        Column name for demographic features (default is 'DemoFP').
    label_col : str, optional
        Column name for target labels (default is 'Label').
    weight_col : str, optional
        Column name for label/sample weights (default is 'Label_weight').
    fold_col : str, optional
        Column name specifying cross-validation fold splits (default is 'Fold').
    sign_col : str, optional
        Column name for signature metadata (default is 'Signature').
    sign_names : List[str], optional
        List of names for signature dimensions (default is ['Sex', 'Age', 'Weight']).

    Attributes
    ----------
    x_array : np.ndarray
        Descriptor matrix for all samples.
    x_demo : np.ndarray
        Demographic feature matrix.
    y_true : np.ndarray
        Target label array.
    y_wgts : np.ndarray
        Sample weight array.
    y_sign : IndexedDict
        Dictionary of signature metadata arrays.
    folds : List[int]
        List of unique fold identifiers.
    train_idxs : Dict[int, np.ndarray]
        Mapping from fold to training sample indices.
    eval_idxs : Dict[int, np.ndarray]
        Mapping from fold to evaluation sample indices.

    Methods
    -------
    get_train_eval_data(fold)
        Returns training and evaluation sets for the specified fold.
    get_full_data()
        Returns the full dataset as a dictionary.

    Notes
    -----
    Seemed like a great idea, but inference is very annoying to set up. Solved with the visualise.export module
    """

    def __init__(self, df: pd.DataFrame, desc_col: str, demo_col: str = 'DemoFP', label_col: str = 'Label',
                 weight_col: str = 'Label_weight', fold_col: str = 'Fold', sign_col: str = 'Signature',
                 sign_names: List[str] = None):

        self.desc_col = desc_col
        self.demo_col = demo_col
        self.label_col = label_col
        self.weight_col = weight_col
        self.fold_col = fold_col
        self.sign_col = sign_col
        self.sign_names = sign_names if sign_names is not None else ['Sex', 'Age', 'Weight']

        self.x_array = np.vstack(df[self.desc_col].to_numpy())
        self.x_demo = np.vstack(df[self.demo_col].to_numpy())

        self.y_true = np.vstack(df[self.label_col].to_numpy()).reshape(-1)
        self.y_wgts = np.vstack(df[self.weight_col].to_numpy()).reshape(-1)

        self.splits = np.vstack(df[self.fold_col].to_numpy()).reshape(-1)
        self.folds = sorted(df[self.fold_col].unique())

        self.y_sign = IndexedDict({key: np.array(values) for key, values in zip(sign_names, list(zip(*df[self.sign_col].tolist())))})

        self.train_idxs = {fold: np.where(self.splits != fold)[0] for fold in self.folds}
        self.eval_idxs = {fold: np.where(self.splits == fold)[0] for fold in self.folds}

        self.selectors = {}
        self.scalers = {}

    def get_train_eval_data(self, fold: int) -> Dict[str, Dict[str, np.ndarray]]:

        train_idx = self.train_idxs[fold]
        eval_idx = self.eval_idxs[fold]

        train_eval_data = {
            'Train': {
                'X': self.x_array[train_idx, :],
                'demo': self.x_demo[train_idx, :],
                'y': self.y_true[train_idx],
                'wgts': self.y_wgts[train_idx],
                'sign': self.y_sign[train_idx]
            },
            'Eval': {
                'X': self.x_array[eval_idx, :],
                'demo': self.x_demo[eval_idx, :],
                'y': self.y_true[eval_idx],
                'wgts': self.y_wgts[eval_idx],
                'sign': self.y_sign[eval_idx]
            }
        }

        return train_eval_data

    def get_full_data(self) -> Dict[str, np.ndarray]:

        full_data = {
            'X': self.x_array,
            'demo': self.x_demo,
            'y': self.y_true,
            'wgts': self.y_wgts,
            'sign': self.y_sign
        }
        return full_data
