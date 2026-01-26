import argparse
import os, joblib
from math import floor
from copy import deepcopy
from collections import defaultdict
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import polars as pl

from scipy.special import expit

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

import optuna


parser = argparse.ArgumentParser(prog="PMD:QCG-PilotJOb")

parser.add_argument("--input_dir", help="Path to directory with CARBIDE datasets")
parser.add_argument("--output_dir", help="Path to directory to save visualise")
parser.add_argument("--dataset_type", help="Dataset type: primary or secondary")
parser.add_argument("--pt_set", help="PT set used for cardiotoxicity definition: Cred or Card or Cvas")
parser.add_argument("--dpa_metric", help="DPA metric used for label calculation")
parser.add_argument("--model_name", help="Name of the ML model to use")
parser.add_argument("--desc_col", help="Descriptors name")
parser.add_argument("--sel_metric", help="Optuna selection metric")
parser.add_argument("--n_trials", help="Number of Optuna trials to run", type=int)
parser.add_argument("--n_jobs", help="Number of CPUs to use", type=int)
parser.add_argument("--test_fold", help="Fold to be used as a test set", type=int)

args = parser.parse_args()


class IndexedDict(dict):
    """
    Custom implementation of a dictionary that uses numpy arrays for splitting held data
    """
    def __getitem__(self, key):
        if isinstance(key, (int, str)):
            # Regular dict behavior for single int or string key
            return super().__getitem__(key)
        elif isinstance(key, (np.ndarray, list)):
            sliced = {}
            for k, arr in self.items():
                if isinstance(arr, np.ndarray):
                    sliced[k] = arr[key]
                else:
                    sliced[k] = arr
            return sliced
        else:
            raise ValueError(f'Unrecognized key format: {type(key)}')


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


class DataTransformer:
    """
    Class for data manipulation prior to fitting ML/DL models.
    __init__(self, use_masks: bool = True, use_corr: bool = False, use_imputer: bool = True,
             use_selector: bool = True, use_scaler: bool = True):
    """
    def __init__(self, use_masks: bool = True, use_corr: bool = False, use_imputer: bool = True,
                 use_selector: bool = True, use_scaler: bool = True):
        """
        Initialize the DataTransformer object.
        TODO: Finish the docstring

        Parameters
        ----------
        use_masks: bool = True
            A flag to remove missing, infinite, very large or zero-variance features.
        use_corr: bool = False
            A flag to perform inter-feature correlation analysis and remove those with correlation above 0.9
        use_imputer: bool = True
            A flag to imputing missing values during inference.
        use_selector: bool = True
            A flag to remove features with near-zero variance. Threshold is 1e-3.
        use_scaler: bool = True
            A flag to scale the features using RobustScaler.
        """

        self.imputer = None
        self.selector = None
        self.scaler = None
        self.masks = {}
        self.n_features = {}
        self.use_masks = use_masks
        self.use_corr = use_corr  # goes with masks, no separate attribute
        self.use_imputer = use_imputer
        self.use_selector = use_selector
        self.use_scaler = use_scaler
        self.masks_fit = False
        self.corr_fit = False
        self.imputer_fit = False
        self.selector_fit = False
        self.scaler_fit = False


    def __repr__(self):
        return (f"<{self.__class__.__name__}>: use_masks: {self.use_masks}, use_corr: {self.use_corr}, "
                f"use_imputer: {self.use_imputer}, use_selector: {self.use_selector}, use_scaler: {self.use_scaler}>")

    def __str__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    def validate_features(x_array: np.ndarray):
        """
        Check validity of passed feature values and convert to 2D numpy array if needed.
        """
        if not isinstance(x_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(x_array)} instead.")

        if x_array.ndim > 2:
            raise ValueError(f"Features may have at most 2 dimensions, got {x_array.ndim} dimensions instead.")

        if x_array.ndim == 1:  # a single entry
            x_array = x_array.reshape(1, -1)

        return x_array

    @staticmethod
    def validate_targets(y_array: np.ndarray):
        """
        Check validity of passed target values and convert to 1D numpy array if needed.
        """
        if not isinstance(y_array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(y_array)} instead.")

        if (y_array.ndim == 2 and y_array.shape[1] > 1) or y_array.ndim > 2:
            raise ValueError(f"Features must be convertible to 1D array, got {y_array.ndim} dimensions instead.")

        if y_array.ndim == 2 and y_array.shape[1] == 1:
            y_array = y_array.reshape(-1)

        return y_array

    def fit_masks(self, x_array: np.ndarray, update: bool = False):
        """
        Prepare numpy masks for filtering out missing, infinite, zero-variance or very large features.
        """
        if self.masks_fit and not update:
            raise RuntimeError('Cannot fit masks twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)

        self.masks['missing'] = np.isnan(x_array).any(axis=0)
        self.masks['infinite'] = np.isinf(x_array).any(axis=0)
        self.masks['zero'] = x_array.var(axis=0) == 0
        self.masks['large'] = (np.abs(x_array) > 1e12).any(axis=0)

        self.masks['combined'] = ~(self.masks['missing'] | self.masks['infinite'] | self.masks['zero'] | self.masks['large'])
        self.masks_fit = True

    def apply_masks(self, x_array: np.ndarray):
        """
        Apply learnt masks.
        """
        if not self.masks_fit:
            raise RuntimeError('Masks have not been fit yet. Please call .fit_masks() first.')

        x_array = self.validate_features(x_array)
        self.n_features['start'] = x_array.shape[1]
        x_array = x_array[:, self.masks['combined']]
        self.n_features['masks'] = x_array.shape[1]

        return x_array

    def fit_corr(self, x_array: np.ndarray, threshold: float = 0.9, frac: float = 1.0, update: bool = False):
        """
        Prepare a feature-correlation mask. Features that correlate highly with multiple features are removed first.
        Mean correlation is used for tie-breakers. The frac parameter can be used to speed up computations
        for larger datasets.
        """

        if self.corr_fit and not update:
            raise RuntimeError('Cannot fit correlation mask twice. To overwrite this setting, set update to True.')

        self.masks['correlation'] = np.ones(x_array.shape[1], dtype=bool)

        if frac < 1.0:
            if frac <= 0.0 or frac > 1.0:
                raise ValueError(f"Fraction {frac} must be between 0 and 1.")

            sample_size = max(floor(x_array.shape[0] * frac), 2)
            np.random.seed(42)
            sample_indices = np.random.choice(x_array.shape[0], size=sample_size, replace=False)
            x_array = x_array[sample_indices, :]

        correlation_matrix = np.abs(np.triu(np.corrcoef(x_array.T), k=1))
        rows, cols = np.where(correlation_matrix >= threshold)

        if len(rows) == 0:  # no highly correlated features
            self.corr_fit = True
            return

        df = pl.DataFrame({
            'Idx1': rows,
            'Idx2': cols,
            'Correlation': [correlation_matrix[row, col] for row, col in zip(rows, cols)]
        }).sort('Correlation', descending=True)

        while len(df) > 0:

            counts = (pl.concat([
                df[['Idx1', 'Correlation']].rename({'Idx1': 'Idx'}),
                df[['Idx2', 'Correlation']].rename({'Idx2': 'Idx'})
            ])
            .group_by('Idx')
            .agg([
                pl.col('Correlation').len().alias('Count'),
                pl.col('Correlation').mean().round(5).alias('Mean_Correlation')
            ])
            .sort(['Count', 'Mean_Correlation'], descending=[True, True])
            )

            max_corr_idx = counts['Idx'].item(0)
            self.masks['correlation'][max_corr_idx] = False

            df = df.filter(
                (pl.col('Idx1') != max_corr_idx) &
                (pl.col('Idx2') != max_corr_idx)
            )

        self.corr_fit = True
        return

    def apply_corr(self, x_array: np.ndarray):
        if not self.corr_fit:
            raise RuntimeError('Correlation mask has not been fit yet. Please call .fit_corr() first.')

        x_array = self.validate_features(x_array)
        x_array = x_array[:, self.masks['correlation']]
        self.n_features['correlation'] = x_array.shape[1]

        return x_array

    def fit_imputer(self, x_array: np.ndarray, update: bool = False):
        if self.imputer_fit and not update:
            raise RuntimeError('Cannot fit imputer twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer.fit(x_array)
        self.imputer_fit = True

    def apply_imputer(self, x_array: np.ndarray):
        if not self.imputer_fit:
            raise RuntimeError('Imputer has not been fit yet. Please call .fit_imputer first.')

        x_array = self.validate_features(x_array)
        x_array = self.imputer.transform(x_array)
        return x_array

    def fit_selector(self, x_array: np.ndarray, selector_threshold: float = 1e-3, update: bool = False):
        if self.selector_fit and not update:
            raise RuntimeError('Cannot fit selector twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.selector = VarianceThreshold(selector_threshold)
        self.selector.fit(x_array)
        self.selector_fit = True

    def apply_selector(self, x_array: np.ndarray):
        if not self.selector_fit:
            raise RuntimeError('Selector has not been fit yet. Please call .fit_selector first.')

        x_array = self.validate_features(x_array)
        x_array = self.selector.transform(x_array)
        self.n_features['selector'] = x_array.shape[1]
        return x_array

    def fit_scaler(self, x_array: np.ndarray, update: bool = False):
        if self.scaler_fit and not update:
            raise RuntimeError('Cannot fit scaler twice. To overwrite this setting, set update to True.')

        x_array = self.validate_features(x_array)
        self.scaler = RobustScaler(unit_variance=True, quantile_range=(5, 95))
        self.scaler.fit(x_array)
        self.scaler_fit = True

    def apply_scaler(self, x_array: np.ndarray):
        if not self.scaler_fit:
            raise RuntimeError('Scaler has not been fit yet. Please call .fit_scaler first.')

        x_array = self.validate_features(x_array)
        x_array = self.scaler.transform(x_array)
        return x_array

    def transform(self, x_array: np.ndarray):
        """
        Transform data for inference or evaluation.
        """
        x_array = self.validate_features(x_array)

        if self.use_masks:
            x_array = self.apply_masks(x_array)
        if self.use_corr:
            x_array = self.apply_corr(x_array)
        if self.use_imputer:
            x_array = self.apply_imputer(x_array)
        if self.use_selector:
            x_array = self.apply_selector(x_array)
        if self.use_scaler:
            x_array = self.apply_scaler(x_array)

        return x_array


def prepare_transformer(descriptors: str):
    """
    Infer correct parameters based on descriptors type.

    Parameters
    ----------
    descriptors: str
        Name of descriptors used

    Returns
    -------
    transformer: DataTransformer
    """

    continuous = {
        'use_masks': True,
        'use_corr': True,
        'use_imputer': True,
        'use_selector': True,
        'use_scaler': True,
    }

    binary = {
        'use_masks': True,
        'use_corr': True,
        'use_imputer': True,
        'use_selector': True,
        'use_scaler': False,
    }

    descriptors_flags = {
        'Morgan': binary,
        'MACCS': binary,
        'Klek': binary,
        'CDDD': continuous,
        'ChemBERTa': continuous,
        'RDKit': continuous,
    }

    if descriptors not in descriptors_flags.keys():
        raise ValueError(f'Unrecognized descriptors: {descriptors} passed during DataTransformer preparation')

    flags = descriptors_flags[descriptors]

    return DataTransformer(**flags)


class FoldUnit:
    """
    FoldUnit encapsulates training, evaluation, and transformation logic for a single fold
    during cross-validation.

    This class manages model fitting, prediction, and scoring using fold-specific feature
    selectors and scalers. It assumes external assignment of selectors and scalers
    through class variables.

    Attributes
    ----------
    model : object
        A scikit-learn-compatible model instance used for training and inference.
    transformer : DataTransformer
        An instance of DataTransformer class.
    fold : int
        The fold index corresponding to this unit.
    scores : dict
        A dictionary storing evaluation metrics across training and evaluation sets.

    Methods
    -------
    fit(x_train, x_demo, y_array, y_wgts)
        Fits the model on transformed training data.
    predict(x_array, x_demo)
        Predicts class labels for the given input.
    predict_proba(x_array, x_demo)
        Predicts probabilities (or scores) for the positive class.
    transform(array)
        Applies the fold-specific selector and scaler to the input features.
    """

    def __init__(self, model, transformer, fold_idx: int):
        self.model = model
        self.transformer = transformer
        self.fold = fold_idx
        self.scores = {}

    def __repr__(self):
        return f"<FoldUnit(fold={self.fold}, model={self.model.__class__.__name__})>"

    def __str__(self):
        eval_score = self.scores.get(('Eval', 'Weighted'), {})
        total_metrics = eval_score.get('Total', {})
        metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in total_metrics.items())
        return f"Fold {self.fold} | Model: {self.model.__class__.__name__} | Eval Scores: {metrics_str}"

    def fit(self, x_train: np.ndarray, x_demo: np.ndarray, y_array: np.ndarray, y_wgts: np.ndarray):

        if self.transformer.use_masks:
            if not self.transformer.masks_fit:
                self.transformer.fit_masks(x_train)
            x_train = self.transformer.apply_masks(x_train)

        if self.transformer.use_corr:
            if not self.transformer.corr_fit:
                self.transformer.fit_corr(x_train)
            x_train = self.transformer.apply_corr(x_train)

        if self.transformer.use_imputer:
            if not self.transformer.imputer_fit:
                self.transformer.fit_imputer(x_train)
            x_train = self.transformer.apply_imputer(x_train)

        if self.transformer.use_selector:
            if not self.transformer.selector_fit:
                self.transformer.fit_selector(x_train)
            x_train = self.transformer.apply_selector(x_train)

        if self.transformer.use_scaler:
            if not self.transformer.scaler_fit:
                self.transformer.fit_scaler(x_train)
            x_train = self.transformer.apply_scaler(x_train)

        x_array = np.hstack((x_train, x_demo))
        self.model.fit(x_array, y_array, sample_weight=y_wgts)

    def predict(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        x_array = self.transformer.transform(x_array)
        x_array = np.hstack((x_array, x_demo))
        return self.model.predict(x_array)

    def predict_proba(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        x_array = self.transformer.transform(x_array)
        x_array = np.hstack((x_array, x_demo))

        if hasattr(self.model, 'predict_proba'):
            y_score = self.model.predict_proba(x_array)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            y_score = expit(self.model.decision_function(x_array))
        else:
            raise ValueError(f"{type(self.model).__name__} must implement either decision_function or predict_proba")
        return y_score


class IterEnsemble:
    """
    IterEnsemble represents an ensemble of FoldUnit models from a single Optuna trial iteration.

    It aggregates fold-level evaluation scores, provides averaged summary statistics,
    and supports ensemble-level probability prediction via averaging.

    Attributes
    ----------
    units : List[FoldUnit]
        List of FoldUnit instances, each corresponding to a single fold in cross-validation.
    iter : int
        Index of the Optuna iteration associated with this ensemble.
    folds : List[int]
        List of fold indices used in this ensemble.
    eval_scores : List[dict]
        List of evaluation scores (Eval, Weighted) from each FoldUnit.
    summary : dict
        Dictionary containing averaged performance metrics across folds.
    optuna_score : Union[float, None]
        Score used by Optuna to evaluate this iteration (usually from the selection metric).
    hyperparameters : Union[Dict, None]
        Hyperparameter set used to initialize models in this ensemble.

    Methods
    -------
    predict_proba(x_array, x_demo)
        Predicts class probabilities by averaging outputs from all FoldUnit models.
    make_summary()
        Computes mean and standard deviation of metrics across folds for summary reporting.
    """

    def __init__(self, units: List[FoldUnit], iter_idx: int):
        self.units = units
        self.iter = iter_idx
        self.folds = [unit.fold for unit in self.units]
        self.eval_scores = [unit.scores[('Eval', 'Weighted')] for unit in self.units]
        self.summary = self.make_summary()
        self.optuna_score = None
        self.hyperparameters = None

    def __repr__(self):
        return f"<IterEnsemble(iter={self.iter}, folds={self.folds})>"

    def __str__(self):
        scores = self.summary.get('AvTotal', {})
        score_str = ', '.join(f"{k}: {v[0]:.4f}±{v[1]:.4f}" for k, v in scores.items())
        return f"IterEnsemble #{self.iter} | Eval Scores: {score_str}"

    def predict(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        predictions = [
            model.predict(x_array, x_demo) for model in self.units
        ]
        y_pred = np.column_stack(predictions).mean(axis=1)
        return y_pred

    def predict_proba(self, x_array: np.ndarray, x_demo: np.ndarray) -> np.ndarray:
        predictions = [
            model.predict_proba(x_array, x_demo) for model in self.units
        ]
        y_score = np.column_stack(predictions).mean(axis=1)
        return y_score

    def make_summary(self) -> Dict[str, Dict]:
        total_metrics = defaultdict(list)
        sign_metrics = defaultdict(lambda: defaultdict(list))  # {sign_key: {metric: [values]}}

        for score in self.eval_scores:
            for metric, value in score['Total'].items():
                total_metrics[metric].append(value)

            for sign_key, metrics in score['Sign'].items():
                for metric, value in metrics.items():
                    sign_metrics[sign_key][metric].append(value)

        av_total = {metric: (np.round(np.mean(vals), 5), np.round(np.std(vals), 5)) for metric, vals in total_metrics.items()}
        av_sign = {
            sign_key: {
                metric: (np.round(np.mean(vals), 5), np.round(np.std(vals), 5)) for metric, vals in metrics.items()
            }
            for sign_key, metrics in sign_metrics.items()
        }

        return {
            'AvTotal': av_total,
            'AvSign': av_sign
        }


class TuneArmy:
    """
    Stores and manages visualise from Optuna trials.

    This class holds a collection of IterEnsemble instances, each corresponding to
    a single hyperparameter optimization trial. It provides functionality to
    retrieve the best-performing ensemble based on the Optuna selection metric.
    Somewhat useless if you do not plan on saving all models. Boh.

    Attributes
    ----------
    ensembles : List[IterEnsemble]
        A list of IterEnsemble objects, each representing an Optuna trial.
    iter_scores : List[float]
        List of Optuna scores (selection metric values) from each ensemble.

    Methods
    -------
    best()
        Returns the IterEnsemble with the highest Optuna score.
    """

    def __init__(self, iter_ensembles: List[IterEnsemble]):
        self.ensembles = iter_ensembles
        self.iter_scores = [ens.optuna_score for ens in iter_ensembles]

    def best(self):
        """Return the ensemble with the highest optuna_score."""
        if not self.ensembles:
            return None
        return max(self.ensembles, key=lambda e: e.optuna_score)


def get_params_xgb_classifier(trial) -> Dict:
    """
    Suggest values for XGBClassifier for optuna
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 5e-3, 1e-1, log=True),
        'max_leaves': trial.suggest_int('max_leaves', 0, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }


def get_params_random_forest_classifier(trial) -> Dict:
    """
    Suggest values for RandomForestClassifier for optuna
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 0.05, log=True)
    }


def get_params_svc(trial) -> Dict:
    """
    Suggest values for SVC for optuna
    """
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    params = {
        'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
        'kernel': kernel
    }
    if kernel in ['poly', 'rbf', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)
    if kernel in ['poly', 'sigmoid']:
        params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)

    return params


def get_params_logistic_regression(trial) -> Dict:
    """
    Suggest values for LogisticRegression for optuna
    """
    valid_combinations = [
        'lbfgs:l2', 'lbfgs:None',
        'liblinear:l1', 'liblinear:l2',
        'newton-cg:l2', 'newton-cg:None',
        'newton-cholesky:l2', 'newton-cholesky:None',
        'sag:l2', 'sag:None', 'saga:elasticnet',
        'saga:l1', 'saga:l2', 'saga:None'
    ]
    combination = trial.suggest_categorical('solver_penalty', valid_combinations)
    solver, penalty = combination.split(':')
    penalty = None if penalty == 'None' else penalty

    params = {
        'solver': solver,
        'penalty': penalty,
        'C': trial.suggest_float('C', 0.001, 10, log=True),
        'max_iter': 1024,
    }

    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    return params


def get_model(name: str) -> Union[LogisticRegression, RandomForestClassifier, SVC, XGBClassifier]:
    """
    Dictionary of name to model
    """
    models = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "XGBClassifier": XGBClassifier
    }

    if name not in models.keys():
        raise ValueError(f'Unknown model: {name}')
    return models.get(name)


def inner_score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, y_wgts: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Used as inner function with 1D numpy arrays.

    Parameters
    ----------

    y_true: np.ndarray
        Array of true labels
    y_pred: np.ndarray
        Array of predicted labels.
    y_score: np.ndarray
        Array of predicted probabilities (from .predict_proba or .decision_function)
    y_wgts: np.ndarray
        Array of weights. Optional.
    """

    def safe_div(numerator, denominator, default_=0.0):

        return numerator / denominator if denominator != 0 else default_

    if y_wgts is None:  # we don't know the weights
        y_wgts = np.ones_like(y_true)  # just say they are equal and don't think about it anymore!

    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=y_wgts)

    if len(np.unique(y_true)) != 2:
        return {}

    tn, fp, fn, tp = conf_mat.ravel()

    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)

    metrics = {
        'TP': np.round(tp, 5),
        'FP': np.round(fp, 5),
        'FN': np.round(fn, 5),
        'TN': np.round(tn, 5),
        'Accuracy': np.round(safe_div(tp + tn, tp + fp + fn + tn), 5),
        'Recall': np.round(rec, 5),
        'Specificity': np.round(spec, 5),
        'Precision': np.round(safe_div(tp, tp + fp), 5),
        'Balanced Accuracy': np.round((rec + spec) / 2, 5),
        'GeomRS': np.round(np.sqrt(rec * spec), 5),
        'HarmRS': np.round(safe_div(2 * rec * spec, rec + spec), 5),
        'F1 Score': np.round(safe_div(2 * tp, 2 * tp + fp + fn), 5),
        'ROC AUC': np.round(roc_auc_score(y_true=y_true, y_score=y_score, sample_weight=y_wgts), 5),
        'MCC': np.round(safe_div((tp * tn) - (fp * fn), np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))), 5)
    }

    return metrics


def outer_score(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, y_wgts: np.ndarray, y_sign: IndexedDict):
    """
    Aggregate predictions based on different criteria. Intended to be used
    with classical ML metrics, purely on predictions

    Parameters
    ----------

    y_true: np.ndarray
        Array of true labels.
    y_pred: np.ndarray
        Array of predicted labels.
    y_score: np.ndarray
        Array of predicted probabilities (from .predict_proba or .decision_function)
    y_wgts: np.ndarray
        Array of weights. Optional.
    y_sign: IndexedDict
        Custom dictionary in form key: np.array(List[str]), where np.arrays can be used to slice it.
    """

    def convert_array(array):
        if array.size == 0:
            raise ValueError(f'Received an empty array')
        elif len(array.shape) == 1:
            return array
        elif array.shape[1] == 1:
            return array.flatten()
        else:
            raise ValueError(f'Unrecognized array shape: {array.shape}')

    y_true = convert_array(y_true)
    y_pred = convert_array(y_pred)
    y_score = convert_array(y_score)
    y_wgts = convert_array(y_wgts)
    y_sign = IndexedDict({key: convert_array(item) for key, item in y_sign.items()})

    # Calculate Overall metrics
    total_metrics = inner_score(y_true=y_true, y_pred=y_pred, y_score=y_score, y_wgts=y_wgts)

    # Calculate per-sign metrics
    per_sign_metrics = defaultdict(dict)  # i.e. sign_type: task: sign_name: metrics | horrible
    for sign_type, sign_values in y_sign.items():
        unique_signs = set(sign_values)
        for sign_name in unique_signs:
            key = (sign_type, sign_name)
            sign_idx = np.where(sign_values == sign_name)[0]
            if len(sign_idx) > 0:
                per_sign_metrics[key] = inner_score(
                    y_true=y_true[sign_idx],
                    y_pred=y_pred[sign_idx],
                    y_score=y_score[sign_idx],
                    y_wgts=y_wgts[sign_idx]
                )
            else:
                per_sign_metrics[key] = {}

    return {
        'Total': total_metrics,
        'Sign': per_sign_metrics
    }


def fold_evaluate(model, dataset_manager: DatasetManager, transformer: DataTransformer, iter_idx: int) -> IterEnsemble:
    """
    Evaluate model using pre-defined folds.

    Parameters
    ----------
    model
        Instance of a model class
    dataset_manager: DatasetManager
        Instance of DatasetManager
    transformer: DataTransformer
        Instance of DataTransformer
    iter_idx: int
        Number of iteration. For logging purposes.
    """
    units = []

    for fold in dataset_manager.folds:

        data = dataset_manager.get_train_eval_data(fold)
        unit = FoldUnit(model=deepcopy(model), fold_idx=deepcopy(fold), transformer=deepcopy(transformer))

        x_train, x_eval = data['Train']['X'], data['Eval']['X']
        y_train, y_eval = data['Train']['y'], data['Eval']['y']
        x_demo_train, x_demo_eval = data['Train']['demo'], data['Eval']['demo']
        y_wgts_train, y_wgts_eval = data['Train']['wgts'], data['Eval']['wgts']
        y_sign_train, y_sign_eval = data['Train']['sign'], data['Eval']['sign'] # these ARE dict[str, np.ndarray]

        unit.fit(x_train=x_train, x_demo=x_demo_train, y_array=y_train, y_wgts=y_wgts_train)

        y_pred_train = unit.predict(x_train, x_demo_train)
        y_pred_eval = unit.predict(x_eval, x_demo_eval)

        y_score_train = unit.predict_proba(x_train, x_demo_train)
        y_score_eval = unit.predict_proba(x_eval, x_demo_eval)

        unit.scores[('Train', 'Unweighted')] = outer_score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train,
                                                           y_wgts=np.ones_like(y_pred_train), y_sign=y_sign_train)
        unit.scores[('Train', 'Weighted')] = outer_score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train,
                                                         y_wgts=y_wgts_train, y_sign=y_sign_train)

        unit.scores[('Eval', 'Unweighted')] = outer_score(y_true=y_eval, y_pred=y_pred_eval, y_score=y_score_eval,
                                                          y_wgts=np.ones_like(y_pred_eval), y_sign=y_sign_eval)
        unit.scores[('Eval', 'Weighted')] = outer_score(y_true=y_eval, y_pred=y_pred_eval, y_score=y_score_eval,
                                                        y_wgts=y_wgts_eval, y_sign=y_sign_eval)

        units.append(unit)

    return IterEnsemble(units, iter_idx=iter_idx)


def ensemble_evaluate(ensemble: IterEnsemble, test_df: pd.DataFrame, desc_col: str,
                      demo_col: str = 'DemoFP', label_col: str = 'Label', weight_col: str = 'Label_weight',
                      sign_col: str = 'Signature', sign_names: List[str] = None):
    """
    Evaluate the trained ensemble on a test set.

    Parameters
    ----------
    ensemble: IterEnsemble
        A trained ensemble of models
    test_df: pd.DataFrame
        Test set dataframe
    desc_col: str
        A name of the column holding descriptors
    demo_col: str
        A name of the column holding demographics vectors
    label_col: str
        A name of the column holding labels
    weight_col: str
        A name of the column holding label weights
    sign_col: str
        A name of the column holding "signatures"
    sign_names: List[str]
        A list of categories for signatures. ['Sex', 'Age', 'Weight']

    Returns
    -------
    pred_df: pd.DataFrame
        A Dataframe with predictions
    test_scores: Dict[str, Dict]
        A dictionary with test set scores

    """

    x_array = np.vstack(test_df[desc_col].to_numpy())
    x_demo = np.vstack(test_df[demo_col].to_numpy())

    y_true = np.vstack(test_df[label_col].to_numpy()).reshape(-1)
    y_wgts = np.vstack(test_df[weight_col].to_numpy()).reshape(-1)

    y_sign = IndexedDict(
        {key: np.array(values) for key, values in zip(sign_names, list(zip(*test_df[sign_col].tolist())))})

    y_score = ensemble.predict_proba(x_array, x_demo)
    y_pred = (y_score >= 0.5).astype(int)

    pred_df = pd.DataFrame({
        'SMILES': test_df['SMILES'],
        'Signature': test_df[sign_col],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score,
    })

    test_scores = {
        'Unweighted': outer_score(y_true=y_true, y_pred=y_pred, y_score=y_score,
                                  y_wgts=np.ones_like(y_pred), y_sign=y_sign),
        'Weighted': outer_score(y_true=y_true, y_pred=y_pred, y_score=y_score,
                                y_wgts=y_wgts, y_sign=y_sign)
    }
    return pred_df, test_scores


def optuna_hyperparameter_search(model_class, dataset_manager: DatasetManager, transformer: DataTransformer,
                                 test_fold: int, selection_metric: str = 'HarmRS', n_trials: int = 32,
                                 n_jobs: int = 1, save_dir: str = None):
    """
    Perform optuna hyperparameter search for a given model.

    Parameters
    ----------
    model_class
        An instance of a model class. Supported are LogisticRegression, RandomForestClassifier, SVC, XGBClassifier
    dataset_manager: DatasetManager
        Instance of DatasetManager class with one of the dataset variants
    transformer: DataTransformer
        Instance of DataTransformer class
    test_fold: int
        Number of fold used for testing
    selection_metric: str
        Name of an evaluation metric used to guide the optimization
    n_trials: int
        Number of Optuna trials
    n_jobs: int
        Number of jobs used. Passed directly to model's.
    save_dir: str
        Directory for saving the visualise

    Returns
    -------
    ensemble: IterEnsemble
        A trained ensemble of models
    """

    print('Beginning optuna optimization search')

    if save_dir is not None:
        save_dir = save_dir.rstrip('/') + '/'

    search_params = {
        'XGBClassifier': get_params_xgb_classifier,
        'RandomForestClassifier': get_params_random_forest_classifier,
        'SVC': get_params_svc,
        'LogisticRegression': get_params_logistic_regression
    }

    if selection_metric in ['Recall', 'Accuracy', 'ROC AUC', 'Precision', 'F1 Score',
                            'MCC', 'R2', 'Balanced Accuracy', 'Specificity', 'GeomRS', 'HarmRS']:
        direction = 'maximize'
        default = -1
    else:
        direction = 'minimize'
        default = 1

    iters = []

    def objective(trial: optuna.Trial):

        model_name = model_class.__name__
        model_params_fn = search_params.get(model_name)

        if model_params_fn is None:
            raise ValueError(f'No hyperparameters defined for: {model_name}')

        try:
            model_params = model_params_fn(trial)
            model_params['n_jobs'] = n_jobs
            model = model_class(**model_params)
        except (ValueError, TypeError) as e:
            print(f"Trial {trial.number} pruned due to invalid parameters: {e}")
            raise optuna.TrialPruned() from e

        try:
            iter_ensemble = fold_evaluate(
                model=model,
                dataset_manager=dataset_manager,
                transformer=transformer,
                iter_idx=trial.number
            )

        except Exception as e:
            print(f"Trial {trial.number} pruned with params: {model_params} due to\n{e}")
            raise optuna.TrialPruned() from e

        trial_mean, trial_std = iter_ensemble.summary.get('AvTotal', {}).get(selection_metric, (default, 0.0))

        iter_ensemble.optuna_score = trial_mean
        iter_ensemble.hyperparameters = model_params

        iters.append(iter_ensemble)

        return trial_mean

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    tune_army = TuneArmy(iters)
    ensemble = tune_army.best()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        study_path = os.path.join(save_dir, f'study_tf_{test_fold}.joblib')
        ensemble_path = os.path.join(save_dir, f'ensemble_tf_{test_fold}.joblib')

        joblib.dump(study, study_path)
        joblib.dump(ensemble, ensemble_path)

    print('Optuna search finished successfully')

    return ensemble


def main(args):

    print('Job starting', flush=True)

    input_dir = args.input_dir
    output_dir = args.output_dir
    dataset_type = args.dataset_type
    pt_set = args.pt_set
    dpa_metric = args.dpa_metric
    model_name = args.model_name
    desc_col = args.desc_col
    sel_metric = args.sel_metric
    n_trials = int(args.n_trials)
    n_jobs = int(args.n_jobs)
    test_fold = int(args.test_fold)
    sign_names = ['Sex', 'Age', 'Weight']

    print('Beginning checks')

    save_dir = str(os.path.join(output_dir, f'{dataset_type}_{pt_set}_{dpa_metric}', model_name, desc_col))
    os.makedirs(save_dir, exist_ok=True)

    # Build paths
    data_path = os.path.join(input_dir, dataset_type, pt_set, f'carbide_{dpa_metric}.joblib')
    desc_path = os.path.join(input_dir, 'descriptors.joblib')
    pred_path = os.path.join(save_dir, f'preds_tf_{test_fold}.joblib')
    scores_path = os.path.join(save_dir, f'scores_tf_{test_fold}.joblib')

    if not os.path.exists(data_path):
        raise OSError(f"CARBIDE dataset file at < {data_path} > does not exist")

    if not os.path.exists(desc_path):
        raise OSError(f"Descriptors file at < {desc_path} > does not exist")

    # Load data
    data = joblib.load(data_path)
    desc = joblib.load(desc_path)

    if "SMILES" not in data.columns:
        raise KeyError(f"SMILES column not found in CARBIDE dataset file")

    if "SMILES" not in desc.columns:
        raise KeyError(f"SMILES column not found in descriptors file")

    if desc_col not in desc.columns:
        raise KeyError(f"Descriptor column < {desc_col} > not found in descriptors file")

    if model_name not in ['XGBClassifier', 'LogisticRegression', 'RandomForestClassifier']:
        raise ValueError(f"Model < {model_name} > not supported")

    if sel_metric not in ["TP", "FP", "FN", "TN", "Accuracy", "Recall", "Specificity", "Precision", "Balanced Accuracy",
                          "GeomRS", "HarmRS", "F1 Score", "ROC AUC", "MCC"]:
        raise ValueError(f"Metric < {sel_metric} > not supported")

    if not isinstance(n_trials, int) or n_trials < 1:
        raise TypeError("Number of trials must be a positive integer")

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise TypeError("Number of jobs must be a positive integer")

    if test_fold not in data['Fold'].unique():
        raise ValueError(f"Test fold < {test_fold} > not found in CARBIDE dataset")

    print('All checks passed')
    print('Processing the CARBIDE dataset')

    # Combine data with selected descriptor types
    df = data.merge(desc[["SMILES", desc_col]], on='SMILES', how='inner')

    # Split data into ml and test
    train_df = df[df['Fold'] != test_fold].reset_index(drop=True)
    test_df = df[df['Fold'] == test_fold].reset_index(drop=True)

    print('Preparing DatasetManager')

    # Prepare training data
    train_manager = DatasetManager(
        df=train_df,
        desc_col=desc_col,
        sign_names=sign_names
    )

    print('Preparing DataTransformer')

    # Prepare DataTransformer for given descriptors
    transformer = prepare_transformer(desc_col)

    print('Performing Optuna HP optimization')

    # Perform hyperparameter optimization of ml set
    ensemble = optuna_hyperparameter_search(
        model_class=get_model(model_name),
        dataset_manager=train_manager,
        transformer=transformer,
        test_fold=test_fold,
        selection_metric=sel_metric,  # see src/training.py:inner_score for allowed metrics
        n_trials=n_trials,
        n_jobs=n_jobs,
        save_dir=save_dir,
    )

    print('Evaluating ensemble')

    # Evaluate the trained ensemble on the test set
    pred_df, test_scores = ensemble_evaluate(
        ensemble=ensemble,
        test_df=test_df,
        desc_col=desc_col,
        sign_names=sign_names
    )

    print('Saving visualise')

    # Save visualise
    joblib.dump(pred_df, pred_path)
    joblib.dump(test_scores, scores_path)

    print('Job finished')

if __name__ == '__main__':
    main(args)