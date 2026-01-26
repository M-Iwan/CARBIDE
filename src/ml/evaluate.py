"""
Module for scoring and evaluation of models
"""

import joblib
from collections import defaultdict
from typing import Optional, Dict, List
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data.utils import IndexedDict
from src.ml.manager import DatasetManager
from src.ml.transform import DataTransformer
from src.ml.models import FoldUnit, IterEnsemble

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
