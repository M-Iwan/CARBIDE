from collections import defaultdict
from typing import ClassVar, Dict, List

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.special import expit


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