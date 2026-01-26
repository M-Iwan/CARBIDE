"""
Module with functions for Optuna-based HP optimization
"""

import os
import pickle
import joblib
from typing import Dict, Union

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import optuna

from src.ml.manager import DatasetManager
from src.ml.transform import DataTransformer
from src.ml.evaluate import fold_evaluate
from src.ml.models import IterEnsemble, TuneArmy


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


def get_params_knn_classifier(trial) -> Dict:
    """
    Suggest values for KNeighborsClassifier for optuna
    """
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 25),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 2)
    }

    return params


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
