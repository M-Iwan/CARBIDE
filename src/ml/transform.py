from math import floor

import polars as pl

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


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
