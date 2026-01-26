"""
Utility functions
"""

import numpy as np
from rdkit import DataStructs


def is_valid_fingerprint(fingerprint: np.ndarray) -> bool:
    """
    Check if the passed numpy array contains only binary values.
    """
    return np.all(np.isin(fingerprint, [0, 1]))


def ndarray_to_binary_string(array: np.ndarray) -> str:
    """
    Convert numpy array to binary string.
    """
    if not len(array) or not is_valid_fingerprint(array):
        raise ValueError(
            "Invalid fingerprint array. Expected binary Morgan fingerprint with 0s and 1s."
        )
    return "".join(array.astype(str).tolist())


def embeddings_to_rdkit(embeddings: np.ndarray) -> list:
    """
    Convert numpy arrays to native RDKit format.
    """

    fingerprints = []
    for emb in embeddings:
        fingerprint = ndarray_to_binary_string(emb)
        fingerprints.append(DataStructs.CreateFromBitString(fingerprint))
    return fingerprints


def round_to_significant(x: float, n: int):
    """
    Round a value to a given number of significant digits.
    """
    if isinstance(x, str):
        return(x)
    if x is None:
        return 'None'

    if x == 0:
        return 0
    else:
        return round(x, n - int(np.floor(np.log10(abs(x)))) - 1)


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


def is_valid_float(value):
    """
    Check if a float holds an actual value
    """
    return isinstance(value, float) and not np.isnan(value)