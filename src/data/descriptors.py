"""
Module with functions for calculating molecular descriptors. ChemBERTa and CDDD descriptors require additional
packages and/or environments
"""

import json
import os
import pickle
from typing import Union, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFingerprintGenerator

from src.data.io import read_df

# Optional imports for ChemBERTa and CDDD descriptors
try:
    from transformers import AutoTokenizer, AutoModel, logging
    import torch
    HAS_TRANSFORMERS = True
    HAS_TORCH = True
except ImportError:
    HAS_TRANSFORMERS = False
    HAS_TORCH = False


def smiles_2_morgan(smiles: Union[str, List[str]], radius: int = 2, nbits: int = 1024,
                    dense: bool = True, mfpgen=None):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    radius: int, optional
        The radius parameter for Morgan FP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is .
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    mfpgen: rdkit.Chem.rdFingerprintGenerator.FingeprintGenerator64, optional
        If passed, the radius and nbits will be ignored.

    Returns
    -------

    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    if mfpgen is None:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    def get_embedding(smi):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array(mfpgen.GetFingerprint(mol), dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_morgan(df: pd.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'Morgan',
                       radius: int = 2, nbits: int = 1024, dense: bool = True):
    """
    Convert SMILES in dataframe to Morgan fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    radius: int, optional
        The radius parameter for Morgan FP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding Morgan fingerprints for given SMILES.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    df[descriptor_col] = smiles_2_morgan(df[smiles_col].tolist(), radius=radius, nbits=nbits, dense=dense, mfpgen=mfpgen)

    return df


def smiles_2_maccs(smiles: Union[str, List[str]], dense: bool = True):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """
    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol), dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_maccs(df: pd.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'MACCS',
                      dense: bool = True):
    """
    Convert SMILES in dataframe to MACCS fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding MACCS fingerprints for given SMILES.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """

    df[descriptor_col] = smiles_2_maccs(df[smiles_col].tolist(), dense=dense)

    return df


def smiles_2_klek(smiles: Union[str, List[str]], dense: bool = True, klek_mols: List = None,
                  source: str = 'src/src_files/klek.pkl'):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    klek_mols: List
        A list of RDKit molecules from Klekota&Roth SMARTS definitions.
        If not passed, they are read from source parameter.
    source: str
        A path to pickled List of klek_mols

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """

    if klek_mols is None:
        klek_mols = pickle.load(open(source, 'rb'))

    def get_embedding(smi: str, kmols: List):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.array([1 if mol.HasSubstructMatch(km) else 0 for km in kmols], dtype=np.uint8)

            return fp if dense else fp.nonzero()[0]

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles, klek_mols)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi, klek_mols) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_klek(df: pd.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'Klek',
                     dense: bool = True, source: str = 'src/src_files/klek.pkl'):
    """
    Calculate Klekota&Roth descriptors based on SMILES in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    dense: bool, optional
        If True, return a dense representation of FP. Default is True.
    source: str
        A path to pickled List of RDKit molecules generated from Klekota&Roth SMARTS.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding Morgan fingerprints for given SMILES.
        If 'dense' is set to False, returns a numpy array with indices of non-zero elements.
    """
    klek_mols = pickle.load(open(source, 'rb'))
    df[descriptor_col] = smiles_2_klek(df[smiles_col].tolist(), dense=dense, klek_mols=klek_mols)

    return df


def smiles_2_rdkit(smiles: Union[str, List[str]], decimals: int = 5):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """
    def get_embedding(smi: str):
        try:
            if (mol := Chem.MolFromSmiles(smi)) is None:
                print(f'Unable to construct a valid molecule from < {smi} >')
                return np.nan

            fp = np.round(np.fromiter(Descriptors.CalcMolDescriptors(mol, silent=False, missingVal=np.nan).values(), dtype=np.float64), decimals)
            return fp

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embedding(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embedding(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_rdkit(df: pd.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'RDKit',
                      decimals: int = 5):
    """
    Convert SMILES in dataframe to RDKit descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding RDKit descriptors for given SMILES.
    """

    df[descriptor_col] = smiles_2_rdkit(df[smiles_col].tolist(), decimals=decimals)

    return df


def smiles_2_chemberta(smiles: Union[str, List[str]], decimals: int = 5):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    array: Union[np.ndarray, List[np.ndarray]]
        A np.ndarray or list of np.ndarrays.
    """
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        print('This function requires Torch and Transformers to be installed.')
        return np.nan

    logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM")
    model = AutoModel.from_pretrained("DeepChem/ChemBERTa-100M-MLM")

    model.eval()

    def get_embeddings(smi: str):

        try:
            tokens = tokenizer(smi, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                output = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
            return np.round(output, decimals)

        except Exception as e:
            print(f'Unable to process < {smi} > due to: \n{e}')
            return np.nan

    if isinstance(smiles, str):
        return get_embeddings(smiles)

    if isinstance(smiles, list) & all(isinstance(smi, str) for smi in smiles):
        return [get_embeddings(smi) for smi in smiles]

    print(f'Expected < smiles > to be str or list, received < {type(smiles)} > instead')
    return np.nan


def dataframe_2_chemberta(df: pd.DataFrame, smiles_col: str = 'SMILES', desc_col: str = 'ChemBERTa',
                          decimals: int = 5):
    """
    Calculate ChemBERTa embeddings based on SMILES in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str
        Name of column with SMILES.
    desc_col : str
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding ChemBERTa embeddings.
    """
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        print('This function requires Torch and Transformers to be installed.')
        df[desc_col] = [np.nan] * len(df)
        return df

    df[desc_col] = smiles_2_chemberta(df[smiles_col].tolist(), decimals=decimals)

    return df


def dataframe_2_cddd(df: pd.DataFrame, smiles_col: str = 'SMILES', desc_col: str = 'CDDD',
                     path_source: str = f'src/src_files/cddd_paths.json', n_cpus: int = 4, decimals: int = 5):
    """
    Convert SMILES in dataframe CDDD descriptors.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with data.
    smiles_col : str
        Name of column with SMILES.
    desc_col : str
        Name of column to which add calculated descriptors.
    path_source : str
        Path to cddd_paths.json file.
    n_cpus : int
        Number of CPUs to use during processing
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : pd.DataFrame
        A pandas Dataframe with added column holding CDDD descriptors for given SMILES.
    """

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]

    with open(path_source, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {desc_col} --n_cpu {n_cpus} --model_dir {paths['model']}")

    # pack the lists to strings so that they don't get broken
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    df.to_csv(paths['input'], sep='\t', header=True, index=False)

    os.system(command)

    df = read_df(paths['output'])
    df.loc[:, desc_col] = df[desc_col].apply(postprocess, decimals_=decimals)

    return df
