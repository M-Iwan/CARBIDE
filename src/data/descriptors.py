import os
import math
import json
import joblib
from joblib import Parallel, delayed
from importlib.resources import files
from typing import Union, List
from itertools import chain

import numpy as np
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFingerprintGenerator

import torch


def smiles_2_ecfp(smiles: Union[str, List[str], np.ndarray[str]], radius: int = 2, nbits: int = 1024, count: bool = False):
    """
    Convert SMILES string(s) to a (Count) Extended Connectivity Fingerprint.

    Parameters
    ----------
    smiles: Union[str, List[str], np.ndarray[str]]
        A SMILES string or a list of SMILES strings.
    radius: int, optional
        The radius parameter for ECFP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of ECFP. Default is False.

    Returns
    -------
    fp: np.ndarray
    OR
    fps: List[np.ndarray]
    """

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        if count:
            fp = gen.GetFingerprintAsNumPy(mol)
        else:
            fp = gen.GetCountFingerprintAsNumPy(mol)

        return fp

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        if count:
            fps = [np.array(fp.ToList(), dtype=np.uint16) for fp in gen.GetCountFingerprints(mols)]  # returns UIntSparseBitVector

        else:
            fps = [np.array(fp, dtype=np.uint8) for fp in gen.GetFingerprints(mols)]  # returns ExplicitBitVector

        return fps

    else:
        raise TypeError(f"Expected smiles to be str or List[str], got {type(smiles)} instead")


def dataframe_2_ecfp(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = None, radius: int = 2,
                     nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES string(s) in a DataFrame to a Extended Connectivity (Count) Fingerprint.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    radius: int, optional
        The radius parameter for ECFP calculation. Default is 2.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of ECFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added ECFP/ECFPCount column.
    """

    if descriptor_col is None:
        if count:
            descriptor_col = 'ECFPCount'
        else:
            descriptor_col = 'ECFP'

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    fps = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_ecfp)(smiles=smi, radius=radius, nbits=nbits, count=count) for smi in smiles_batches
    )

    fps = chain.from_iterable(fps)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: fps
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_daylight(smiles: Union[str, List[str], np.ndarray[str]], min_path: int = 1, max_path: int = 7,
                      nbits: int = 1024, count: bool = False):
    """
    Convert SMILES in a DataFrame to Daylight (Count) Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], np.ndarray[str]]
        A SMILES string or a list of SMILES strings.
    min_path: int
        Smallest path length to consider. Default is 1.
    max_path: int
        Biggest path length to consider. Default is 7.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of DaylightFP. Default is False.

    Returns
    -------
    fp: np.ndarray
    OR
    fps: List[np.ndarray]
    """

    gen = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=min_path, maxPath=max_path, fpSize=nbits)

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        if count:
            fp = gen.GetFingerprintAsNumPy(mol)
        else:
            fp = gen.GetCountFingerprintAsNumPy(mol)

        return fp

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        if count:
            fps = [np.array(fp.ToList(), dtype=np.uint16) for fp in gen.GetCountFingerprints(mols)]  # returns UIntSparseBitVector

        else:
            fps = [np.array(fp, dtype=np.uint8) for fp in gen.GetFingerprints(mols)]  # returns ExplicitBitVector

        return fps

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_daylight(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = None, min_path: int = 1,
                         max_path: int = 7, nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to Daylight (Count) Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    min_path: int
        Smallest path length to consider. Default is 1.
    max_path: int
        Biggest path length to consider. Default is 7.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of DaylightFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Daylight/DaylightCount column.
    """

    if descriptor_col is None:
        if count:
            descriptor_col = 'DaylightCount'
        else:
            descriptor_col = 'Daylight'

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    fps = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_daylight)(smiles=smi, min_path=min_path, max_path=max_path, nbits=nbits, count=count) for smi in smiles_batches
    )

    fps = chain.from_iterable(fps)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: fps
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_atompair(smiles: Union[str, List[str], np.ndarray[str]], min_distance: int = 1, max_distance: int = 7,
                      nbits: int = 1024, count: bool = False):
    """
    Convert SMILES in to AtomPair (Count) Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str], np.ndarray[str]]
        A SMILES string or a list of SMILES strings.
    min_distance: int
        Smallest distance between two atoms to consider. Default is 1.
    max_distance: int
        Largest distance between two atoms to consider. Default is 30.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of AtomPairFP. Default is False.

    Returns
    -------
    fp: np.ndarray
    OR
    fps: List[np.ndarray]
    """

    gen = rdFingerprintGenerator.GetAtomPairGenerator(minDistance=min_distance, maxDistance=max_distance, fpSize=nbits)

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        if count:
            fp = gen.GetFingerprintAsNumPy(mol)
        else:
            fp = gen.GetCountFingerprintAsNumPy(mol)

        return fp

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        if count:
            fps = [np.array(fp.ToList(), dtype=np.uint16) for fp in gen.GetCountFingerprints(mols)]  # returns UIntSparseBitVector

        else:
            fps = [np.array(fp, dtype=np.uint8) for fp in gen.GetFingerprints(mols)]  # returns ExplicitBitVector

        return fps

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_atompair(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = None, min_distance: int = 1,
                         max_distance: int = 7, nbits: int = 1024, count: bool = False, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to AtomPair (Count) Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    min_distance: int
        Smallest distance between two atoms to consider. Default is 1.
    max_distance: int
        Largest distance between two atoms to consider. Default is 30.
    nbits: int, optional
        The length of the FP. Default is 1024.
    count: bool
        If True, return a count version of AtomPairFP. Default is False.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added AtomPair/AtomPairCount column.
    """

    if descriptor_col is None:
        if count:
            descriptor_col = 'AtomPairCount'
        else:
            descriptor_col = 'AtomPair'

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    fps = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_atompair)(smiles=smi, min_distance=min_distance, max_distance=max_distance, nbits=nbits, count=count)
        for smi in smiles_batches
    )

    fps = chain.from_iterable(fps)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: fps
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_maccs(smiles: Union[str, List[str], np.ndarray[str]]):
    """
    Convert SMILES to MACCS Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str]]
        A SMILES or list of SMILES strings.

    Returns
    -------
    fp: np.ndarray
    OR
    fps: List[np.ndarray]
    """

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        fp = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol), dtype=np.uint8)
        return fp

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        fps = [np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol), dtype=np.uint8) for mol in mols]
        return fps

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_maccs(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'MACCS',
                      n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to MACCS Fingerprints.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added MACCS fingerprints for given SMILES.
    """

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    fps = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_maccs)(smiles=smi) for smi in smiles_batches
    )

    fps = chain.from_iterable(fps)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: fps
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_klek(smiles: Union[str, List[str], np.ndarray[str]], klek_path: str = '../src/files/klek_mols.joblib'):
    """
    Convert SMILES to Klekota&Roth Fingerprints.

    Parameters
    ----------
    smiles: Union[str, List[str]]
        A SMILES or list of SMILES strings.
    klek_path: str, optional
        Path to file with encoded Klekota&Roth SMARTS

    Returns
    -------
    fp: np.ndarray
    OR
    fps: List[np.ndarray]
    """

    def get_fp(mol, smarts):
        fp = [1 if mol.HasSubstructMatch(sm) else 0 for sm in smarts]
        fp = np.array(fp, dtype=np.uint8)
        return fp

    klekota_smarts = joblib.load(klek_path)

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        return get_fp(mol=mol, smarts=klekota_smarts)

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        return [get_fp(mol=mol, smarts=klekota_smarts) for mol in mols]

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_klek(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'Klek',
                     n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to Klekota&Roth Fingerprints.

    Parameters
    ----------
    df: pl.DataFrame
        A polars DataFrame with data.
    smiles_col: str, optional
        Name of column with SMILES.
    descriptor_col: str, optional
        Name of column to which add calculated descriptors.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added Klek column.
    """

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    fps = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_klek)(smiles=smi) for smi in smiles_batches
    )

    fps = chain.from_iterable(fps)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: fps
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_rdkit(smiles: Union[str, List[str], np.ndarray[str]], decimals: int = 5):
    """
    Convert SMILES to RDKit descriptors.

    Parameters
    ----------
    smiles: Union[str, List[str], np.ndarray[str]]
        A SMILES or list of SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """

    def get_desc(mol, decimals: int):
        desc = Descriptors.CalcMolDescriptors(mol, silent=False, missingVal=np.nan).values()
        desc = np.round(np.fromiter(desc, dtype=np.float64), decimals)
        return desc

    if isinstance(smiles, str):
        if (mol := Chem.MolFromSmiles(smiles)) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        return get_desc(mol=mol, decimals=decimals)

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        return [get_desc(mol=mol, decimals=decimals) for mol in mols]

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_rdkit(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'RDKit',
                      decimals: int = 5, n_jobs: int = 1, batch_size: int = 512):
    """
    Convert SMILES in a DataFrame to RDKit descriptors.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame with data.
    smiles_col : str, optional
        Name of column with SMILES.
    descriptor_col : str, optional
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added RDKit column.
    """

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    desc = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_rdkit)(smiles=smi, decimals=decimals) for smi in smiles_batches
    )

    desc = chain.from_iterable(desc)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: desc
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def smiles_2_chemberta(smiles: Union[str, List[str], np.ndarray[str]], decimals: int = 5,):
    """
    Parameters
    ----------
    smiles: Union[str, List[str]]
        A valid SMILES or list of valid SMILES strings.
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
    """

    def get_emb(smiles, decimals: int):

        tokens = tokenizer(smiles, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            emb = model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()
        return np.round(emb, decimals)

    try:
        from transformers import AutoTokenizer, AutoModel, logging
    except ImportError:
        raise ImportError("Function < smiles_2_chemberta > requires < transformers > library. Please install it "
                          "with < pip install transformers >")

    logging.set_verbosity_error()

    tokenizer = joblib.load(files('CARBIDE.files').joinpath('ChemBERTa-tokenizer.joblib'))
    model = joblib.load(files('CARBIDE.files').joinpath('ChemBERTa-model.joblib'))
    model.eval()

    if isinstance(smiles, str):
        if Chem.MolFromSmiles(smiles) is None:
            print(f'Unable to construct a valid molecule from < {smiles} >')
            return np.nan

        try:
            return get_emb(smiles=smiles, decimals=decimals)
        except Exception as e:
            print(f'Unable to process < {smiles} > due to: \n{e}')
            return np.nan

    elif isinstance(smiles, list) or isinstance(smiles, np.ndarray):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        if any([mol is None for mol in mols]):
            print(f"At least one valid molecule cannot be constructed from provided SMILES")
            return [np.nan] * len(mols)

        try:
            return [get_emb(smiles=smi, decimals=decimals) for smi in smiles]
        except Exception as e:
            print(f'Unable to process at least one SMILES due to: \n{e}')
            return np.nan * len(mols)

    else:
        raise TypeError(f"Expected smiles to be str or Union[List[str], np.ndarray[str]], got {type(smiles)} instead")


def dataframe_2_chemberta(df: pl.DataFrame, smiles_col: str = 'SMILES', descriptor_col: str = 'ChemBERTa',
                          decimals: int = 5, n_jobs: int = 1, batch_size: int = 512, base_path: str = '../src/files'):
    """
    Convert SMILES in a polars DataFrame to ChemBERTa embeddings.

    Parameters
    ----------
    df : pl.DataFrame
        A polars DataFrame.
    smiles_col : str
        Name of column with SMILES.
    descriptor_col : str
        Name of column to which add calculated descriptors.
    decimals: int
        Number of decimals to keep.
    n_jobs: int, optional
        Number of cores to use for calculations.
    batch_size: int, optional
        Number of SMILES per batch.
    base_path: str, optional
        Base path to saved tokenizer and model.

    Returns
    -------
    df : pl.DataFrame
        A polars Dataframe with added ChemBERTa column.
    """

    try:
        from transformers import AutoTokenizer, AutoModel, logging
    except ImportError:
        raise ImportError("Function < dataframe_2_chemberta > requires < transformers > library. Please install it "
                          "with < pip install transformers >")

    if not os.path.isfile(str(os.path.join(base_path, 'ChemBERTa-tokenizer.joblib'))):
        raise FileNotFoundError(f"ChemBERTa-tokenizer.joblib not found. Please call get_chemberta first.")
    if not os.path.isfile(str(os.path.join(base_path, 'ChemBERTa-model.joblib'))):
        raise FileNotFoundError(f"ChemBERTa-model.joblib not found. Please call get_chemberta first.")

    smiles = list(set(df[smiles_col].to_list()))
    n_batches = math.ceil(len(smiles) / batch_size)
    smiles_batches = np.array_split(smiles, n_batches)

    embs = Parallel(n_jobs=n_jobs, verbose=1, timeout=60, backend='loky')(
        delayed(smiles_2_chemberta)(smiles=smi, decimals=decimals) for smi in smiles_batches
    )

    embs = chain.from_iterable(embs)

    smiles_df = pl.DataFrame({
        smiles_col: smiles,
        descriptor_col: embs
    })

    df = df.join(smiles_df, on=smiles_col, how='left')

    return df


def dataframe_2_cddd(df: Union[pd.DataFrame, pl.DataFrame], cddd_paths: str, smiles_col: str = 'SMILES',
                     descriptor_col: str = 'CDDD', n_cpus: int = 4, decimals: int = 5):
    """
    Convert SMILES in a DataFrame to CDDD descriptors.
    TODO: This already

    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars DataFrame with data.
    cddd_paths : str
        Path to a JSON file containing paths to CDDD-related files. Generated using CDDD_conf.sh
    smiles_col : str
        Name of column with SMILES.
    descriptor_col : str
        Name of column to which add calculated descriptors.
    n_cpus : int
        Number of CPUs to use during processing
    decimals: int
        Number of decimals to keep.

    Returns
    -------
    df : Union[pd.DataFrame, pl.DataFrame]
        A pandas/polars Dataframe with added column holding CDDD descriptors for given SMILES.
    """

    def postprocess(entry, decimals_):
        if not isinstance(entry, (np.ndarray, list)):
            return np.nan
        if isinstance(entry, np.ndarray):
            return np.round(entry.reshape(-1), decimals_)
        if isinstance(entry, list):
            return [np.round(array.reshape(-1), decimals_) for array in entry]
        else:
            raise TypeError(f"Expected numpy array or list, got {type(entry)} instead.")

    with open(cddd_paths, 'r') as file:
        paths = json.load(file)

    command = (f"{paths['python']} {paths['wrapper']} --input {paths['input']} --output {paths['output']} "
               f"--smiles_col {smiles_col} --descriptor_col {descriptor_col} --n_cpu {n_cpus} --model_dir {paths['model']}")

    if is_polars := isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # pack the lists to strings so that they don't get broken;
    df.loc[:, smiles_col] = df[smiles_col].apply(lambda entry: ' : '.join(entry) if isinstance(entry, list) else entry)
    df.to_csv(paths['input'], sep='\t', header=True, index=False)

    os.system(command)

    df = joblib.load(paths['output'])
    df.loc[:, descriptor_col] = df[descriptor_col].apply(postprocess, decimals_=decimals)

    if is_polars:
        df = pl.from_pandas(df)

    return df
