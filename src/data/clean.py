"""
File with functions related to dataset preparation and filtering.
"""
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover, RemoveStereochemistry
from rdkit.Chem.MolStandardize import rdMolStandardize


def remove_minor(df: pd.DataFrame, smiles_column: str = 'SMILES', out_column: str = 'Fragment'):
    """
    Remove small fragments from SMILES, keeping only the biggest one

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_column : str
        Name of the output column
    """
    RDLogger.DisableLog('rdApp.*')

    fragment_remover = rdMolStandardize.LargestFragmentChooser()

    def process_smiles(smiles, remover):

        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead.')

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan

            mol = remover.choose(mol)
            return Chem.MolToSmiles(mol)

        except Exception as e:
            print(f'Could not find the biggest fragment in <{smiles}> due to {e}')
            return np.nan

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles, remover=fragment_remover)

    return df


def remove_metals(df: pd.DataFrame, smiles_column: str = 'SMILES', out_column: str = 'Metal'):
    """
    Remove metals from molecules

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_column : str
        Name of the output column
    """
    RDLogger.DisableLog('rdApp.*')

    metal_remover = rdMolStandardize.MetalDisconnector()

    def process_smiles(smiles, remover):
        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead')
            return np.nan
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan

            mol = remover.Disconnect(mol)
            mol = rdMolStandardize.DisconnectOrganometallics(mol)
            return Chem.MolToSmiles(mol)

        except Exception as e:
            print(f'Could not remove metals from <{smiles}> due to {e}')
            return np.nan

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles, remover=metal_remover)

    return df


def remove_salts(df: pd.DataFrame, smiles_column: str = 'SMILES', out_columns: List = None):
    """
    Remove salts from molecules

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_columns : List[str, str]
        Names of the columns for the output. First for stripped SMILES, second for removed fragment
    """
    RDLogger.DisableLog('rdApp.*')

    salt_remover = SaltRemover.SaltRemover()

    out_columns = ['Salt', 'Removed'] if out_columns is None else out_columns

    def process_smiles(smiles, remover):
        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead')
            return np.nan, np.nan

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan, np.nan

            new_smiles, deleted_list = remover.StripMolWithDeleted(mol, dontRemoveEverything=True)
            # changed due to some compounds being removed
            return Chem.MolToSmiles(new_smiles), [Chem.MolToSmiles(deleted) for deleted in deleted_list]

        except Exception as e:
            print(f'Could not remove salts from < {smiles} > due to {e}')
            return np.nan, np.nan

    out = df[smiles_column].apply(process_smiles, remover=salt_remover)

    df.loc[:, out_columns[0]] = [result[0] for result in out]
    df.loc[:, out_columns[1]] = [result[1] for result in out]

    return df


def remove_stereochemistry(df: pd.DataFrame, smiles_column: str = 'SMILES', out_column: str = 'Stereo'):
    """
    Remove stereochemistry from molecules

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_column : str
        Name of the output column
    """

    RDLogger.DisableLog('rdApp.*')

    def process_smiles(smiles):
        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead')
            return np.nan

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan

            RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol)

        except Exception as e:
            print(f'Could not remove stereochemistry from < {smiles} > due to {e}')
            return np.nan

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles)
    return df


def remove_charges(df: pd.DataFrame, smiles_column: str = 'SMILES', out_column: str = 'Charge'):
    """
    Attempt to neutralize charges from molecules

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILES
    out_column : str
        Name of the output column
    """

    RDLogger.DisableLog('rdApp.*')

    charge_remover = rdMolStandardize.Uncharger()

    def process_smiles(smiles, remover):
        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead')
            return np.nan

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan

            mol = remover.uncharge(mol)
            return Chem.MolToSmiles(mol)

        except Exception as e:
            print(f'Could not remove charges from <{smiles}> due to {e}')
            return np.nan

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles, remover=charge_remover)
    return df


def remove_tautomers(df: pd.DataFrame, smiles_column: str = 'SMILES', max_num_tautomers: int = 10,
                     out_column: str = 'Tautomer'):
    """
    Attempt to return canonical tautomer from a molecule

    Parameters
    --------------
    df : pd.DataFrame
        pd.DataFrame with SMILES
    smiles_column : str
        Name of a column holding SMILE
    max_num_tautomers : int
        Number of tautomers to create while evaluating the canonical
    out_column : str
        Name of the output column
    """

    RDLogger.DisableLog('rdApp.*')

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_num_tautomers)

    def process_smiles(smiles, enum):
        if not isinstance(smiles, str):
            print(f'Expected SMILES to be of type str, got {type(smiles)} instead')
            return np.nan

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f'Could not convert < {smiles} > to a valid molecule')
                return np.nan

            mol = enum.Canonicalize(mol)
            return Chem.MolToSmiles(mol)

        except Exception as e:
            print(f'Could not find canonical tautomer from <{smiles}> due to {e}')
            return np.nan

    df.loc[:, out_column] = df[smiles_column].apply(process_smiles, enum=enumerator)

    return df


def pipeline_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a chemical compound DataFrame through a standardization pipeline.

    This function applies a series of cleaning operations to chemical structures
    including parent compound selection, metal disconnection, salt stripping,
    stereochemistry removal, charge neutralization, and tautomer canonicalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing chemical structures. Must include a column
        that can be processed as SMILES strings.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with standardized chemical structures.

    Notes
    -----
    The pipeline performs the following operations in order:
    1. Selects parent compounds
    2. Disconnects metals from structures
    3. Removes salt counterions
    4. Removes stereochemistry information (flattens compounds)
    5. Neutralizes formal charges
    6. Selects canonical tautomer (limited to 16 maximum tautomers)
    """
    print('> Selecting parent compounds')
    df = remove_minor(df, out_column='CanSMILES')
    print('> Disconnecting metals')
    df = remove_metals(df, smiles_column='CanSMILES', out_column='CanSMILES')
    print('> Stripping salts')
    df = remove_salts(df, smiles_column='CanSMILES', out_columns=['CanSMILES', 'Removed_Salts'])
    print('> Flattening compounds')
    df = remove_stereochemistry(df, smiles_column='CanSMILES', out_column='CanSMILES')
    print('> Neutralizing charges')
    df = remove_charges(df, smiles_column='CanSMILES', out_column='CanSMILES')
    print('> Selecting canonical tautomer')
    df = remove_tautomers(df, smiles_column='CanSMILES', out_column='CanSMILES', max_num_tautomers=64)
    return df