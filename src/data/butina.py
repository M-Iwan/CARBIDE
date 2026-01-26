from typing import List, Union

import numpy as np
import pandas as pd
from rdkit import DataStructs
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedGroupKFold

from src.data.utils import embeddings_to_rdkit


def butina_cluster(df: pd.DataFrame, fp_col: str = 'Morgan', threshold: float = 0.3,
                   n_jobs: int = -2, batch_size: int = 256) -> pd.DataFrame:
    """
    Performs Butina Clustering using RDKit. Parallel computing and batch processing is supported.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame with fingerprint column.
    fp_col: str
        Name of the column with fingerprints. Default is Morgan
    threshold: float
        Distance threshold. Pair of FPs is marked as neighbors if distance <= threshold. Default is 0.3
    n_jobs: int
        Number of cores to use. Default is -2 (i.e. use all but one cores)
    batch_size: int
        Number of rows allocated to each process during neighborhood matrix calculation.

    Returns
    -------
    df: pd.DataFrame
        Pandas DataFrame with assigned neighbors and cluster_id columns.

    Notes
    -----
    Suggested starting thresholds:
    - between 0.3 and 0.4
    """
    df = df.reset_index(drop=True)  # Needed for correct indices

    n_samples = df.shape[0]
    seen = np.zeros(n_samples, dtype=bool)
    cluster_ids = np.full(n_samples, -1, dtype=int)

    bit_vectors = embeddings_to_rdkit(np.vstack(df[fp_col].to_numpy()))

    batch_indices = list(range(0, n_samples - 1, batch_size)) + [n_samples - 1]

    def compute_neighbors_batch(start_idx, end_idx, _bit_vectors, _threshold):

        similarities = []

        for i in range(start_idx, end_idx):
            row_similarities = DataStructs.BulkTanimotoSimilarity(_bit_vectors[i], _bit_vectors[i + 1:])
            similarities.append((1 - np.array(row_similarities)) <= _threshold)

        return similarities

    # Compute neighbors in batches
    neighbor_batches = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(compute_neighbors_batch)(batch_indices[i], min(batch_indices[i + 1], n_samples - 1), bit_vectors, threshold)
        for i in range(len(batch_indices) - 1)
    )

    # Combine visualise into the neighbor mask
    neighbor_mask = np.zeros((n_samples, n_samples), dtype=bool)
    row_start = 0

    for batch in neighbor_batches:
        for i, row in enumerate(batch):
            row_len = len(row)
            neighbor_mask[row_start + i, row_start + i + 1: row_start + i + 1 + row_len] = row
            neighbor_mask[row_start + i + 1: row_start + i + 1 + row_len, row_start + i] = row  # Symmetric
        row_start += len(batch)

    df['neighbors'] = np.sum(neighbor_mask, axis=1)
    current_cluster_id = 0

    while not np.all(seen):

        unassigned_indices = np.where(~seen)[0]

        if np.all(df.loc[unassigned_indices, 'neighbors'] == 0):

            for idx in unassigned_indices:
                cluster_ids[idx] = current_cluster_id
                seen[idx] = True
                current_cluster_id += 1
            break

        most_neighbors_idx = unassigned_indices[np.argmax(df.loc[unassigned_indices, 'neighbors'])]

        cluster_members = np.where(neighbor_mask[most_neighbors_idx] & ~seen)[0]
        cluster_members = np.append(cluster_members, most_neighbors_idx)

        cluster_ids[cluster_members] = current_cluster_id
        seen[cluster_members] = True

        neighbor_mask[cluster_members, :] = False
        neighbor_mask[:, cluster_members] = False

        df.loc[~seen, 'neighbors'] = np.sum(neighbor_mask[~seen][:, ~seen], axis=1)  # potential improvement - use np array instead of pandas df

        current_cluster_id += 1

    df['Cluster_ID'] = cluster_ids
    df = df.drop(columns='neighbors')
    return df


def butina_split(df: pd.DataFrame, smiles_col: str = 'SMILES', fp_col: str = 'Morgan', strat_col: Union[str, List[str]] = 'Class',
                 threshold: float = 0.3, batch_size: int = 256, n_splits: int = 10, n_jobs: int = -2,
                 random_state: int = 0, cluster_col: str = None):
    """
    Perform Butina clustering and stratified group K-fold splitting.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame with fingerprint column.
    smiles_col: str
        name of the column with SMILES strings.
    fp_col: str
        Name of the column with fingerprints. Default is Morgan
    strat_col: Union[str, List[str]]
        Name(s) of column(s) to use for stratified split.
    threshold: float
        Distance threshold. Pair of FPs is marked as neighbors if distance <= threshold. Default is 0.4
    batch_size: int
        Number of rows allocated to each process during neighborhood matrix calculation.
    n_splits: int
        Number of folds to generate. Default is 10.
    n_jobs: int
        Number of cores to use. Default is -2 (i.e. use all but one cores)
    random_state: int
        Seed for pseudo-random number generator. Default is 0.
    cluster_col: str
        Name of the column with cluster IDs.

    Returns
    -------
    df: pd.DataFrame
        Modified dataframe with added cluster_id and fold columns.
    """
    if isinstance(strat_col, str):  # compatibility with 'strat_key' assignment
        strat_col = [strat_col]

    random_state = np.random.RandomState(random_state)

    if cluster_col is None:  # if cluster ids are not provided

        sub_df = df[[smiles_col, fp_col]].drop_duplicates(subset=[smiles_col])  # operate only on unique SMILES
        sub_df = butina_cluster(sub_df, fp_col=fp_col, threshold=threshold, n_jobs=n_jobs, batch_size=batch_size)
        df = df.merge(sub_df.drop(columns=fp_col), on=smiles_col, how='inner')

    x_values = csr_matrix(np.vstack(df[fp_col].to_numpy()))
    y_values = df[strat_col].to_numpy()
    groups = df['Cluster_ID'].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold, (_, test_idx) in enumerate(sgkf.split(x_values, y_values, groups)):
        df.loc[test_idx, 'Fold'] = fold

    df = df.astype({'Cluster_ID': 'UInt16', 'Fold': 'UInt8'})

    return df

