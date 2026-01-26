import joblib
import pandas as pd


def read_df(path: str):
    """
    Helper function for reading pandas DataFrames
    """
    ext = path.split('.')[-1]
    if ext == 'xlsx':
        df = pd.read_excel(path)
    elif ext == 'csv':
        df = pd.read_csv(path)
    elif ext == 'tsv':
        df = pd.read_csv(path, sep='\t')
    elif ext == 'parquet':
        df = pd.read_parquet(path)
    elif ext == 'pkl':
        df = pd.read_pickle(path)
    elif ext == 'joblib':
        df = joblib.load(path)
    else:
        raise ValueError('Incorrect extension')

    return df


def write_df(df: pd.DataFrame, path: str):
    """
    Helper function for writing pandas DataFrames
    """
    ext = path.split('.')[-1]
    if len(df) >= 1:
        if ext == 'xlsx':
            df.to_excel(path, index=False)
        elif ext == 'csv':
            df.to_csv(path, index=False, header=True)
        elif ext == 'tsv':
            df.to_csv(path, index=False, header=True, sep='\t')
        elif ext == 'parquet':
            df.to_parquet(path)
        elif ext == 'pkl':
            df.to_pickle(path)
        elif ext == 'joblib':
            joblib.dump(df, path)
        else:
            raise ValueError('Incorrect extension')
