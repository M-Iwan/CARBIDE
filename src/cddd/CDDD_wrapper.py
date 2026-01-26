""" Wrapper for CDDD.py due to requirement of specific python environment """


import argparse
import ast
import pandas as pd
import numpy as np
import pickle
from cddd.inference import InferenceModel
from typing import Union, List

parser = argparse.ArgumentParser(prog='CDDD', description='Calculate CDDD descriptors from SMILES')

parser.add_argument('-i', '--input', help='Path to an input file. Must be a .csv file.')
parser.add_argument('-o', '--output', help='Path to an output file. Must be a .pkl file.')
parser.add_argument('-s', '--smiles_col', help='Name of the column holding the SMILES')
parser.add_argument('-d', '--descriptor_col', help='Name of the column to which write the descriptors')
parser.add_argument('-n', '--n_cpu', help='Number of CPU threads to use during calculations')
parser.add_argument('-m', '--model_dir', help='Path to directory with CDDD default model')

parsed_args = parser.parse_args()


def main(args):

    input_ = args.input  # .csv file
    output_ = args.output   # .pkl file
    smiles_col_ = args.smiles_col
    descriptor_col_ = args.descriptor_col
    n_cpu = int(args.n_cpu)
    model_dir = args.model_dir

    model = InferenceModel(model_dir, use_gpu=False, cpu_threads=n_cpu)

    df = pd.read_csv(input_, sep='\t')
    # unpack lists
    df.loc[:, smiles_col_] = df[smiles_col_].apply(lambda entry: entry.split(' : ') if ' : ' in entry else entry)

    def process(smiles: Union[str, List[str]], model_):

        if isinstance(smiles, str):
            try:
                fp = model_.seq_to_emb(smiles)[0]
                return fp

            except Exception as e:
                print(f'Could not process < {smiles} > due to the following error:\n{e}')
                return np.nan

        if isinstance(smiles, list) & all([isinstance(smi, str) for smi in smiles]):
            try:
                fps = [model_.seq_to_emb(smi)[0] for smi in smiles]
                return fps

            except Exception as e:
                print(f'Unable to process < {smiles} > due to: \n{e}')
                return np.nan

        print(f'Expected < smiles > argument to be of type str, received < {type(smiles)} > instead.')
        return np.nan

    df[descriptor_col_] = df[smiles_col_].apply(process, model_=model)

    pickle.dump(df, open(output_, 'wb'), protocol=4)


if __name__ == '__main__':
    main(parsed_args)
