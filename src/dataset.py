import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix

def load_from_csv(filename):
    """
    Loads a sparse dataset from a CSV file.
    """
    data = pd.read_csv(filename)
    data = [
        ( row['Prediction'], row['Id'].split('_') )
        for idx, row in tqdm(data.iterrows(), total=data.shape[0])
    ]
    preds, rows, cols = [d[0] for d in data], [int(d[1][0][1:]) for d in data], [int(d[1][1][1:]) for d in data]
    assert len(preds) == len(rows) == len(cols)
    rows = np.subtract(rows, 1)
    cols = np.subtract(cols, 1)
    assert np.min(rows) == 0
    assert np.min(cols) == 0
    return csr_matrix((preds, (rows, cols)))

def _load_cached(filename):
    if not os.path.exists(filename + ".npy"):
        print("Caching dataset %s ..." % filename)
        dataset = load_from_csv(filename)
        np.save(filename + ".npy", dataset)
    return np.load(filename + ".npy")


def load_datasets():
    X_train = _load_cached("../data/data_train.csv")
    X_test = _load_cached("../data/sampleSubmission.csv")
    X_valid = None
    # TODO: train-dev split
    return X_train, X_valid, X_test
