import os
import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix

"""
This file contains preprocessing utilities used by neural models.
"""

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
    return csr_matrix((preds, (rows, cols)))

def _load_cached(filename):
    if not os.path.exists(filename + ".npy"):
        print("Caching dataset %s ..." % filename)
        dataset = load_from_csv(filename)
        np.save(filename + ".npy", dataset)
    return np.load(filename + ".npy", allow_pickle=True).tolist() # will give csr_matrix


def create_n_splits(filename, n=5, random_state=42):
    """
    Splits a CSV file into n equal sized parts, stratified by row(user).
    Splits will be saved as "original_filename.number_of_split".

    This will always produce identical splits given identical random_state parameter.
    """
    n_rows, n_cols = 10000, 1000  # it's faster this way, see above
    data = pd.read_csv(filename)
    user_data = dict()

    # obtain list of user-movie positions for each user
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        user_id = row['Id'].split('_')[0][1:]
        user_data.setdefault(user_id, []).append(idx)
    
    # item2fold = {}  # mapping item -> fold index
    fold2items = [[] for i in range(n)] # mapping fold index -> items

    # assign 1/N user-movie positions to each fold
    for user, items in user_data.items():
        items = sklearn.utils.shuffle(list(items), random_state=random_state)
        items_per_fold = len(items) // n  # yes, last fold will have more items
        for i, item in enumerate(items):
            #item2fold[item] = i // items_per_fold 
            if items_per_fold <= 0:
                # not enough items, we'll just put it in random fold
                fold2items[i % n].append(item)
            else:
                # put each item in the correct fold
                fold2items[min(i // items_per_fold, n-1)].append(item)

    fold_filenames = [f"{filename}.{i}" for i in range(n)]
    for i in range(n):
        fold_data = data.iloc[fold2items[i]]
        fold_data.to_csv(fold_filenames[i], index=False)


def load_datasets(train_idx=[0,1,2,3], valid_idx=4):
    X_train = np.sum([_load_cached("../data/data-train.csv.%d" % i) for i in train_idx])
    X_test = _load_cached("../data/sample-submission.csv")
    X_valid = _load_cached(f"../data/data-train.csv.{valid_idx}")

    return X_train, X_valid, X_test
