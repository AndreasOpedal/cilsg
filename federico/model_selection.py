import numpy as np
import numpy.matlib
import pandas as pd
from scipy.sparse import dok_matrix
import math
from tqdm.auto import tqdm
import os
import utils

def train_test_split(X, min_user_ratings=100, test_movies_pct=0.4):
    '''
    Splits the data into a training and testing data sets.

    Source:
    https://gist.github.com/curiousily/62d84edbdbe9f0f069ef3812c9c54dd4#file-rs-train-test-split-py

    Parameters:
    X (scipy.sparse.dok_matrix): the matrix to be split
    min_user_ratings (int): the minimum number of movies a user must have rated. By default 100
    test_movies_pct (float): the percentage of movies review by a user to put into the test set. By default 0.4

    Returns:
    X_train, X_test (scipy.sparse.dok_matrix, scipy.sparse.dok_matrix): the training and testing sets
    '''

    # Initialization
    X_train = X.copy()
    X_test = dok_matrix(X.shape)

    for user in np.arange(X.shape[0]):
        len_nonzero = len(X[user,:].nonzero()[0])
        if len_nonzero >= min_user_ratings:
            test_movie_index = np.random.choice(len_nonzero, int(test_movies_pct*len_nonzero), replace=False)
            X_train[user, test_movie_index] = 0
            X_test[user, test_movie_index] = X[user, test_movie_index]

    return X_train, X_test

def kfold_split(X_df, k=5, save=False):
    '''
    Splits the provided data into k folds. The strategy is to sample users.

    Parameters:
    X_df (pandas.DataFrame): the data to split
    k (int): the number of folds. By default 5
    save (boolean): whether to save the folds as csv files. The default file path is './cv-k-fold/fold-i.csv, where k represents
                    the chosen k and fold-i represents the i-th fold. By default False

    Returns:
    fold_list (list): a list of size k, where each entry holds a data frame, which corresponds to a single fold
    '''

    # Create empty df to fill with shuffled data
    shuffled_df = pd.DataFrame(columns=['row', 'col', 'Prediction'])

    for i in tqdm(np.arange(10000)):
        num_rows = X_df[X_df['row'] == i].shape[0]
        row_sample = X_df[X_df['row'] == i].sample(n=num_rows)
        shuffled_df = shuffled_df.append(row_sample)

    # Assign folds to rows
    folds = np.append(np.matlib.repmat(np.arange(k)+1, 1, math.floor(shuffled_df.shape[0]/k)),
                     (np.arange(k)+1)[0:shuffled_df.shape[0]%k])

    shuffled_df['Fold'] = folds

    # Return separate fold data frames in list, drop fold column
    fold_list = []
    for j in np.arange(k)+1:
        fold_list.append(shuffled_df[shuffled_df['Fold'] == j].drop('Fold', axis = 1))

    if save:
        dir_name = 'cv-' + str(k) + '-fold'
        dir_path = os.path.join('./', dir_name)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        for i in range(k):
            file_name = 'fold-' + str(i) + '.csv'
            file_path = './' + dir_name + '/' + file_name
            fold_list[i].to_csv(file_path)

    return fold_list

def kfold_cv(model, metric, folds=None, k=5, imputer=None, dir_path=None):
    '''
    Performs k-fold crossvalidation on either the given data or the folds at the given location.

    Parameters:
    folds (list): the list containing the folds on which to perform kfold CV. Each entry should be a pandas DataFrame.
                  By default None
    k (int): the number of folds. This number is necessary if the folds are to be read from disk. By default 5
    model (class): the class representing the model
    metric (function): the metric to be used to compute the error
    imputer (function): the imputer to be applied to both train and test matrices. By default None
    dir_path (string): the directory where the folds are stored. The file for the i-th fold should named as
                       follows: fold-i.csv, where i represents the i-th fold. Note that numbering starts from 0.
                       By default None

    Returns:
    mean, var (float, float): the mean and variance of the error computed at each iteration (with respect to the selected metric)
    '''

    if folds is None:
        # Initialize folds as list and read from disk
        folds = []
        for i in range(k):
            file_path = dir_path + 'fold-' + str(i) + '.csv'
            # Read into data frame
            df = pd.read_csv(file_path)
            folds.append(df)

    # Initalize scores
    scores = []

    for i in tqdm(range(k)):
        # Create training and test set
        df_train = pd.concat([x for j, x in enumerate(folds) if j!=i])
        df_test = folds[i]
        # Transform train and test data frames into sparse matrices
        X_train = utils.data_frame_to_sparse_matrix(df_train)
        X_test = utils.data_frame_to_sparse_matrix(df_test)
        if not (imputer is None):
            X_train = imputer(X_train)
            X_test = imputer(X_test)
        # Run model
        model.fit(X_train)
        # Make prediction
        X_pred = model.transform()
        # Compute error
        score = metric(X_test, X_pred)
        # Add error to scores
        scores.append(score)

    # Compute mean and variance
    mean, var = np.mean(scores), np.var(scores)

    return mean, var
