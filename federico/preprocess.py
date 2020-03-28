import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

def impute(X, missing_values=0, strategy='mean'):
    '''
    Imputes the missing values of the given matrix, according to the given strategy.

    Parameters:
    X (scipy.sparse.dok_matrix): the matrix to be imputed
    missing_values (number, string, np.nan or None): the values to be replaced. By default 0
    strategy (string): which strategy to apply for imputing. By default 'mean'

    Returns:
    X (numpy.ndarray): the imputed matrix
    '''

    X = X.todense()

    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(X)
    X = imputer.transform(X)

    return X

def knn_impute(X, missing_values=0, n_neighbors=5):
    '''
    Imputes the missing values of the given matrix, using kNN.

    Parameters:
    X (scipy.sparse.dok_matrix): the matrix to be imputed
    missing_values (number, string, np.nan or None): the values to be replaced. By default 0
    n_neighbors (int): the number or neirest neighbors

    Returns:
    X (numpy.ndarray): the imputed matrix
    '''

    X = X.todense()

    imputer = KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors)
    imputer.fit(X)
    X = imputer.transform(X)

    return X
