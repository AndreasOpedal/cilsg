import numpy as np
from sklearn.impute import SimpleImputer

def impute(X, missing_values=0, strategy='mean'):
    '''
    Imputes the missing values of the given matrix, according to the given strategy.

    Parameters:
    X (scipy.sparse.dok_matrix): the matrix to be imputed
    missing_values: the values to be replaced. By default 0
    strategy (string): which strategy to apply for imputing. By default 'mean'

    Returns:
    X (numpy.ndarray): the imputed matrix
    '''

    X = X.todense()

    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(X)
    X = imputer.transform(X)

    return X
