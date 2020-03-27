import numpy as np
from scipy.sparse import dok_matrix

MIN_USER_RATINGS = 100
TEST_MOVIES_PCT = 0.4

def train_test_split(X):
    '''
    Splits the data into a training and testing data sets.

    Source:
    https://gist.github.com/curiousily/62d84edbdbe9f0f069ef3812c9c54dd4#file-rs-train-test-split-py

    Parameters:
    X (scipy.sparse.dok_matrix): the matrix to be split

    Returns:
    X_train, X_test (scipy.sparse.dok_matrix, scipy.sparse.dok_matrix): the training and testing sets
    '''

    # Initialization
    X_train = X.copy()
    X_test = dok_matrix(X.shape)

    for user in np.arange(X.shape[0]):
        len_nonzero = len(X[user,:].nonzero()[0])
        if len_nonzero >= MIN_USER_RATINGS:
            test_movie_index = np.random.choice(len_nonzero, int(TEST_MOVIES_PCT*len_nonzero), replace=False)
            X_train[user, test_movie_index] = 0
            X_test[user, test_movie_index] = X[user, test_movie_index]

    return X_train, X_test
