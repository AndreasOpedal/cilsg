import numpy as np

def mse(X_true, X_pred):
    '''
    Computes the MSE error between two provided matrices.

    Parameters:
    X_true (scipy.sparse.dok_matrix or numpy.ndarray): the ground-truth matrix
    X_pred (numpy.ndarray): the prediction matrix

    Returns:
    error (float): the error between the two matrices
    '''

    indexes = X_true.nonzero()
    error = np.sum(np.square(X_true[indexes]-X_pred[indexes])/2)/indexes[0].shape[0]

    return error

def rmse(X_true, X_pred):
    '''
    Computes the RMSE error between two provided matrices. It calls the
    mse function and then applies the square root.

    Parameters:
    X_true (scipy.sparse.dok_matrix or numpy.ndarray): the ground-truth matrix
    X_pred (numpy.ndarray): the prediction matrix

    Returns:
    error (float): the error between the two matrices
    '''

    error = np.sqrt(mse(X_true, X_pred))
    return error
