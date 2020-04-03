import numpy as np

def mse(X_true, X_pred):
    indexes = X_true.nonzero()
    return np.sum(np.square(X_true[indexes]-X_pred[indexes])/2)/indexes[0].shape[0]

def rmse(X_true, X_pred):
    return np.sqrt(mse(X_true, X_pred))
