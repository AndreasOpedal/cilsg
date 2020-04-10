import numpy as np
import math

def intify_prediction(X_pred, low=1, high=5, conf=0.4):
    '''
    The prediction matrix can have non-integer results. The actual ratings are obviously integers, so having non-integer
    number can yield worse performance.
    Each prediction element is converted to an integer: however, what if the predicted value is 3.4, we set it to 3, but the
    real prediction was 4? Do mitigate this problem only predictions withing the provided confidence interval (conf) are modified,
    i.e. either floor or ceil is applied. A prediction falls withing the confidence interval if |0.5-(1-d(p))|>=conf, where d(p) is the
    decimal part of the prediction. Then floor or ceil applied accordingly.
    The predictions are also clipped: if any prediction is outside of the specified interval (low, high), then it is clipped.

    Parameters:
    X_pred (numpy.ndarray): the prediction matrix
    low (int): the lower bound
    high (int): the upper bound
    conf (float): the confidence interval for modifying the prediction

    Returns:
    X_pred (numpy.ndarray): the modified prediction matrix
    '''

    # Get dimensions of the matrix
    m, n = X_pred.shape[0], X_pred.shape[1]

    # Begin modification
    for i in range(m):
        for j in range(n):
            if X_pred[i,j] < low:
                X_pred[i,j] = low
            if X_pred[i,j] > high:
                X_pred[i,j] = high
            delta = 1-(X_pred[i,j]%1)
            if 0.5-delta >= conf:
                X_pred[i,j] = math.ceil((X_pred[i,j]))
            elif delta-0.5 >= conf:
                X_pred[i,j] = math.floor((X_pred[i,j]))

    return X_pred
