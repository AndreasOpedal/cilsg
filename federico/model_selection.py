import numpy as np
from surprise import accuracy

def targeting_rmse(model, testset):
    '''
    Computes the RMSE for each target rating, where each rating belongs to {1,2,3,4,5}.

    Parameters:
    model (AlgoBase): the model to use to test the data. The model should have been trained
    testset (list): the test set
    '''

    # Array holding values
    rmses = np.zeros(5)

    # Compute errors
    for u, i, r in testset:
        rui = model.predict(u, i).est # prediction
        rmses[r-1] += r - rui

    # Compute RMSEs
    for r in range(5):
        rmse = np.sqrt((rmses[r-1]**2)/len(errors))
        print('Rating: {}, RMSE: {}, #: {}'.format(key, rmse, len(errors)))
