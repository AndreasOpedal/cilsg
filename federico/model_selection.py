import numpy as np
from tqdm.auto import tqdm
from itertools import product
from surprise import accuracy

def grid_search(algorithm, param_grid, trainset, testset, target_score=None, verbose=False):
    '''
    Performs grid search.

    Parameters:
    algorithm (AlgoBase): the algorithm class to use
    param_grid (dictionary): the dictionary containing the model parameters
    trainset (Trainset): the training set
    testset (Trainset): the test set
    target_score (float): a target score which should be reached by the model. If specified, then only configurations beating
                          the target score will be saved. By default None
    verbose (boolean): whether the algorithm should print the configuration at each round. By default False

    Returns:
    results (list): the list of all configurations. Each element is a tuple, the first element being the score and the
                    second being the configuration
    '''

    # Build configurations
    configurations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]

    # List to save configurations
    results = []

    # Best configuration
    best_config = None

    # Best score
    best_score = None

    for config in tqdm(configurations):
        # Create, train, and test model
        model = algorithm(**config)
        model.fit(trainset)
        predictions = model.test(testset)
        score = accuracy.rmse(predictions)
        if target_score is None or score < target_score:
            results.append((score, config))
            if verbose:
                print(config)
        # Update (if needed) best configuration
        if best_score is None or score < best_score:
            best_score = score
            best_config = config

    # Print best score and best configuration
    print('Best score/configuration')
    print(best_score)
    print(best_config)

    print('####################################################')
    print('####################################################')

    # Sort results based on score
    results.sort(key = lambda x: x[0])

    return results
