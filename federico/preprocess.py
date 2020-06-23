import numpy as np
import pandas as pd
import random

def build_weights(trainset):
    '''
    Given training data, build a matrix which associates a weight to each entry. Specifically, a weight will be associated to each
    rating {1,2,3,4,5}.

    Parameters:
    trainset (surprise.Trainset): the training data

    Returns:
    weights (numpy.ndarray): the vectors of weights with size for each rating
    '''

    # Define weights matrix
    weights = np.zeros(trainset.n_ratings,)

    # Define array holding the ratings
    freqs = np.zeros(5)

    # Compute frequencies
    for _, _, r in trainset.all_ratings():
        freqs[int(r)-1] += 1

    # Rescale frequencies
    freqs /= np.sum(freqs)

    # Flip array
    freqs = np.flip(freqs)

    # Weights row counter
    rows = 0

    # Associate frequencies to weights
    for _, _, r in trainset.all_ratings():
        weights[rows] = freqs[int(r)-1]
        rows += 1

    # Return weights
    return weights

def over_sample(trainset):
    '''
    Computes for each rating belonging to {1,2,3,4,5} the number of times a tuple (u,i,r) should be iterated over in the
    optimization procedure. The number of repetitions is proportional to the maximum number of times a rating appears.
    For example, if the rating 1 appears 10 times, and the rating 5 has the maximum number of appearances with 100, then
    each rating 1 will be iterated over 100/10=10 times in the optimization procedure.

    Parameters:
    trainset (surprise.Trainset): the training data

    Returns:
    reps (numpy.ndarray): the array where each entry holds the number of repetitions for a given rating
    '''

    # Repetitions array
    reps = np.zeros(5, dtype=np.int64)

    # Fill reps
    for _, _, r in trainset.all_ratings():
        reps[int(r)-1] += 1

    # Compute maximum number rating
    max_num = np.max(reps)

    # Compute final number of repetitions
    for r in range(5):
        reps[r] = int(max_num/reps[r])

    # Return repetitions array
    return reps
