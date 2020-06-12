import numpy as np
import pandas as pd
import random

def synthetic_ratings(df, indexes, to_add=90000, rating=1):
    '''
    Adds the specified ratings to the specified DataFrame.

    Parameters:
    df (pandas.DataFrame): the data to modify
    indexes (list): the list of empty indexes
    to_add (int): the amount of ratings to add. By default 90000
    rating (int): the rating to add. By default 1

    Returns:
    df (pandas.DataFrame): the modified DataFrame
    '''

    print('Processing data...')

    # Random sample of indexes
    selected = random.sample(indexes, to_add)

    # Counter for iloc
    counter = 1

    # Iterate through selected indexes
    for u, i in selected:
        df.loc[df.size+counter] = [u, i, rating]
        counter += 1

    print('Finished processing.')

    return df

def build_weights(trainset):
    '''
    Given training data, build a matrix which associates a weight to each entry. Specifically, a weight will be associated to each
    rating {1,2,3,4,5}.
    The matrix of weights can then be used in the SGDweighted algorithm.

    Parameters:
    trainset (surprise.Trainset): the training data

    Returns:
    weights (numpy.ndarray): the matrix of weights with size (trainset.n_users, trainset.n_items)
    '''

    # Define weights matrix
    weights = np.zeros((trainset.n_users, trainset.n_items))

    # Define array holding the ratings
    freqs = np.zeros(5)

    # Compute frequencies
    for _, _, r in trainset.all_ratings():
        freqs[int(r)-1] += 1

    # Rescale frequencies
    freqs /= np.sum(freqs)

    # Flip array
    freqs = np.flip(freqs)

    # Associate frequencies to weights
    for u, i, r in trainset.all_ratings():
        weights[u,i] = freqs[int(r)-1]

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
    reps = np.zeros(5)

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
