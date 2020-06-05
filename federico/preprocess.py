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
    weights (np.ndarray): the matrix of the weights, with size (trainset.n_users, trainset.n_items)
    '''

    # Define weights matrix
    weights = np.zeros((trainset.n_users, trainset.n_items))

    # Define array holding the ratings
    freqs = np.zeros(5)

    # Compute frequencies
    for u, i, r in trainset.all_ratings():
        freqs[int(r)-1] += 1

    # Compute sum of frequencies
    sum = freqs[0] + freqs[1] + freqs[2] + freqs[3] + freqs[4]

    # Rescale frequencies
    freqs[0] /= sum
    freqs[1] /= sum
    freqs[2] /= sum
    freqs[3] /= sum
    freqs[4] /= sum

    # Associate frequencies to weights
    for u, i, r in trainset.all_ratings():
        weights[u,i] = freqs[int(r)-1]

    # Return weights
    return weights
