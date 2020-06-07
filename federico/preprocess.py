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
    weights (np.ndarray): the matrix of weights with size (trainset.n_users, trainset.n_items)
    '''

    # Define weights matrix
    weights = np.zeros((trainset.n_users, trainset.n_items))

    # Define array holding the ratings
    freqs = np.zeros(5)

    # Compute frequencies
    for _, _, r in trainset.all_ratings():
        freqs[int(r)-1] += 1

    # Compute sum of frequencies
    sum = np.sum(freqs)

    # Divide by sum
    freqs /= sum

    # Flip array
    freqs = np.flip(freqs)

    # Associate frequencies to weights
    for u, i, r in trainset.all_ratings():
        weights[u,i] = freqs[int(r)-1]

    # Return weights
    return weights

def items_frequency(trainset):
    '''
    Computes the frequency of each item. Frequency is computed as the number of users rating item i over the total number of users.

    Parameters:
    trainset (surprise.Trainset): the training data

    Returns:
    freqs (np.ndarray): the vector of frequencies with size (trainset.n_items)
    '''

    # Define frequency vector
    freqs = np.zeros(trainset.n_items)

    # Compute unscaled frequencies
    for _, i, _ in trainset.all_ratings():
        freqs[i] += 1

    # Scale frequencies
    for i in range(trainset.n_items):
        freqs[i] /= trainset.n_users

    # Return frequencies
    return freqs
