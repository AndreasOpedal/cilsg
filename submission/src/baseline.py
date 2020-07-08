'''
This file contains some baseline algorithms.

The algorithms are implementated with the help of the Surprise package, which significantly helps with data management
and cross-validation.

The implemented algorithms are:
- Mean
- SVD
- ALS
'''

import numpy as np
import math
from surprise import AlgoBase, Dataset, PredictionImpossible
from surprise.prediction_algorithms.knns import KNNBasic

class Mean(AlgoBase):
    '''
    Basic approach which predicts empty ratings with the ratings global mean.
    '''

    def __init__(self):
        '''
        Initializes the class with the given parameters.
        '''

        self.trainset = None
        self.mu = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Compute global mean
        self.mu = self.trainset.global_mean

        return self

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Retuns:
        rui (float): the prediction
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            # Compute prediction
            rui = self.mu
        else:
            raise PredictionImpossible('User and item are unknown.')

        return rui

class SVD(AlgoBase):
    '''
    Implementation of SVD.
    '''

    def __init__(self, n_factors=160, impute_strategy='zeros'):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 160
        impute_strategy (object): the strategy to use to impute the non-rated items. The options are 'zeros', 'mean', 'median'.
                                  By default 'zeros'
        '''

        self.n_factors = n_factors
        self.impute_strategy = impute_strategy

        self.trainset = None
        self.U = None
        self.V = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Call SVD
        self.svd()

        return self

    def svd(self):
        '''
        Finds matrices P, Q by computing the SVD relative to the training data.
        '''

        # Build the training matrix
        X = np.zeros((self.trainset.n_users,self.trainset.n_items))

        # Fill the training matrix
        for u, i, r in self.trainset.all_ratings():
            X[u,i] = r

        # Impute empty ratings (if instructed)
        if self.impute_strategy == 'mean':
            X[X<1] = self.trainset.global_mean
        elif self.impute_strategy == 'median':
            median = np.median(X)
            X[X<1] = median

        # Compute the SVD of X
        U, S, Vt = np.linalg.svd(X)
        D = np.zeros(shape=(S.shape[0],S.shape[0])) # create diagonal matrix D
        np.fill_diagonal(D,S) # fill D with S

        # Square root of D
        D = np.sqrt(D)

        # Pad D
        D_p = np.append(D, np.zeros((U.shape[0]-D.shape[0],D.shape[0])), axis=0)

        U = U.dot(D_p)
        V = D.dot(Vt.T)

        # Select vectors from U, V
        self.U = U[:,:self.n_factors]
        self.V = V[:,:self.n_factors]

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Retuns:
        est (float): the rating estimate
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            # Compute prediction
            est = np.dot(self.U[u,:], self.V[i,:])
            # Clip result
            est = np.clip(est, self.low, self.high)
        else:
            raise PredictionImpossible('User and item are unknown.')

        return est

class ALS(AlgoBase):
    '''
    Implementation of ALS.
    '''

    def __init__(self, n_factors=160, n_epochs=5, init_mean=0, init_std=0.1, reg=1, low=1, high=5, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 5
        init_mean (float): initialization mean. By default 0
        init_std (float): initialization standard deviation. By default 0.1
        reg (float): the regularization strength. By default 1
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        verbose (bool): whether the algorithm should be verbose. By default False
        '''

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.reg = reg
        self.low = low
        self.high = high
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Call ALS
        self.als()

        return self

    def als(self):
        '''
        Finds matrices P, Q by optimizing the following objective function via ALS:

        H(P,Q) = (r[u,i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)
        '''

        # Set up rating matrix
        A = np.zeros((self.trainset.n_users,self.trainset.n_items))

        # Fill rating matrix
        for u, i, r in self.trainset.all_ratings():
            A[u,i] = r

        # Initialize P, Q
        P = np.random.normal(self.init_mean, self.init_std, (self.n_factors,self.trainset.n_users))
        Q = np.random.normal(self.init_mean, self.init_std, (self.n_factors,self.trainset.n_items))

        # Identity of size k
        Id_k = np.identity(self.n_factors)

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            for u in range(self.trainset.n_users):
                obs_items = np.nonzero(A[u,:])
                sum_qqT = np.sum([Q[:, i]*np.reshape(Q[:, i], (-1, 1)) for i in obs_items[0]], 0)
                P[:, u] = np.squeeze(np.linalg.inv(sum_qqT+self.reg*Id_k) @ np.reshape(np.sum([A[u,item]*Q[:, item] for item in obs_items[0]], 0), (-1, 1)))
            for i in range(self.trainset.n_items):
                obs_users = np.nonzero(A[:, i])
                sum_ppT = np.sum([P[:, u]*np.reshape(P[:, u], (-1, 1)) for u in obs_users[0]], 0)
                Q[:, i] = np.squeeze(np.linalg.inv(sum_ppT+self.reg*Id_k) @ np.reshape(np.sum([A[u, i]*P[:, u] for u in obs_users[0]], 0), (-1, 1)))

        # Write parameters
        self.P, self.Q = P, Q

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Retuns:
        est (float): the rating estimate
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            # Compute estimate
            est = np.dot(self.P[:,u], self.Q[:,i])
            # Clip result
            est = np.clip(est, self.low, self.high)
        else:
            raise PredictionImpossible('User and item are unknown.')

        return est
