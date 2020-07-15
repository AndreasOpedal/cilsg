'''
This file contains the implementation of baseline algorithms.

The algorithm is implementated with the help of the Surprise package, which significantly helps with data management
and cross-validation. Furthermore, the notation used in the implmentation mirrors the one used in the Surprise base
algorithms (see their GitHub repository).

The implemented algorithms are:
- SVD
- ALS
'''

import numpy as np
import math
from surprise import AlgoBase, Dataset, PredictionImpossible
from surprise.prediction_algorithms.knns import KNNBasic

class SVD(AlgoBase):
    '''
    Implementation of SVD.
    '''

    def __init__(self, n_factors=160, impute_strategy=None):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 160
        impute_strategy (object): the strategy to use to impute the non-rated items. The options are 'mean', 'median', or any integer
                                  value. By default None (filled with zeros)
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.impute_strategy = impute_strategy

        self.U = None
        self.V = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

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
        if isinstance(self.impute_strategy, int):
            X[X<1] = int(self.impute_strategy)
        elif self.impute_strategy == 'mean':
            for u in range(self.trainset.n_users):
                X[u,:] = np.where(X[u,:]<1, np.mean(X[u,:]), X[u,:])
        elif self.impute_strategy == 'median':
            for u in range(self.trainset.n_users):
                X[u,:] = np.where(X[u,:]<1, np.median(X[u,:]), X[u,:])

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

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.reg = reg
        self.low = low
        self.high = high
        self.verbose = verbose

        self.P = None
        self.Q = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

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
