'''
This file contains some baseline algorithms.

The algorithms are implementated with the help of the Surprise package, which significantly helps with data management
and cross-validation.

The implemented algorithms are:
- ALS
'''

cimport numpy as np
import numpy as np
import math
from preprocess import build_weights, items_frequency
from surprise import AlgoBase, Dataset, PredictionImpossible

class ALS(AlgoBase):
    '''
    Implementation of the ALS algorithm
    '''

    def __init__(self, n_factors=160, n_epochs=20, init_mean=0, init_std=0.1, reg=0.5, low=1, high=5, conf=None, verbose=True):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): initialization mean. By default 0
        init_std (float): initialization standard deviation. By default 0.1
        reg (float): the regularization strength. By default 0.5
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.reg = reg
        self.low = low
        self.high = high
        self.conf = conf
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

        # Initialize P, Q
        self.P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        self.Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize identity
        indentity = np.identity((self.n_factors,self.n_factors))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            for u in range(self.trainset.n_users):
                temp_pq = np.zeros((self.n_factors,self.n_factors))
                temp_r = np.zeros((self.n_factors,self.n_factors))
                for i, r in self.trainset.ur[u]:
                    temp_pq += np.dot(self.Q[i,:], self.Q[i,:].T)
                    temp_r += r*self.Q[i,:]
                temp_pq += self.reg*indentity
                self.P[u,] = np.dot(np.linalg.inv(temp_pq), temp_r)
            for i in range(self.trainset.n_items):
                temp_pq = np.zeros((self.n_factors,self.n_factors))
                temp_r = np.zeros((self.n_factors,self.n_factors))
                for u, r in self.trainset.ir[i]:
                    temp_pq += np.dot(self.P[u,:], self.P[u,:].T)
                    temp_r += r*self.Q[u,:]
                temp_pq += self.reg*indentity
                self.Q[i,] = np.dot(np.linalg.inv(temp_pq), temp_r)

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
            rui = np.dot(self.P[u,:], self.Q[i,:])
            # Clip result
            if rui < self.low:
                rui = self.low
            if rui > self.high:
                rui = self.high
            if self.conf is not None:
                # Intify prediction
                delta = 1-(rui%1)
                if 0.5-delta >= self.conf:
                    rui = math.ceil(rui)
                elif delta-0.5 >= self.conf:
                    rui = math.floor(rui)
        else:
            raise PredictionImpossible('User and item are unknown.')

        return rui
