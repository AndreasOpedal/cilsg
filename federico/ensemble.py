import numpy as np
import math
from surprise import AlgoBase, PredictionImpossible
from factorization import SGDheu, SGDheu

class WeightedAvg(AlgoBase):
    '''
    An ensemble algorithm where different models are averaged using a weighted averaging scheme.
    Users provided paths to weights of already trained models.
    '''

    def __init__(self, base_models, n_epochs=20, lr=0.1, normalize=True, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        base_models (list): a list where each element is a model instance. As for the type of allowed models to use, check the
                            Surprise documentation, and the classes in factorization.pyx and baseline.py.
        n_epochs (int): the number of iterations. By default 20
        lr (float): the learning rate. By default
        normalize (bool): whether to normalize the weights at each iteration. By default True
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (bool): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)
        self.base_models = base_models
        self.n_epochs = n_epochs
        self.lr = lr
        self.normalize = normalize
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.weights = []
        self.n = -1

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Number of base models
        self.n = len(self.base_models)

        if self.verbose:
            print('Training base models...')
        for j in range(self.n):
            self.base_models[j].fit(self.trainset)
        if self.verbose:
            print('Finished training base models.')

        # Call SGD
        self.sgd()

    def sgd(self):
        '''
        Learns the weights for the averaging.
        '''

        # Initialize weights
        for j in range(self.n):
            self.weights.append(1)
        if self.normalize:
            ws_norm = 0
            for j in range(self.n):
                ws_norm += self.weights[j]**2
            ws_norm = math.sqrt(ws_norm)
            for j in range(self.n):
                self.weights[j] /= ws_norm

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            for u, i, r in self.trainset.all_ratings():
                num = 0
                den = 0
                for j in range(self.n):
                    num += self.weights[j]*self.base_models[j].estimate(u,i)
                    den += self.weights[j]
                rui = num/den
                err = r - rui
                for j in range(self.n):
                    self.weights[j] += self.lr*(err*((self.base_models[j].estimate(u,i)*den - num)/(den**2)))
                if self.normalize:
                    ws_norm = 0
                    for j in range(self.n):
                        ws_norm += self.weights[j]**2
                    ws_norm = math.sqrt(ws_norm)
                    for j in range(self.n):
                        self.weights[j] /= ws_norm

        if self.verbose:
            print('Learned weights: ', self.weights)

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Returns:
        rui (float): the prediction
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            # Compute prediction
            num = 0
            den = 0
            for j in range(self.n):
                num += self.weights[j]*self.base_models[j].estimate(u,i)
                den += self.weights[j]
            rui = num/den
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
