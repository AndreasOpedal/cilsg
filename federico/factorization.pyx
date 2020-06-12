'''
This file contains an array of factorization methods based on SVD being solved via SGD.

The algorithms are implementated with the help of the Surprise package, which significantly helps with data management
and cross-validation.

The optimization is written in cython, in order to speed up the costly computations.

All algorithms follow the same structure as Simon Funk's SVD. The implemented algorithms are:
- SGDweighted, where each rating(u,i) is weighted according to its representation in the training data
- SGDheu, where biases are added according to heuristics
- SGDbound, where heavy regularization is imposed in order for the prediction to remain within a specified bound
- SVDPP, an implentation of Koren's SVD++
'''

cimport numpy as np
import numpy as np
import math
from preprocess import build_weights
from surprise import AlgoBase, Dataset, PredictionImpossible
from baseline import SVD

class SGDweighted(AlgoBase):
    '''
    Implementation of a weighted, heuristic version of SVD. The heuristics are applied to the biases, while the weighted model
    is based on the paper on fast matrix factorization from He et al. (2016).
    The weights are used for balancing reasons: this is because the ratings are imbalanced.
    '''

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.1, lr_qi=0.1, decay_pu=0.1, decay_qi=0.1, reg_pu=0.5, reg_qi=0.5, lambda_bu=1, lambda_bi=1, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): initialization mean. By default 0
        lr_pu (float): the learning rate for P. By default 0.1
        lr_qi (float): the learning rate for Q. By default 0.1
        decay_pu (float): the decay associated with lr_pu. By default 0.1
        decay_qi (float): the decay associated with lr_pu. By default 0.1
        init_std (float): initialization standard deviation. By default 0.1
        reg_pu (float): the regularization strength for P. By default 0.5
        reg_qi (float): the regularization strength for Q. By default 0.5
        lambda_bu (float): the regularizer for the initialization of b[u]. By default 1
        lambda_bi (float): the regularizer for the initialization of b[i]. By default 1
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.decay_pu = decay_pu
        self.decay_qi = decay_qi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.weights = None
        self.P = None
        self.Q = None
        self.bias_u = None
        self.bias_i = None
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

        # Compute weights
        self.weights = build_weights(self.trainset)

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P,Q)[u,i] = weights[u,i]*(r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)

        where b[u], b[i] are estimated via heuristics.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] weights
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef double mu = self.trainset.global_mean

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double decay_pu = self.decay_pu
        cdef double decay_qi = self.decay_qi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi

        cdef double lr0_pu = lr_pu
        cdef double lr0_qi = lr_qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif

        # Set weights and frequencies
        weights = self.weights

        # Initialize P, Q (while keeping entries in range)
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        for i in range(self.trainset.n_items):
            neigh_i = self.trainset.ir[i]
            for u, r in neigh_i:
                bias_i[i] += r - mu
            bias_i[i] /= (lambda_bi + len(neigh_i))
        for u in range(self.trainset.n_users):
            neigh_u = self.trainset.ur[u]
            for i, r in neigh_u:
                bias_u[u] += r - mu - bias_i[i]
            bias_u[u] /= (lambda_bu + len(neigh_u))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay
            lr_pu = lr0_pu/(1 + decay_pu*current_epoch)
            lr_qi = lr0_qi/(1 + decay_qi*current_epoch)
            for u, i, r in self.trainset.all_ratings():
                # Compute estimated rating
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*P[u,f]
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    P[u,f] += lr_pu*(weights[u,i]*err*qif - reg_pu*puf)
                    Q[i,f] += lr_qi*(weights[u,i]*err*puf - reg_qi*qif)

        # Write parameters
        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu

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
            rui = np.dot(self.P[u,:], self.Q[i,:]) + self.bias_u[u] + self.bias_i[i] + self.mu
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

class SGD(AlgoBase):
    '''
    Implementation of a biased version of Simon Funk's SVD. In this algorithm, the biases are computed via heuristics.
    In the optimization, both learning rate (linear) decay and gradient momentum are used.
    '''

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.01, lr_qi=0.01, alpha_pu=0.01, alpha_qi=0.01, decay_pu=0.1, decay_qi=0.1, reg_pu=0.5, reg_qi=0.5, lambda_bu=1, lambda_bi=1, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): initialization mean. By default 0
        init_std (float): initialization standard deviation. By default 0.1
        lr_pu (float): the learning rate for P. By default 0.01
        lr_qi (float): the learning rate for P. By default 0.01
        alpha_pu (float): the strength of the gradient momentum of P. By default 0.01
        alpha_qi (float): the strength of the gradient momentum of Q. By default 0.01
        decay_pu (float): the decay associated with lr_pu. By default 0.1
        decay_qi (float): the decay associated with lr_pu. By default 0.1
        reg_pu (float): the regularization strength for P. By default 0.5
        reg_qi (float): the regularization strength for Q. By default 0.5
        lambda_bu (float): the regularizer for the initialization of b[u]. By default 1
        lambda_bi (float): the regularizer for the initialization of b[i]. By default 1
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.alpha_pu = alpha_pu
        self.alpha_qi = alpha_qi
        self.decay_pu = decay_pu
        self.decay_qi = decay_qi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.bias_u = None
        self.bias_i = None
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

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P,Q)[u,i] = (r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)

        where b[u], b[i] are estimated via heuristics.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef double mu = self.trainset.global_mean

        cdef np.ndarray[np.double_t, ndim=2] delta_g_pu
        cdef np.ndarray[np.double_t, ndim=2] delta_g_qi

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double alpha_pu = self.alpha_pu
        cdef double alpha_qi = self.alpha_qi
        cdef double decay_pu = self.decay_pu
        cdef double decay_qi = self.decay_qi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi

        cdef double lr0_pu = lr_pu
        cdef double lr0_qi = lr_qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif, gradient_pu, gradient_qi

        # Initialize P, Q
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        for i in range(self.trainset.n_items):
            neigh_i = self.trainset.ir[i]
            for u, r in neigh_i:
                bias_i[i] += r - mu
            bias_i[i] /= (lambda_bi + len(neigh_i))
        for u in range(self.trainset.n_users):
            neigh_u = self.trainset.ur[u]
            for i, r in neigh_u:
                bias_u[u] += r - mu - bias_i[i]
            bias_u[u] /= (lambda_bu + len(neigh_u))

        # Initialize mometum
        delta_g_pu = np.zeros((self.trainset.n_users,self.n_factors))
        delta_g_qi = np.zeros((self.trainset.n_items,self.n_factors))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay
            lr_pu = lr0_pu/(1 + decay_pu*current_epoch)
            lr_qi = lr0_qi/(1 + decay_qi*current_epoch)
            for u, i, r in self.trainset.all_ratings():
                # Dot product
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*P[u,f]
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    # Update momentum
                    delta_g_pu[u,f] = alpha_pu*delta_g_pu[u,f] + lr_pu*(err*qif - reg_pu*puf)
                    delta_g_qi[i,f] = alpha_qi*delta_g_qi[i,f] + lr_qi*(err*puf - reg_qi*qif)
                    # Update P, Q
                    P[u,f] += delta_g_pu[u,f]
                    Q[i,f] += delta_g_qi[i,f]

        # Write parameters
        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu

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
            rui = np.dot(self.P[u,:], self.Q[i,:]) + self.bias_u[u] + self.bias_i[i] + self.mu
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

class SGDbound(AlgoBase):
    '''
    Implementation of the bounded SVD (with bias via heuristics) proposed by Le et al. (2016).
    '''

    def __init__(self, n_factors=100, n_epochs=20, lr_pu=0.01, lr_qi=0.01, reg_pu=1, reg_qi=1, lambda_bu=1, lambda_bi=1, r_min=1, r_max=5, max_init_p=1, max_init_q=1, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        lr_pu (float): the learning rate for P. By default 0.01
        lr_qi (float): the learning rate for P. By default 0.01
        reg_pu (float): the regularization strength for P. By default 1
        reg_qi (float): the regularization strength for Q. By default 1
        lambda_bu (float): the regularizer for the initialization of b[u]. By default 1
        lambda_bi (float): the regularizer for the initialization of b[i]. By default 1
        r_min (float): the minimum rating value. By default 1
        r_max (float): the maximum rating value. By default 5
        max_init_p (float): the maximum value for the initalization of P. By default 1
        max_init_q (float): the maximum value for the initalization of Q. By default 1
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.r_min = r_min
        self.r_max = r_max
        self.max_init_p = max_init_p
        self.max_init_q = max_init_q
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.bias_u = None
        self.bias_i = None
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

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P,Q)[u,i] = (r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2
                    + (exp(mu + b[u] + b[i] + p[u]*q[i] - r_max) + exp(r_min - mu - b[u] - b[i] - p[u]*q[i]))
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef double mu = self.trainset.global_mean

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi
        cdef double r_min = self.r_min
        cdef double r_max = self.r_max

        cdef int u, i, f
        cdef double r, rui, h, err, dot, puf, qif

        # Initalize P, Q
        P = np.random.uniform(0, self.max_init_p, (self.trainset.n_users,self.n_factors))
        Q = np.random.uniform(0, self.max_init_q, (self.trainset.n_items,self.n_factors))


        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        for i in range(self.trainset.n_items):
            neigh_i = self.trainset.ir[i]
            for u, r in neigh_i:
                bias_i[i] += r - mu
            bias_i[i] /= (lambda_bi + len(neigh_i))
        for u in range(self.trainset.n_users):
            neigh_u = self.trainset.ur[u]
            for i, r in neigh_u:
                bias_u[u] += r - mu - bias_i[i]
            bias_u[u] /= (lambda_bu + len(neigh_u))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            for u, i, r in self.trainset.all_ratings():
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*P[u,f]
                rui = mu + bias_u[u] + bias_i[i] + dot
                err = r - rui
                h = np.exp(rui - r_max) - np.exp(r_min - rui)
                # Update step
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    P[u,f] += lr_pu*qif*(err - reg_pu*h)
                    Q[i,f] += lr_qi*puf*(err - reg_qi*h)

        # Write parameters
        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu

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
            rui = np.dot(self.P[u,:], self.Q[i,:]) + self.bias_u[u] + self.bias_i[i] + self.mu
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

class SGDPP(AlgoBase):
    '''
    An implementation of Koren's SVD++ via stochastic gradient descent.
    In the optimization, both learning rate (linear) decay and gradient momentum are used.
    '''

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.01, lr_qi=0.01, lr_yj=0.01, alpha_pu=0.01, alpha_qi=0.01, alpha_yj=0.01, decay_pu=0.01, decay_qi=0.01, decay_yj=0.01, reg_pu=0.5, reg_qi=0.5, reg_yj=0.5, lambda_bu=1, lambda_bi=1, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): the initialization mean for factors. By default 0
        init_std (float): the initialization standard deviation for factors. By default 0.1
        lr_pu (float): the learning rate for P. By default 0.01
        lr_qi (float): the learning rate for Q. By default 0.01
        lr_yj (float): the learning rate for the item factors. By default 0.01
        alpha_pu (float): the strength of the gradient momentum of P. By default 0.01
        alpha_qi (float): the strength of the gradient momentum of Q. By default 0.01
        alpha_yj (float): the strength of the gradient momentum of Y. By default 0.01
        decay_pu (float): the decay of the learning rate of P. By default 0.01
        decay_qi (float): the decay of the learning rate of P. By default 0.01
        decay_yj (float): the decay of the learning rate of the item factors. By default 0.01
        reg_pu (float): the regularization term for P. By default 0.5
        reg_qi (float): the regularization term for Q. By default 0.5
        reg_yj (float): the regularization term of the item factors. By default 0.5
        lambda_bu (float): the regularizer for the initialization of b[u]. By default 1
        lambda_bi (float): the regularizer for the initialization of b[i]. By default 1
        low (int): the lower bound for a prediction. By default 1
        high (int): the upper bound for a prediction. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.lr_yj = lr_yj
        self.alpha_pu = alpha_pu
        self.alpha_qi = alpha_qi
        self.alpha_yj = alpha_yj
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.reg_yj = reg_yj
        self.decay_pu = decay_pu
        self.decay_qi = decay_qi
        self.decay_yj = decay_yj
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.Y = None
        self.bias_u = None
        self.bias_i = None
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

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q, biases, and item factors by optimizing the following objective function:

        H(P,Q)[u,i] = (r[u,i] - mu - b[u] - b[i] - q[i]*(p[u] + 1/sqrt(|N(u)|)*sum(y[j])))^2 +
                    + l*(b[u]^2 + b[i]^2 + ||p[u]||^2 + ||q[i]||^2 + sum(||y[j]||^2))

        where mu is the global (rating) average, b[u] and b[i] are the users and items biases respectively,
        N(u) represent the items rated by user u, and each y[j] represent one item factor.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t, ndim=2] Y
        cdef np.ndarray[np.double_t] u_impl_fdb
        cdef double mu = self.trainset.global_mean

        cdef np.ndarray[np.double_t, ndim=2] delta_g_pu
        cdef np.ndarray[np.double_t, ndim=2] delta_g_qi
        cdef np.ndarray[np.double_t, ndim=2] delta_g_yj

        cdef int u, i, f, j
        cdef int current_epoch
        cdef double r, err, dot, puf, qif, sqrt_Iu, _

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj
        cdef double alpha_pu = self.alpha_pu
        cdef double alpha_qi = self.alpha_qi
        cdef double alpha_yj = self.alpha_yj
        cdef double decay_pu = self.decay_pu
        cdef double decay_qi = self.decay_qi
        cdef double decay_yj = self.decay_yj
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi

        cdef double lr0_pu = lr_pu
        cdef double lr0_qi = lr_qi
        cdef double lr0_yj = lr_yj

        # Initialize P, Q (while keeping entries in range)
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))
        Y = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        # Initalize the biases
        for i in range(self.trainset.n_items):
            neigh_i = self.trainset.ir[i]
            for u, r in neigh_i:
                bias_i[i] += r - mu
            bias_i[i] /= (lambda_bu + len(neigh_i))
        for u in range(self.trainset.n_users):
            neigh_u = self.trainset.ur[u]
            for i, r in neigh_u:
                bias_u[u] += r - mu - bias_i[i]
            bias_u[u] /= (lambda_bi + len(neigh_u))

        # Initialize momentum
        delta_g_pu = np.zeros((self.trainset.n_users,self.n_factors))
        delta_g_qi = np.zeros((self.trainset.n_items,self.n_factors))
        delta_g_yj = np.zeros((self.trainset.n_items,self.n_factors))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay of learning rate
            lr_pu = lr0_pu/(1 + decay_pu*current_epoch)
            lr_qi = lr0_qi/(1 + decay_qi*current_epoch)
            lr_yj = lr0_yj/(1 + decay_yj*current_epoch)
            for u, i, r in self.trainset.all_ratings():
                Iu = [j for (j, _) in self.trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += Y[j,f]/sqrt_Iu
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*(P[u,f] + u_impl_fdb[f])
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    # Update mometum
                    delta_g_pu[u,f] = alpha_pu*delta_g_pu[u,f] + lr_pu*(err*qif - reg_pu*puf)
                    delta_g_qi[i,f] = alpha_qi*delta_g_qi[i,f] + lr_qi*(err*(puf + u_impl_fdb[f]) - reg_qi*qif)
                    # Update P, Q
                    P[u,f] += delta_g_pu[u,f]
                    Q[i,f] += delta_g_qi[i,f]
                    for j in Iu:
                        # Update mometum
                        delta_g_yj[j,f] = alpha_yj*delta_g_yj[j,f] + lr_yj*(err*(qif/sqrt_Iu) - reg_yj*Y[j,f])
                        # Update Y
                        Y[j,f] += delta_g_yj[j,f]

        # Write parameters
        self.P = P
        self.Q = Q
        self.Y = Y
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu

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
            rui = self.mu + self.bias_u[u] + self.bias_i[i]
            Iu = len(self.trainset.ur[u])
            u_impl_feedback = (sum(self.Y[j] for (j, _) in self.trainset.ur[u])/np.sqrt(Iu))
            rui += np.dot(self.Q[i,:], self.P[u,:] + u_impl_feedback)
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
            raise PredictionImpossible('User and item are unkown.')

        return rui

class SGDPP2(AlgoBase):
    '''
    Implementation of SVD++. In this algorithm, the biases and item factors are computed via heuristics.
    In the optimization, both learning rate (linear) decay and gradient momentum are used.
    '''

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.01, lr_qi=0.01, alpha_pu=0.01, alpha_qi=0.01, decay_pu=0.1, decay_qi=0.1, reg_pu=0.5, reg_qi=0.5, lambda_bu=1, lambda_bi=1, lambda_yj=1, impute_strategy=None, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 100
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): initialization mean. By default 0
        init_std (float): initialization standard deviation. By default 0.1
        lr_pu (float): the learning rate for P. By default 0.01
        lr_qi (float): the learning rate for P. By default 0.01
        alpha_pu (float): the strength of the gradient momentum of P. By default 0.01
        alpha_qi (float): the strength of the gradient momentum of Q. By default 0.01
        decay_pu (float): the decay associated with lr_pu. By default 0.1
        decay_qi (float): the decay associated with lr_pu. By default 0.1
        reg_pu (float): the regularization strength for P. By default 0.5
        reg_qi (float): the regularization strength for Q. By default 0.5
        lambda_bu (float): the regularizer for the initialization of b[u]. By default 1
        lambda_bi (float): the regularizer for the initialization of b[i]. By default 1
        lambda_bi (float): the regularizer for the initialization of the item factors. By default 1
        impute_strategy (string): the strategy to use to impute the non-rated items. The options are None (0), 'mean', and
                                  'median'. By default None
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.alpha_pu = alpha_pu
        self.alpha_qi = alpha_qi
        self.decay_pu = decay_pu
        self.decay_qi = decay_qi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.lambda_yj = lambda_yj
        self.impute_strategy = impute_strategy
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.u_impl_fdb = None
        self.bias_u = None
        self.bias_i = None
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

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P,Q)[u,i] = (r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)

        where b[u], b[i] are estimated via heuristics.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t, ndim=2] u_impl_fdb
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef double mu = self.trainset.global_mean

        cdef np.ndarray[np.double_t, ndim=2] V_k

        cdef np.ndarray[np.double_t, ndim=2] delta_g_pu
        cdef np.ndarray[np.double_t, ndim=2] delta_g_qi

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double alpha_pu = self.alpha_pu
        cdef double alpha_qi = self.alpha_qi
        cdef double decay_pu = self.decay_pu
        cdef double decay_qi = self.decay_qi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi
        cdef double lambda_yj = self.lambda_yj

        cdef double lr0_pu = lr_pu
        cdef double lr0_qi = lr_qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif, gradient_pu, gradient_qi

        # Initialize P, Q
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        for i in range(self.trainset.n_items):
            neigh_i = self.trainset.ir[i]
            for u, r in neigh_i:
                bias_i[i] += r - mu
            bias_i[i] /= (lambda_bi + len(neigh_i))
        for u in range(self.trainset.n_users):
            neigh_u = self.trainset.ur[u]
            for i, r in neigh_u:
                bias_u[u] += r - mu - bias_i[i]
            bias_u[u] /= (lambda_bu + len(neigh_u))

        # Initialize item factors
        u_impl_fdb = np.zeros((self.trainset.n_users,self.n_factors))

        # Use SVD to get item factors
        svd = SVD(self.n_factors, self.impute_strategy)
        svd.fit(self.trainset)
        V_k = svd.V_k

        for u in range(self.trainset.n_users):
            u_rated = 0
            for i, _ in self.trainset.ur[u]:
                u_rated += 1
                for f in range(self.n_factors):
                    u_impl_fdb[u,f] += V_k[i,f]
            for f in range(self.n_factors):
                u_impl_fdb[u,f] /= ((lambda_yj + u_rated)*np.sqrt(u_rated))

        # Initialize mometum
        delta_g_pu = np.zeros((self.trainset.n_users,self.n_factors))
        delta_g_qi = np.zeros((self.trainset.n_items,self.n_factors))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay
            lr_pu = lr0_pu/(1 + decay_pu*current_epoch)
            lr_qi = lr0_qi/(1 + decay_qi*current_epoch)
            for u, i, r in self.trainset.all_ratings():
                # Dot product
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*(P[u,f] + u_impl_fdb[u,f])
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    # Update momentum
                    delta_g_pu[u,f] = alpha_pu*delta_g_pu[u,f] + lr_pu*(err*qif - reg_pu*puf)
                    delta_g_qi[i,f] = alpha_qi*delta_g_qi[i,f] + lr_qi*(err*(puf + u_impl_fdb[u,f]) - reg_qi*qif)
                    # Update P, Q
                    P[u,f] += delta_g_pu[u,f]
                    Q[i,f] += delta_g_qi[i,f]

        # Write parameters
        self.P = P
        self.Q = Q
        self.u_impl_fdb = u_impl_fdb
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu

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
            rui = np.dot(self.Q[i,:], self.P[u,:] + self.u_impl_fdb[u,:]) + self.bias_u[u] + self.bias_i[i] + self.mu
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
