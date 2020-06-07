cimport numpy as np
import numpy as np
import math
from preprocess import build_weights, items_frequency
from surprise import AlgoBase, Dataset, PredictionImpossible

class SGDweighted(AlgoBase):
    '''
    Implementation of a weighted, heuristic version of SVD. The heuristics are applied to the biases, while the weighted model
    is based on the paper on fast matrix factorization from He et al. (2016).
    The weights are used for balancing reasons: this is because the ratings are imbalanced.
    '''

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std=0.1, alpha_pu=0.1, alpha_qi=0.1, reg_pu=0.5, reg_qi=0.5, lambda_bu=1, lambda_bi=1, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 20
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): initialization mean. By default 0
        alpha_pu (float): the dampening to apply to the weights at P. By default 0.1
        alpha_qi (float): the dampening to apply to the weights at Q. By default 0.1
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
        self.alpha_pu = alpha_pu
        self.alpha_qi = alpha_qi
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
        self.weights = build_weights(trainset)

        # Call SGD
        self.sgd()

        return self

    def sgd(self):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P, Q) = weights[u,i]*(r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)

        where b[u], b[i] are estimated via heuristics.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] weights
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef double mu = self.trainset.global_mean

        cdef double alpha_pu = self.alpha_pu
        cdef double alpha_qi = self.alpha_qi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double lambda_bu = self.lambda_bu
        cdef double lambda_bi = self.lambda_bi

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
            for u, i, r in self.trainset.all_ratings():
                # Compute estimated rating
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*P[u,f]
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    P[u,f] += alpha_pu*weights[u,i]*(err*qif - reg_pu*puf)
                    Q[i,f] += alpha_qi*weights[u,i]*(err*puf - reg_qi*qif)

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

class SGDheu(AlgoBase):
    '''
    Implementation of a heuristic version of Funk's SVD. The heuristics are applied to the biases.
    In the optimization, both learning rate (linear) decay and gradient momentum are used.
    '''

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.01, lr_qi=0.01, alpha_pu=0.01, alpha_qi=0.01, decay_pu=0.1, decay_qi=0.1, reg_pu=0.5, reg_qi=0.5, lambda_bu=1, lambda_bi=1, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 20
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

        H(P, Q) = (r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + (reg_pu*||p[u]||^2 + reg_qi*||q[i]||^2)

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

    def __init__(self, n_factors=20, n_epochs=20, lr_pu=0.01, lr_qi=0.01, reg_pu=1, reg_qi=1, lambda_bu=1, lambda_bi=1, r_min=1, r_max=5, max_init_p=1, max_init_q=1, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 20
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

        H(P, Q) = (r[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2
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

class SGDtum(AlgoBase):
    '''
    An implementation the biased version of SVD via stochastic gradient descent with momentum.
    '''

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std=0.1, lmb_bu=0.1, lmb_bi=0.1, lr0_pu=0.005, lr0_qi=0.005, lr0_bu=0.005, lr0_bi=0.005, reg_pu=0.02, reg_qi=0.02, reg_bu=0.002, reg_bi=0.002, decay_pu=0.01, decay_qi=0.01, decay_bu=0.01, decay_bi=0.01, alpha_pu=0.1, alpha_qi=0.1, alpha_bu=0.1, alpha_bi=0.1, init_bias=True, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 20
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): the initialization mean for factors. By default 0
        init_std (float): the initialization standard deviation for factors. By default 0.1
        lmb_bu (float): regularizer for the initalization of bias[u]. By default 0.1
        lmb_bi (float): regularizer for the initalization of bias[i]. By default 0.1
        lr0_pu (float): the initial learning rate for P. By default 0.005
        lr0_qi (float): the initial learning rate for Q. By default 0.005
        lr0_bu (float): the initial learning rate for bias[u]. By default 0.005
        lr0_bi (float): the initial learning rate for bias[i]. By default 0.005
        reg_pu (float): the regularization term for P. By default 0.02
        reg_qi (float): the regularization term for Q. By default 0.02
        reg_bu (float): the regularization term of bias[u]. By default 0.02
        reg_bi (float): the regularization term of bias[i]. By default 0.02
        decay_pu (float): the decay of the learning rate of P. By default 0.01
        decay_qi (float): the decay of the learning rate of Q. By default 0.01
        decay_bu (float): the decay of the learning rate of bias[u]. By default 0.01
        decay_bi (float): the decay of the learning rate of bias[i]. By default 0.01
        alpha_pu (float): the weight associated with the momentum of P. By default 0.01
        alpha_qi (float): the weight associated with the momentum of Q. By default 0.01
        alpha_bu (float): the weight associated with the momentum of bias[u]. By default 0.01
        alpha_bi (float): the weight associated with the momentum of bias[u]. By default 0.01
        init_bias (boolean): whether to initalize the biases via heuristics. By default True
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
        self.lmb_bu = lmb_bu
        self.lmb_bi = lmb_bi
        self.lr0_pu = lr0_pu
        self.lr0_qi = lr0_qi
        self.lr0_bu = lr0_bu
        self.lr0_bi = lr0_bi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.decay_pu = decay_pu
        self.decay_qi = decay_qi
        self.decay_bu = decay_bu
        self.decay_bi = decay_bi
        self.alpha_pu = alpha_pu
        self.alpha_qi = alpha_qi
        self.alpha_bu = alpha_bu
        self.alpha_bi = alpha_bi
        self.init_bias = init_bias
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.mu = None
        self.bias_u = None
        self.bias_i = None

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

        H(P, Q) = (X[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2 + b[u]^2 + b[i]^2)
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t] bias_u
        cdef np.ndarray[np.double_t] bias_i
        cdef np.ndarray[np.double_t, ndim=2] P
        cdef np.ndarray[np.double_t, ndim=2] Q

        cdef np.ndarray[np.double_t, ndim=2] delta_g_pu
        cdef np.ndarray[np.double_t, ndim=2] delta_g_qi
        cdef np.ndarray[np.double_t] delta_g_bu
        cdef np.ndarray[np.double_t] delta_g_bi

        cdef int u, i, f
        cdef int current_epoch
        cdef double r, err, dot, puf, qif
        cdef double mu = self.trainset.global_mean

        cdef double lmb_bu = self.lmb_bu
        cdef double lmb_bi = self.lmb_bi

        cdef double lr0_pu = self.lr0_pu
        cdef double lr0_qi = self.lr0_qi
        cdef double lr0_bu = self.lr0_bu
        cdef double lr0_bi = self.lr0_bi
        cdef double lr_pu = lr0_pu
        cdef double lr_qi = lr0_qi
        cdef double lr_bu = lr0_bu
        cdef double lr_bi = lr0_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double alpha_pu = self.alpha_pu
        cdef double alpha_qi = self.alpha_qi
        cdef double alpha_bu = self.alpha_bu
        cdef double alpha_bi = self.alpha_bi
        cdef double decay_pu = self.decay_pu
        cdef double decay_qi = self.decay_qi
        cdef double decay_bu = self.decay_bu
        cdef double decay_bi = self.decay_bi

        # Initialize P, Q (while keeping entries in range)
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        # Check for bias initalization
        if self.init_bias:
            for i in range(self.trainset.n_items):
                neigh_i = self.trainset.ir[i]
                for u, r in neigh_i:
                    bias_i[i] += r - mu
                bias_i[i] /= (lmb_bi + len(neigh_i))
            for u in range(self.trainset.n_users):
                neigh_u = self.trainset.ur[u]
                for i, r in neigh_u:
                    bias_u[u] += r - mu - bias_i[i]
                bias_u[u] /= (lmb_bu + len(neigh_u))

        # Initialize the deltas for the momentum
        delta_g_pu = np.zeros((self.trainset.n_users, self.n_factors))
        delta_g_qi = np.zeros((self.trainset.n_items, self.n_factors))
        delta_g_bu = np.zeros(self.trainset.n_users)
        delta_g_bi = np.zeros(self.trainset.n_items)

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay of learning rate
            lr_pu = lr0_pu/(1 + decay_pu*current_epoch)
            lr_qi = lr0_qi/(1 + decay_qi*current_epoch)
            lr_bu = lr0_bu/(1 + decay_bu*current_epoch)
            lr_bi = lr0_bi/(1 + decay_bi*current_epoch)
            for u, i, r in self.trainset.all_ratings():
                dot = 0
                for f in range(self.n_factors):
                    dot += Q[i,f]*P[u,f]
                err = r - (mu + bias_u[u] + bias_i[i] + dot)
                # Update step
                bias_u += lr_bu*(err - reg_bu*bias_u[u]) + alpha_bu*delta_g_bu[u]
                bias_i += lr_bi*(err - reg_bi*bias_i[i]) + alpha_bi*delta_g_bi[i]
                # Update momentum
                delta_g_bu[u] = alpha_bu*delta_g_bu[u] + lr_bu*(err - reg_bu*bias_u[u])
                delta_g_bi[i] = alpha_bi*delta_g_bi[i] + lr_bi*(err - reg_bi*bias_i[i])
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    P[u,f] += lr_pu*(err*qif - reg_pu*puf) + alpha_pu*delta_g_pu[u,f]
                    Q[i,f] += lr_qi*(err*puf - reg_qi*qif) + alpha_qi*delta_g_qi[i,f]
                    # Update momentum
                    delta_g_pu[u,f] = alpha_pu*delta_g_pu[u,f] + lr_pu*(err*qif - reg_pu*puf)
                    delta_g_qi[i,f] = alpha_qi*delta_g_qi[i,f] + lr_qi*(err*puf - reg_qi*qif)

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
            rui = np.dot(self.P[u,:], self.Q[i,:]) + self.mu + self.bias_u[u] + self.bias_i[i]
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

class SGDPP(AlgoBase):
    '''
    An implementation SVD++ via stochastic gradient descent.
    '''

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std=0.1, lr0=0.005, lr0_b=0.005, lr0_yj=0.005, reg=0.02, reg_bu=None, reg_bi=None, reg_yj=None, decay=0.01, decay_b=0.01, decay_yj=0.01, init_bias=True, low=1, high=5, conf=None, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_factors (int): the number of latent features. By default 20
        n_epochs (int): the number of iterations. By default 20
        init_mean (float): the initialization mean for factors. By default 0
        init_std (float): the initialization standard deviation for factors. By default 0.1
        lr0 (float): the initial learning rate. By default 0.005
        lr0_b (float): the initial learning rate for the biases. By default 0.005
        lr0_yj (float): the initial learning rate for the item factors. By default 0.005
        reg (float): the regularization term. By default 0.02
        reg_bu (float): the regularization term of bias[u]. By default 0.02
        reg_bi (float): the regularization term of bias[i]. By default 0.02
        reg_yj (float): the regularization term of the item factors. By default 0.02
        decay (float): the decay of the learning rate. By default 0.01
        decay_b (float): the decay of the learning rate of the biases. By default 0.01
        decay_yj (float): the decay of the learning rate of the item factors. By default 0.01
        init_bias (boolean): whether to initalize the biases via heuristics. By default False
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
        self.lr0 = lr0
        self.lr0_b = lr0_b
        self.lr0_yj = lr0_yj
        self.reg = reg
        self.reg_bu = reg_bu if reg_bu is not None else reg
        self.reg_bi = reg_bi if reg_bi is not None else reg
        self.reg_yj = reg_yj if reg_yj is not None else reg
        self.decay = decay
        self.decay_b = decay_b
        self.decay_yj = decay_yj
        self.init_bias = init_bias
        self.low = low
        self.high = high
        self.conf = conf
        self.verbose = verbose

        self.trainset = None
        self.P = None
        self.Q = None
        self.Y = None
        self.mu = None
        self.bias_u = None
        self.bias_i = None

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

        H(P, Q) = (X[u,i] - mu - b[u] - b[i] - q[i]*(p[u] + 1/sqrt(|N(u)|)*sum(y[j])))^2 +
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

        cdef int u, i, j, f
        cdef int current_epoch
        cdef double r, err, dot, puf, qif, sqrt_Iu, _
        cdef double mu = self.trainset.global_mean

        cdef double lr0 = self.lr0
        cdef double lr0_b = self.lr0_b
        cdef double lr0_yj = self.lr0_yj
        cdef double lr = lr0
        cdef double lr_b = lr0_b
        cdef double lr_yj = lr0_yj
        cdef double reg = self.reg
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_yj = self.reg_yj
        cdef double decay = self.decay
        cdef double decay_b = self.decay_b
        cdef double decay_yj = self.decay_yj

        # Initialize P, Q (while keeping entries in range)
        P = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_users,self.n_factors))
        Q = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))
        Y = np.random.normal(self.init_mean, self.init_std, (self.trainset.n_items,self.n_factors))

        # Initialize biases
        bias_u = np.zeros(self.trainset.n_users)
        bias_i = np.zeros(self.trainset.n_items)

        # Check for bias initalization
        if self.init_bias:
            for i in range(self.trainset.n_items):
                neigh_i = self.trainset.ir[i]
                for u, r in neigh_i:
                    bias_i[i] += r - mu
                bias_i[i] /= (reg_bi + len(neigh_i))
            for u in range(self.trainset.n_users):
                neigh_u = self.trainset.ur[u]
                for i, r in neigh_u:
                    bias_u[u] += r - mu - bias_i[i]
                bias_u[u] /= (reg_bu + len(neigh_u))

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # Decay of learning rate
            lr = lr0/(1 + decay*current_epoch)
            lr_b = lr0_b/(1 + decay_b*current_epoch)
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
                # Update steps
                bias_u += lr_b*(err - reg_bu*bias_u[u])
                bias_i += lr_b*(err - reg_bi*bias_i[i])
                for f in range(self.n_factors):
                    puf, qif = P[u,f], Q[i,f]
                    P[u,f] += lr*(err*qif - reg*puf)
                    Q[i,f] += lr*(err*(puf + u_impl_fdb[f]) - reg*qif)
                    for j in Iu:
                        Y[j,f] += lr_yj*(err*(qif/sqrt_Iu) - reg_yj*Y[j,f])

        # Write parameters
        self.P = P
        self.Q = Q
        self.bias_u = bias_u
        self.bias_i = bias_i
        self.mu = mu
        self.Y = Y

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
