'''
This file contains an array of factorization methods based on SVD being solved via SGD.

The algorithms are implementated with the help of the Surprise package, which significantly helps with data management
and cross-validation.

The optimization is written in Cython, in order to speed up the costly computations.
This approach is similar to the one used in the Surprise package algorithms.

All algorithms follow the same structure as Simon Funk's SVD. The implemented algorithms are:
- SGDheu, where biases are added according to heuristics
- SGDPP2, an implentation of Koren's SVD++ using heuristics
'''

cimport numpy as np
import numpy as np
import math
from surprise import AlgoBase, PredictionImpossible
from baseline import SVD

class SGDheu(AlgoBase):
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
        cdef double r, err, dot, puf, qif

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

        Returns:
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

    def save_weights(self, dir):
        '''
        Save the weights of the model in the given directory

        dir (str): the directory to write the files into
        '''

        # Save mean
        np.save(self.mu, dir+'mu')
        # Save biases
        np.save(self.bias_u, dir+'bias_u')
        np.save(self.bias_i, dir+'bias_i')
        # Save P, Q
        np.save(self.P, dir+'P')
        np.save(self.Q, dir+'Q')

    def load_weights(self, dir):
        '''
        Loads the weights of the model from the given directory

        dir (str): the directory to write the files into
        '''

        # Load mean
        self.mu = np.load(dir+'mu.npy')
        # Load biases
        self.bias_u = np.load(dir+'bias_u.npy')
        self.bias_i = np.load(dir+'bias_i.npy')
        # Load P, Q
        self.P = np.load(dir+'P.npy')
        self.Q = np.load(dir+'Q.npy')

    def get_features(self, u, i):
        '''
        Returns the features for the given (user, index).
        The features have the form [mu, bias_u[u], bias_i[i], dot(P[u,:],Q[i,:])]

        Parameters:
        u (int): the user index
        i (int): the item index

        Returns:
        fv (numpy.ndarray): the feature vector
        '''

        # Initialize feature vector
        fv = np.zeros(4)

        # Fill feature vector
        fv[0] = self.mu
        fv[1] = self.bias_u[u]
        fv[2] = self.bias_i[i]
        fv[3] = np.dot(self.P[u,:], self.Q[i,:])

        # Return feature vector
        return fv

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

        cdef np.ndarray[np.double_t, ndim=2] V

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
        cdef double r, err, dot, puf, qif

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
        svd = SVD(n_factors=self.n_factors, impute_strategy=self.impute_strategy)
        svd.fit(self.trainset)
        V = svd.V

        for u in range(self.trainset.n_users):
            u_rated = 0
            for i, _ in self.trainset.ur[u]:
                u_rated += 1
                for f in range(self.n_factors):
                    u_impl_fdb[u,f] += V[i,f]
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

        Returns:
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

    def save_weights(self, dir):
        '''
        Save the weights of the model in the given directory

        dir (str): the directory to write the files into
        '''

        # Save mean
        np.save(self.mu, dir+'mu')
        # Save biases
        np.save(self.bias_u, dir+'bias_u')
        np.save(self.bias_i, dir+'bias_i')
        # Save item factors
        np.save(self.u_impl_fdb, dir+'u_impl_fdb')
        # Save P, Q
        np.save(self.P, dir+'P')
        np.save(self.Q, dir+'Q')

    def load_weights(self, dir):
        '''
        Loads the weights of the model from the given directory

        dir (str): the directory to write the files into
        '''

        # Load mean
        self.mu = np.load(dir+'mu.npy')
        # Load biases
        self.bias_u = np.load(dir+'bias_u.npy')
        self.bias_i = np.load(dir+'bias_i.npy')
        # Load item factors
        self.u_impl_fdb = np.load(dir+'u_impl_fdb.npy')
        # Load P, Q
        self.P = np.load(dir+'P.npy')
        self.Q = np.load(dir+'Q.npy')

    def get_features(self, u, i):
        '''
        Returns the features for the given (user, index).
        The features have the form [mu, bias_u[u], bias_i[i], dot(Q[i,:],P[u,:]+u_impl_fdb[u,:])]

        Parameters:
        u (int): the user index
        i (int): the item index

        Returns:
        fv (numpy.ndarray): the feature vector
        '''

        # Initialize feature vector
        fv = np.zeros(4)

        # Fill feature vector
        fv[0] = self.mu
        fv[1] = self.bias_u[u]
        fv[2] = self.bias_i[i]
        fv[3] = np.dot(self.Q[i,:], self.P[u,:] + self.u_impl_fdb[u,:])

        # Return feature vector
        return fv
