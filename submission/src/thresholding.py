import numpy as np
from scipy.sparse import linalg
from scipy.sparse import csc_matrix
import math
from surprise import AlgoBase, PredictionImpossible

class SVDthr(AlgoBase):
    '''
    Implementation of SVD thresholding.
    '''

    def __init__(self, tao=10000, step_size=1.99, eps=0.2, low=1, high=5, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        tao (float): by how much should singular values be reduced. By default 10000
        step_size (float): the step size to apply in the projection. By default 1.99
        eps (float): the tolerance on the training error. By default 0.2
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        conf (float, [0,0.5]): the confidence interval for modifying the prediction. By default None
        verbose (bool): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.tao = tao
        self.step_size = step_size
        self.eps = eps
        self.low = low
        self.high = high
        self.verbose = verbose

        self.trainset = None
        self.pred_matrix = None

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset.

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Call SVD thresholding
        self.svd_thresholding()

        return self

    def svd_thresholding(self):
        '''
        Performs SVD thresholding on the training set.
        '''

        # Prepare training matrix A
        A = np.zeros((self.trainset.n_users,self.trainset.n_items))

        # Fill A
        for u, i, r in self.trainset.all_ratings():
            A[u,i] = r

        # Get non-zero indeces
        non_zero_indeces = np.nonzero(A)

        # Initialize X, Y
        X = np.zeros((self.trainset.n_users,self.trainset.n_items))
        Y = np.zeros((self.trainset.n_users,self.trainset.n_items))

        # Initialize heuristics to decide how many singular values to be used for the reconstruction
        r = 1
        l = 5

        # Current epoch counter
        current_epoch = 0

        # Initialize error such that it is larger than the tolerance
        err = self.eps+1

        # Optimize
        while err > self.eps:
            # Shrinkage
            sigma_min = self.tao + 1
            s = max(r,1)
            while sigma_min > self.tao:
                U, sigma, Vh = linalg.svds(csc_matrix(Y), k=min(s,999))
                sigma_min = min(sigma)
                s += l
            r = np.count_nonzero(sigma > self.tao)
            sigma_diag = np.diag(sigma)
            sigma_new = sigma_diag - self.tao
            sigma_new[sigma_new < 0] = 0
            X = np.dot(U, np.dot(sigma_new, Vh))
            # Projection
            proj = np.zeros((self.trainset.n_users,self.trainset.n_items))
            #print(non_zero_indeces)
            proj[non_zero_indeces] = (A - X)[non_zero_indeces]
            # Step forward
            Y += self.step_size*proj
            # Update error
            err = np.linalg.norm((X-A)[non_zero_indeces], ord=2)/np.linalg.norm(A[non_zero_indeces])
            if self.verbose:
                print('Error at epoch ' + str(current_epoch+1) + ': ' + str(err))
            # Update epoch counter
            current_epoch += 1

        # Final shrinkage
        U, sigma, Vh = np.linalg.svd(Y)
        sigma_diag = np.concatenate((np.diag(sigma), np.zeros((self.trainset.n_users-self.trainset.n_items,self.trainset.n_items))), axis=0)
        sigma_new = sigma_diag - self.tao
        sigma_new[sigma_new < 0] = 0
        self.pred_matrix = np.dot(U, np.dot(sigma_new, Vh))

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Returns:
        est (float): the rating estimate
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            # Compute estimate
            est = self.pred_matrix[u,i]
            # Clip result
            est = np.clip(est, self.low, self.high)
        else:
            raise PredictionImpossible('User and item are unknown.')

        return est
