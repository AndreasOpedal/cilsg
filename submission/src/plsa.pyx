cimport numpy as np
import numpy as np
import pandas as pd
from surprise import Reader, Dataset
from surprise import AlgoBase, PredictionImpossible

class pLSA(AlgoBase):
    '''
    Implementation of the pLSA algorithm using the EM-algorithm. After that, SVD is used to get the final
    prediction matrix
    '''

    def __init__(self, n_latent=20, n_eig=8, n_epochs=5, to_normalize=True, alpha=5, low=1, high=5, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_latent (int): the number of latent states. By default 5
        n_eig (int): the number of eigenvalues to use in the SVD. By default 8.
        n_epochs (int): the number of iterations. By default 5
        normalized (bool): whether to normalize observed rating matrix. By default False
        alpha (float): smooth factor to calculate user's mean and variance. Only used if "normalized" is True. By default 5
        low (int): the lowest rating value. By default 1
        high (int): the highest rating value. By default 5
        verbose (bool): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_latent = n_latent
        self.n_eig = n_eig
        self.n_epochs = n_epochs
        self.to_normalize = to_normalize
        self.alpha = alpha
        self.low = low
        self.high = high
        self.verbose = verbose

        self.trainset = None
        self.df = None
        self.X = None
        self.p_z = None
        self.mu_iz = None
        self.user_mu = None
        self.user_var = None
        self.pred_matrix = None

    def gaussian_model(self, float rating, np.ndarray mean, np.ndarray var):
        '''
        Gaussian distribution densify function.

        Parameters:
        rating (float): the rating
        mean (numpy.ndarray): the array where each entry represents the mean for a latent feature
        var (numpy.ndarray): the array where each entry represents the variance for a latent feature

        Returns:
        density (numpy.ndarray): the gaussian density for the given parameters
        '''

        # Set up ratings array
        ratings = np.zeros(mean.shape[0])

        # Fill ratings array
        for i in range(mean.shape[0]):
            ratings[i] = rating

        # Initialize, compute and return density
        cdef np.ndarray[np.double_t] density
        density = np.exp(-(ratings-mean)**2/(2*var))/(np.sqrt(2*np.pi*var))
        return density

    def normalize(self):
        '''
        Normalize observed rating matrix on users. Ratings of each user is subtracted by that user's smoothed
        rating mean and divided by smoothed standard deviation.
        '''

        # Compute variables
        mu = np.mean(self.df['Prediction'])
        var = np.var(self.df['Prediction'])
        rating_sum_per_user = self.df.groupby(by = 'row')['Prediction'].sum()
        counts = self.df.groupby(by = 'row')['row'].count()
        deviation_per_user = np.multiply(self.df.groupby(by = 'row')['Prediction'].var(), counts)

        # Compute normalization
        self.user_mu = (rating_sum_per_user + self.alpha*mu)/(counts + self.alpha)
        self.user_var = (deviation_per_user + self.alpha*var)/(counts + self.alpha)

        # Normalized data frame
        self.df = self.df.sort_values(by='row')
        norm_ratings = np.divide(self.df['Prediction'].to_numpy() - np.repeat(self.user_mu, counts), np.repeat(np.sqrt(self.user_var), counts))
        self.df['Prediction'] = norm_ratings.to_numpy()

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset.

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Normalize
        if self.to_normalize:
            # Prepare dataframe
            data = []
            for u, i, r in self.trainset.all_ratings():
                data.append([u,i,r])
            self.df = pd.DataFrame(data, columns=['row', 'col', 'Prediction'])
            # Perform normalization
            self.normalize()
            # Recreate trainset
            reader = Reader()
            self.trainset = Dataset.load_from_df(self.df[['row', 'col', 'Prediction']], reader).build_full_trainset()

        # Prepare training matrix
        self.X = np.zeros((self.trainset.n_users,self.trainset.n_items))
        for u, i, r in self.trainset.all_ratings():
            self.X[u,i] = r

        # Call EM algorithm
        self.em()

    def svd(self):
        '''
        Performs SVD with the training matrix imputed using the values from the matrix computed from the EM-algorithm.
        '''

        # Matrix holding the observed ratings
        A = np.zeros((self.trainset.n_users,self.trainset.n_items))

        # Fill the ratings
        for u, i, r in self.trainset.all_ratings():
            A[u,i] = r

        # Get non-zero indeces
        non_zeros_indeces = np.nonzero(A)

        # Impute using (computed) prediction matrix
        A[non_zeros_indeces] = self.pred_matrix[non_zeros_indeces]

        # Compute SVD
        U, s, Vh = np.linalg.svd(A)
        s = s[:self.n_eig]
        self.pred_matrix = U[:, :s.shape[0]]@np.diag(s)@Vh[:s.shape[0], :]

    def em(self):
        '''
        Gaussian probabilistic latent semantic analysis via EM-algorithm.
        '''

        # Cython initialization
        cdef np.ndarray[np.double_t, ndim=2] p_z
        cdef np.ndarray[np.double_t, ndim=2] mu_iz
        cdef np.ndarray[np.double_t, ndim=2] sigma2_iz
        cdef np.ndarray[np.double_t, ndim=3] p_z_given_uri
        cdef np.ndarray[np.double_t] p_rating_item
        cdef np.ndarray[np.double_t] item_rating

        cdef int z, u, i, user, item
        cdef double denom, nom, rating

        # Initalize
        p_z = np.random.rand(self.trainset.n_users,self.n_latent) #P(z|u)
        p_z /= p_z.sum(1)[:,np.newaxis]
        mu_iz = np.zeros((self.trainset.n_items,self.n_latent))
        sigma2_iz = np.ones((self.trainset.n_items,self.n_latent))
        p_z_given_uri = np.zeros((self.trainset.n_users,self.trainset.n_items,self.n_latent))

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # E-step
            if self.verbose:
                print('E-step...')
            for user, item, rating in self.trainset.all_ratings():
                p_rating_item = self.gaussian_model(rating, mu_iz[item,:], sigma2_iz[item,:]) # all p(rating, item|z)
                denom = np.dot(p_rating_item, p_z[user,:]) # p(rating,item|z)p(z|u)
                for z in range(self.n_latent):
                    nom = self.gaussian_model(rating, np.array([mu_iz[item,z]]), np.array([sigma2_iz[item,z]]))*p_z[user,z]
                    p_z_given_uri[user,item,z] = nom/denom # p(z|user, rating, item;thetahat)
            # M-step
            if self.verbose:
                print('M-step...')
            for u in range(self.trainset.n_users):
                denom = np.sum(p_z_given_uri[u]) # sum_z sum_u P(z|u, v, y;hattheta)
                for z in range(self.n_latent):
                    p_z[u,z] = np.sum(p_z_given_uri[u][:,z])/denom # sum of P(z|user, rating, item; thetahat)
            for i in range(self.trainset.n_items):
                for z in range(self.n_latent):
                    denom = np.sum(p_z_given_uri[:,i,z]) #sum_y {(z|user, rating, item)}
                    item_rating = self.X[:,i]
                    mu_iz[i,z] = np.dot(p_z_given_uri[:,i,z], item_rating)/denom
                    sigma2_iz[i,z] = np.dot(np.square(item_rating - mu_iz[i,z]), p_z_given_uri[:,i,z])/denom

        # Write parameters
        self.p_z = p_z
        self.mu_iz = mu_iz

        # Compute prediction matrix
        self.pred_matrix = np.nan_to_num(self.p_z)@np.nan_to_num(self.mu_iz.T)
        if self.to_normalize:
            self.pred_matrix = np.add(np.multiply(self.pred_matrix, self.user_var[:,np.newaxis]), self.user_mu[:,np.newaxis])

        # Apply SVD
        self.svd()

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
