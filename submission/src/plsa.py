import numpy as np
import pandas as pd
from surprise import Reader, Dataset
from surprise import AlgoBase, PredictionImpossible

class PLSA(AlgoBase):
    '''
    Implementation of the pLSA algorithm using the EM-algorithm. After that, SVD is used to get the final
    prediction matrix
    '''

    def __init__(self, n_latent=20, n_eig=8, n_epochs=5, to_normalize=True, alpha=5, low=1, high=5, verbose=False):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_latent (int): the number of latent states. By default 20
        n_eig (int): the number of eigenvalues to use in the SVD. By default 8.
        n_epochs (int): the number of iterations. By default 5
        normalized (bool): whether to normalize observed rating matrix. By default True
        alpha (float): smoothing factor to calculate user's mean and variance. Only used if "to_normalize" is True. By default 5
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

        self.pred_matrix = None

    def gaussian_model(self, rating, mean, var):
        '''
        Gaussian distribution densify function.

        Parameters:
        rating (float): the rating
        mean (numpy.ndarray): the array where each entry represents the mean for a latent feature
        var (numpy.ndarray): the array where each entry represents the variance for a latent feature

        Returns:
        density (numpy.ndarray): the gaussian density for the given parameters
        '''
         # Compute density
        density = np.exp(-(rating-mean)**2/(2*var))/(np.sqrt(2*np.pi*var))

        return density

    def df_to_mat(self, ratings):
        '''
        Convert a rating table with information of row index, column index and associated rating to a 2d-array.

        Parameters:
        ratings (pandas.DataFrame): 3 columns with observed row index, column index and rating of full rating matrix

        Returns:
        result (numpy.ndarray): the rating matrix with zeros on unobserved values
        '''

        result = np.zeros((self.trainset.n_users,self.trainset.n_items))
        for i in range(self.trainset.n_users):
            colidx = ratings[ratings['row'] == i]['col']
            result[i,colidx] = ratings[ratings['row'] == i]['Prediction']
        return result

    def normalize(self, ratings):
        '''
        Normalize observed rating matrix on users. Ratings of each user is subtracted by that user's smoothed
        rating mean and divided by smoothed standard deviation.

        Parameters:
        ratings (pandas.DataFrame): 3 columns with observed row index, column index and rating of full rating matrix

        Returns:
        usermu, uservar, result (numpy.ndarray, numpy.ndarray, pandas.DataFrame)
        '''

        mu = np.mean(ratings['Prediction'])
        var = np.var(ratings['Prediction'])
        rating_sum_per_user = ratings.groupby(by = 'row')['Prediction'].sum()
        counts = ratings.groupby(by = 'row')['row'].count()
        deviation_per_user = np.multiply(ratings.groupby(by = 'row')['Prediction'].var(), counts)
        usermu = (rating_sum_per_user + self.alpha*mu)/(counts + self.alpha)
        uservar = (deviation_per_user + self.alpha*var)/(counts + self.alpha)
        result = ratings.copy().sort_values(by='row')
        norm_ratings = np.divide(result['Prediction'].to_numpy() - np.repeat(usermu, counts), np.repeat(np.sqrt(uservar), counts))
        result['Prediction'] = norm_ratings.to_numpy()
        return usermu, uservar, result

    def svd(self, ratings, prediction):
        '''
        Compute the SVD with the training matrix imputed with the predictions

        Parameters:
        ratings (pandas.DataFrame): 3 columns with observed row index, column index and rating of full rating matrix
        prediction (numpy.ndarray): full rating matrix generated from the model

        Returns:
        svdresult (numpy.ndarray): full rating matrix computed by SVD
        '''

        train_matrix = self.df_to_mat(ratings)
        train_r, train_c = ratings.loc[:, 'row'], ratings.loc[:, 'col']
        prediction[train_r, train_c] = train_matrix[train_r, train_c]
        u, s, vh = np.linalg.svd(prediction)
        s = s[:self.n_eig]
        svdresult = u[:, :s.shape[0]]@np.diag(s)@vh[:s.shape[0], :]
        return svdresult

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset.

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Call EM algorithm
        self.em()

    def em(self):
        '''
        Gaussian probabilistic latent semantic analysis via EM-algorithm.
        '''

        # Prepare (training) ratings dataframe
        ratings = []
        for u, i, r in self.trainset.all_ratings():
            ratings.append([u, i, r])
        ratings = pd.DataFrame(ratings, columns=['row', 'col', 'Prediction'])

        # Initialization
        p_z = np.random.rand(self.trainset.n_users,self.n_latent) #P(z|u)
        p_z /= p_z.sum(1)[:,np.newaxis]

        if self.to_normalize:
            usermu, uservar, ratings = self.normalize(ratings)

        mu_iz = np.zeros((self.trainset.n_items,self.n_latent))
        sigma2_iz = np.ones((self.trainset.n_items,self.n_latent))
        p_z_given_uri = np.zeros((self.trainset.n_users,self.trainset.n_items,self.n_latent))
        rating_mat = self.df_to_mat(ratings)

        # Optimize
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print('Processing epoch {}'.format(current_epoch+1))
            # E-step
            if self.verbose:
                print('E-step...')
                def helper(row):
                    user = np.int(row[0])
                    item = np.int(row[1])
                    rating = row[2]
                    p_rating_item = self.gaussian_model(rating, mu_iz[item], sigma2_iz[item]) # all p(rating,item|z)
                    denom = np.dot(p_rating_item, p_z[user]) #p(rating, item|z)p(z|u)
                    for z in range(self.n_latent):
                        nom = self.gaussian_model(rating, mu_iz[item, z], sigma2_iz[item, z])*p_z[user, z]
                        p_z_given_uri[user, item, z] = nom/denom #p(z|user, rating, item;thetahat)
                ratings.apply(helper, 1)
            # M-step
            if self.verbose:
                print('M-step...')
            for u in range(self.trainset.n_users):
                denom = np.sum(p_z_given_uri[u]) # sum_z sum_u P(z|u, v, y;hattheta)
                for z in range(self.n_latent):
                    p_z[u,z] = np.sum(p_z_given_uri[u][:,z])/denom # sum of P(z|user, rating, item; thetahat)
            for i in range(self.trainset.n_items):
                for z in range(self.n_latent):
                    denom = np.sum(p_z_given_uri[:,i,z]) #sum_y  {(z|user, rating, item)}
                    item_rating = rating_mat[:,i]
                    mu_iz[i,z] = np.dot(p_z_given_uri[:,i,z], item_rating)/denom
                    sigma2_iz[i,z] = np.dot(np.square(item_rating - mu_iz[i,z]), p_z_given_uri[:,i,z])/denom

        # Compute prediction
        pred = np.nan_to_num(p_z)@np.nan_to_num(mu_iz.T)

        if self.to_normalize:
            pred = np.add(np.multiply(pred, uservar[:,np.newaxis]), usermu[:,np.newaxis])

        # Compute SVD of prediction
        self.pred_matrix = self.svd(ratings, pred)

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
