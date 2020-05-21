import numpy as np
from tqdm.auto import tqdm

class SVD:
    '''
    This class implements the basic SVD algorithm.
    '''

    def __init__(self, k=50):
        '''
        Initializes the class with the given parameters.

        Parameters:
        k (int): the number of dimensions to select for reconstruction. By default 50
        '''

        self.k = k
        self.U_k = None
        self.V_k = None
        self.X = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]

    def fit(self, X):
        '''
        Computes the SVD composition of the given matrix.

        Parameters:
        X (numpy.ndarray): the data matrix
        '''

        # Read X
        self.X = X

        # Compute SVD of X
        U, S, Vt = np.linalg.svd(self.X)
        D = np.zeros(shape=(S.shape[0], S.shape[0])) # create diagonal matrix D
        np.fill_diagonal(D, S) # fill D with S

        # Square root of D
        D = np.sqrt(D)

        # Pad D
        D_p = np.append(D, np.zeros((U.shape[0]-D.shape[0], D.shape[0])), axis=0)

        # Scale U, Vt
        U = U.dot(D_p)
        V = D.dot(Vt.T)

        # Select vectors from U, V
        self.U_k = U[:,:self.k]
        self.V_k = V[:,:self.k]

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices U, V

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Reconstruct matrix
        X_pred = self.U_k.dot(self.V_k.T)

        return X_pred

class ALS:
    '''
    This class implements an element-wise version of the alternating least squares (ALS) algorithm.
    '''

    def __init__(self, k=50, l=1, epochs=1000):
        '''
        Initializes the class with the given parameters.

        Parameters:
        k (int): the number of latent features. By default 50
        l (float): the strenght of the regularizer. By default 1
        epochs (int): the number of iterations. By default 1000
        '''

        self.k = k
        self.l = l
        self.epochs = epochs
        self.X = None
        self.P = None
        self.Q = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]
        self.l = params[1]
        self.epochs = params[2]

    def fit(self, X):
        '''
        Finds matrices P, Q by optimizing the following objective function (element-wise):

        H(P, Q)[u,i] = (X[u,i] - t(P[u])*Q[i])^2 + l*(||P[u]||^2 + ||Q[i]||^2)

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        '''

        # Read X
        self.X = X

        # Create matrices U, V, and randomly initalize them
        self.P = np.random.rand(self.X.shape[0], self.k)
        self.Q = np.random.rand(self.k, self.X.shape[1])

        # Extract non zero entries
        users, items = self.X.nonzero()
        observed = tuple(zip(users, items))

        # Begin loop
        for epoch in tqdm(range(self.epochs)):
            # Get indexes
            uss, iss = observed[0], observed[1]
            # Update P
            for u in uss:
                for f in range(self.k):
                    nominator, denominator = 0, 0
                    for i in iss:
                        rui = self.P[u,:].T.dot(self.Q[:,i])
                        nominator = nominator + (self.X[u,i] - (rui - self.P[u,f]*self.Q[f,i]))*self.Q[f,i]
                        denominator = denominator + self.Q[f,i]**2*self.l
                    self.P[u,f] = nominator/denominator
            # Update Q
            for i in iss:
                for f in range(self.k):
                    nominator, denominator = 0, 0
                    for u in uss:
                        rui = self.P[u,:].T.dot(self.Q[:,i])
                        nominator = nominator + (self.X[u,i] - (rui - self.P[u,f]*self.Q[f,i]))*self.P[u,f]
                        denominator = denominator + self.P[u,f]**2*self.l
                    self.Q[f,i] = nominator/denominator

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices P, Q

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Reconstruct matrix
        X_pred = self.P.dot(self.Q)

        return X_pred

class SVDFunk:
    '''
    This class implements the SVD variant popularized by Simon Funk.
    '''

    def __init__(self, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
        '''
        Initializes the class with the given parameters.

        Parameters:
        k (int): the number of latent features. By default 50
        l (float): the strenght of the regularizer. By default 1
        eta (float): the learning rate. By default 0.01
        batch_size (int): the number of samples to be used in the SGD step. By default 50
        epochs (int): the number of iterations. By default 1000
        '''

        self.k = k
        self.l = l
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = None
        self.P = None
        self.Q = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]
        self.l = params[1]
        self.eta = params[2]
        self.batch_size = params[3]
        self.epochs = params[4]

    def fit(self, X):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P, Q) = (X[u,i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2)

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        '''

        # Read X
        self.X = X

        # Initialize P, Q
        self.P = np.random.rand(self.X.shape[0],self.k)
        self.Q = np.random.rand(self.X.shape[1],self.k)

        # Extract non zero entries
        users, items = self.X.nonzero()
        observed = tuple(zip(users, items))

        for epoch in tqdm(range(self.epochs)):
            indexes = np.random.randint(low=0, high=len(observed), size=self.batch_size)
            for index in indexes:
                # Extract index
                u, i = observed[index]
                # Local variables
                pu, qi = self.P[u,:], self.Q[i,:]
                error = self.X[u,i] - pu.dot(qi)
                # Update step
                self.P[u,:] = self.P[u,:] + self.eta*(error*qi - self.l*pu)
                self.Q[i,:] = self.Q[i,:] + self.eta*(error*pu - self.l*qi)

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices P, Q

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Reconstruct matrix
        X_pred = self.P.dot(self.Q.T)

        return X_pred

class SVDBiased:
    '''
    This class implements a biased version of Funk's SVD.
    '''

    def __init__(self, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
        '''
        Initializes the class with the given parameters.

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        k (int): the number of latent features. By default 50
        l (float): the strenght of the regularizer. By default 1
        eta (float): the learning rate. By default 0.01
        batch_size (int): the number of samples to be used in the SGD step. By default 50
        epochs (int): the number of iterations. By default 1000
        '''

        self.k = k
        self.l = l
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = None
        self.P = None
        self.Q = None
        self.mu = None
        self.bias_u = None
        self.bias_i = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]
        self.l = params[1]
        self.eta = params[2]
        self.batch_size = params[3]
        self.epochs = params[4]

    def fit(self, X):
        '''
        Finds matrices P, Q and biases by optimizing the following objective function:

        H(P, Q) = (X[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2 + b[u]^2 + b[i]^2)

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        '''

        # Read X
        self.X = X

        # Initialize P, Q
        self.P = np.random.rand(self.X.shape[0],self.k)
        self.Q = np.random.rand(self.X.shape[1],self.k)

        # Initialize biases
        self.bias_u = np.zeros(self.X.shape[0])
        self.bias_i = np.zeros(self.X.shape[1])

        # Extract non zero entries
        users, items = self.X.nonzero()
        observed = tuple(zip(users, items))

        # Compute global average
        self.mu = self.X.mean()

        for epoch in tqdm(range(self.epochs)):
            indexes = np.random.randint(low=0, high=len(observed), size=self.batch_size)
            for index in indexes:
                # Extract indexes
                u, i = observed[index]
                # Local variables
                pu, qi = self.P[u,:], self.Q[i,:]
                bu, bi = self.bias_u[u], self.bias_i[i]
                # Compute prediction error
                error = self.X[u,i] - (pu.dot(qi) + bu + bi + self.mu)
                # Update step
                self.P[u,:] = self.P[u,:] + self.eta*(error*qi - self.l*pu)
                self.Q[i,:] = self.Q[i,:] + self.eta*(error*pu - self.l*qi)
                self.bias_u = self.bias_u + self.eta*(error - self.l*bu)
                self.bias_i = self.bias_i + self.eta*(error - self.l*bi)

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices P, Q and the biases

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Reconstruct matrix
        X_pred = self.P.dot(self.Q.T)

        # Add global mean
        X_pred += self.mu

        # Add biases
        for u in range(X_pred.shape[0]):
            X_pred[u,:] += self.bias_i
        for i in range(X_pred.shape[1]):
            X_pred[:,i] += self.bias_u

        return X_pred

class SVDPP:
    '''
    This class implements the SVD++ algorithm, as described in Koren's paper.
    '''

    def __init__(self, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
        '''
        Initializes the class with the given parameters.

        Parameters:
        k (int): the number of latent features. By default 50
        l (float): the strenght of the regularizer. By default 1
        eta (float): the learning rate. By default 0.01
        batch_size (int): the number of samples to be used in the SGD step. By default 50
        epochs (int): the number of iterations. By default 1000
        '''

        self.k = k
        self.l = l
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.X = None
        self.P = None
        self.Q = None
        self.mu = None
        self.bias_u = None
        self.bias_i = None
        self.Y = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]
        self.l = params[1]
        self.eta = params[2]
        self.batch_size = params[3]
        self.epochs = params[4]

    def fit(self, X):
        '''
        Finds matrices P, Q, biases, and item factors by optimizing the following objective function:

        H(P, Q) = (X[u,i] - mu - b[u] - b[i] - q[i]*(p[u] + 1/sqrt(|N(u)|)*sum(y[j])))^2 +
                + l*(b[u]^2 + b[i]^2 + ||p[u]||^2 + ||q[i]||^2 + sum(||y[j]||^2))

        where mu is the global (rating) average, b[u] and b[i] are the users and items biases respectively,
        N(u) represent the items rated by user u, and each y[j] represent one item factor.

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        '''

        # Read X
        self.X = X

        # Initialize P, Q
        self.P = np.random.rand(self.X.shape[0],self.k)
        self.Q = np.random.rand(self.X.shape[1],self.k)

        # Initialize biases
        self.bias_u = np.zeros(self.X.shape[0])
        self.bias_i = np.zeros(self.X.shape[1])

        # Initialize item factors
        self.Y = np.random.normal(0, 1, (self.X.shape[1],self.k))

        # Compute global average
        self.mu = self.X.mean()

        # Extract non zero entries
        users, items = self.X.nonzero()
        observed = tuple(zip(users, items))

        for epoch in tqdm(range(self.epochs)):
            indexes = np.random.randint(low=0, high=len(observed), size=self.batch_size)
            for index in indexes:
                # Extract indexes
                u, i = observed[index]
                # Local variables
                pu, qi = self.P[u,:], self.Q[i,:]
                bu, bi = self.bias_u[u], self.bias_i[i]
                # Items rated by user u
                nu = self.X[u,:].nonzero()[1] # dimension 1 because we need the items indexes
                sqrt_len_nu = np.sqrt(len(nu))
                # Initialize implicit feedback vector
                ifbv = np.zeros(self.k)
                # Build implicit feedback vector
                for j in nu:
                    ifbv += self.Y[j,:]/sqrt_len_nu # done so for stability
                # Compute prediction error
                error = self.X[u,i] - (qi.dot(pu + ifbv) + bu + bi + self.mu)
                # Update step
                self.P[u,:] = self.P[u,:] + self.eta*(error*qi - self.l*pu)
                self.Q[i,:] = self.Q[i,:] + self.eta*(error*(pu + ifbv) - self.l*qi)
                self.bias_u = self.bias_u + self.eta*(error - self.l*bu)
                self.bias_i = self.bias_i + self.eta*(error - self.l*bi)
                for j in nu:
                    self.Y[j,:] = self.Y[j,:] + self.eta*(error*(qi/sqrt_len_nu) - self.l*self.Y[j,:])

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices P, Q, the biases, and the
        item factors.

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Add item factors to P
        for u in range(self.X.shape[0]):
            nu = self.X[u,:].nonzero()[1]
            sqrt_len_nu = np.sqrt(len(nu))
            ifbv = np.zeros(self.k)
            # Compute feedback for user
            for j in nu:
                ifbv += self.Y[j,:]/sqrt_len_nu
            # Update prediction matrix at u
            self.P[u,:] += ifbv

        # Reconstruct matrix
        X_pred = self.P.dot(self.Q.T)

        # Add global mean
        X_pred += self.mu

        # Add biases
        for u in range(X_pred.shape[0]):
            X_pred[u,:] += self.bias_i
        for i in range(X_pred.shape[1]):
            X_pred[:,i] += self.bias_u

        return X_pred

class SVDBox:
    '''
    An implementation of a box-constrained version of SVD.
    Because each prediction value must be within a certain range, each vectors P[u,:], Q[i,:] should be within a certain range.
    '''

    def __init__(self, k=50, l=1, eta=0.01, batch_size=50, epochs=1000, low=1, high=5):
        '''
        Initializes the class with the given parameters.

        Parameters:
        k (int): the number of latent features. By default 50
        l (float): the strenght of the regularizer. By default 1
        eta (float): the learning rate. By default 0.01
        batch_size (int): the number of samples to be used in the SGD step. By default 50
        epochs (int): the number of iterations. By default 1000
        low (int): the lower bound for a prediction. By default 1
        high (int): the upper bound for a prediction. By default 5
        '''

        self.k = k
        self.l = l
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.low = low
        self.high = high
        self.X = None
        self.P = None
        self.Q = None

    def set_params(self, params):
        '''
        Set the model's parameters by reading the given list of parameters.

        Parameters:
        params (list): the list of parameters. Note that the list's length must be equal to the number of parameters, and
                       the ordering must also be the same
        '''

        self.k = params[0]
        self.l = params[1]
        self.eta = params[2]
        self.batch_size = params[3]
        self.epochs = params[4]
        self.low = params[5]
        self.high = params[6]

    def fit(self, X):
        '''
        Finds matrices P, Q by optimizing the following objective function:

        H(P, Q) = (X[u,i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2)

        Parameters:
        X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
        '''

        # Read X
        self.X = X

        # Maximum and minimum entries for P, Q elements
        min_entry = np.sqrt(self.low/self.k)
        max_entry = np.sqrt(self.high/self.k)

        # Initialize P, Q (while keeping entries in range)
        self.P = np.random.uniform(min_entry, max_entry, (self.X.shape[0],self.k))
        self.Q = np.random.uniform(min_entry, max_entry, (self.X.shape[1],self.k))

        # Extract non zero entries
        users, items = self.X.nonzero()
        observed = tuple(zip(users, items))

        for epoch in tqdm(range(self.epochs)):
            indexes = np.random.randint(low=0, high=len(observed), size=self.batch_size)
            for index in indexes:
                # Extract index
                u, i = observed[index]
                # Local variables
                pu, qi = self.P[u,:], self.Q[i,:]
                error = self.X[u,i] - pu.dot(qi)
                # Update step
                self.P[u,:] = self.P[u,:] + self.eta*(error*qi - self.l*pu)
                self.Q[i,:] = self.Q[i,:] + self.eta*(error*pu - self.l*qi)
                # Clip the vectors
                np.clip(self.P[u,:], min_entry, max_entry)
                np.clip(self.Q[i,:], min_entry, max_entry)

    def transform(self):
        '''
        Computes the prediction matrix based on the computed matrices P, Q

        Returns:
        X_pred (numpy.ndarray): the reconstructed matrix
        '''

        # Reconstruct matrix
        X_pred = self.P.dot(self.Q.T)

        return X_pred
