import numpy as np

def svd(X, k=50):
    '''
    Performs SVD on the covariance matrix of the given matrix.

    Parameters:
    X (numpy.ndarray): the data matrix
    k (int): the number of dimensions to select for reconstruction. By default 50

    Returns:
    X_pred (numpy.ndarray): the reconstructed matrix
    '''

    # Compute SVD of X
    U, S, Vt = np.linalg.svd(X)
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
    U_k = U[:,:k]
    V_k = V[:,:k]

    # Reconstruct matrix
    X_pred = U_k.dot(V_k.T)

    return X_pred


def svd_funk(X, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
    '''
    Performs SVD as popularized by Simon Funk. The goal of obtaining matrix P, Q
    such that X = P*t(Q).
    The object function is the following:

    H(P, Q) = (X[u,i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2)

    Parameters:
    X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
    k (int): the number of latent features. By default 50
    l (float): the strenght of the regularizer. By default 1
    eta (float): the learning rate. By default 0.01
    batch_size (int): the number of samples to be used in the SGD step. By default 50
    epochs (int): the number of iterations. By default 1000

    Returns:
    X_pred (numpy.ndarray): the reconstructed matrix
    '''

    # Initialize P, Q
    P = np.random.rand(X.shape[0],k)
    Q = np.random.rand(X.shape[1],k)

    # Extract non zero entries
    users, items = X.nonzero()
    observed = tuple(zip(users, items))

    for epoch in range(epochs):
        indexes = np.random.randint(low=0, high=len(observed), size=batch_size)
        for index in indexes:
            # Extract index
            u, i = observed[index]
            # Local variables
            pu, qi = U[u,:], Z[i,:]
            error = X[u,i] - pu.dot(qi)
            # Update step
            P[u,:] = P[u,:] + eta*(delta*qi - l*pu)
            Q[i,:] = Q[i,:] + eta*(delta*pu - l*qi)

    # Reconstruct matrix
    X_pred = P.dot(Q.T)

    return X_pred

def svd_biased(X, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
    '''
    Performs a biased version of SVD with the goal of obtaining matrix P, Q.
    such that X = U*t(Z).
    The object function is the following:

    H(P, Q) = (X[u,i] - mu - b[u] - b[i] - p[u]*q[i])^2 + l*(||p[u]||^2 + ||q[i]||^2 + b[u]^2 + b[i]^2)

    where mu is the global (rating) average, b[u] and b[i] are the users and items biases respectively.

    Parameters:
    X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
    k (int): the number of latent features. By default 50
    l (float): the strenght of the regularizer. By default 1
    eta (float): the learning rate. By default 0.01
    batch_size (int): the number of samples to be used in the SGD step. By default 50
    epochs (int): the number of iterations. By default 1000

    Returns:
    X_pred (numpy.ndarray): the reconstructed matrix
    '''

    # Initialize P, Q
    P = np.random.rand(X.shape[0],k)
    Q = np.random.rand(X.shape[1],k)

    # Initialize biases
    bias_u = np.zeros(X.shape[0])
    bias_i = np.zeros(X.shape[1])

    # Extract non zero entries
    users, items = X.nonzero()
    observed = tuple(zip(users, items))

    # Compute global average
    mu = X.mean()

    for epoch in range(epochs):
        indexes = np.random.randint(low=0, high=len(observed), size=batch_size)
        for index in indexes:
            # Extract indexes
            u, i = observed[index]
            # Local variables
            pu, qi = P[u,:], Q[i,:]
            bu, bi = bias_u[u], bias_i[i]
            # Compute prediction error
            error = X[u,i] - (pu.dot(qi) + bd + bn + mu)
            # Update step
            P[u,:] = P[u,:] + eta*(error*qi - l*pu)
            Q[i,:] = Q[i,:] + eta*(error*pu - l*qi)
            bias_u = bias_u + eta*(error - l*bu)
            bias_i = bias_i + eta*(error - l*bi)

    # Reconstruct matrix
    X_pred = P.dot(Q.T)

    return X_pred

def svd_pp(X, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
    '''
    Performs the SVD++ algorithm based on Koren's paper. The goal is to find matrices P, Q
    such that X = P*t(Q).
    The object function is the following:

    H(P, Q) = (X[u,i] - mu - b[u] - b[i] - q[i]*(p[u] + 1/sqrt(|N(u)|)*sum(y[j])))^2 +
            + l*(b[u]^2 + b[i]^2 + ||p[u]||^2 + ||q[i]||^2 + sum(||y[j]||^2))

    where mu is the global (rating) average, b[u] and b[i] are the users and items biases respectively,
    N(u) represent the items rated by user u, and each y[j] represent one item factor.

    Parameters:
    X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
    k (int): the number of latent features. By default 50
    l (float): the strenght of the regularizer. By default 1
    eta (float): the learning rate. By default 0.01
    batch_size (int): the number of samples to be used by SGD. By default 50
    epochs (int): the number of iterations. By default 1000

    Returns:
    X_pred (numpy.ndarray): the reconstructed matrix
    '''

    # Initialize P, Q
    P = np.random.rand(X.shape[0],k)
    Q = np.random.rand(X.shape[1],k)

    # Initialize biases
    bias_u = np.zeros(X.shape[0])
    bias_i = np.zeros(X.shape[1])

    # Initialize item factors
    Y = np.random.normal(0, 1, (X.shape[1],k))

    # Extract non zero entries
    users, items = X.nonzero()
    observed = tuple(zip(users, items))

    # Compute global average
    mu = X.mean()

    for epoch in range(epochs):
        indexes = np.random.randint(low=0, high=len(observed), size=batch_size)
        for index in indexes:
            # Extract indexes
            u, i = observed[index]
            # Local variables
            pu, qi = P[u,:], Q[i,:]
            bu, bi = bias_u[u], bias_i[i]
            # Items rated by user u
            nu = X[u,:].nonzero()[1] # dimension 1 because we need the items indexes
            sqrt_len_nu = np.sqrt(len(nu))
            # Initialize implicit feedback vector
            i_feed = np.zeros(k)
            # Build implicit feedback vector
            for j in nu:
                i_feed += Y[j,:]/sqrt_len_nu # done so for stability
            # Compute prediction error
            error = X[u,i] - (qi.dot(pu + i_feed) + bu + bi + mu)
            # Update step
            P[u,:] = P[u,:] + eta*(error*qi - l*pu)
            Q[i,:] = Q[i,:] + eta*(error*(pu + i_feed) - l*qi)
            bias_u = bias_u + eta*(error - l*bu)
            bias_i = bias_i + eta*(error - l*bi)
            for j in nu:
                Y[j,:] = Y[j,:] + eta*(error*(qi/sqrt_len_nu) - l*Y[j,:])

    # Reconstruct matrix
    X_pred = P.dot(Q.T)

    return X_pred
