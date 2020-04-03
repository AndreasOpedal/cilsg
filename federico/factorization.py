import numpy as np

def svd_funk(X, k=50, l=1, eta=0.01, batch_size=50, epochs=1000):
    '''
    Performs the SVD version popularized by Simon Funk, with the goal of obtaining matrix P, Q
    such that X = P*t(Q).

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
    such that X = U*t(Z). The prediction for rating R[u, i] is

    R[u,i] = P[u,:]*Q[i,:] + bias[u] + bias[i]

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
