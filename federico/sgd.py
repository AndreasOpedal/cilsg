import numpy as np

def sgd(X, k=50, l=1, eta=0.01, epochs=1000):
    '''
    Performs stochastic gradient descent on with the goal of obtaining matrix U, Z
    such that X = U*t(Z).

    Parameters:
    X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
    k (int): the number of latent features. By default 50
    l (float): the strenght of the regularizer. By default 1
    eta (float): the learning rate. By default 0.01
    epochs (int): the number of iterations. By default 1000

    Returns:
    U, Z (numpy.ndarray, numpy.ndarray): matrices U and Z such that X = U*t(Z) (approximately)
    '''

    # Initialize U, Z
    U = np.random.rand(X.shape[0], k)
    Z = np.random.rand(X.shape[1], k)

    # Extract non zero entries
    users, movies = X.nonzero()

    for epoch in range(epochs):
        print(epoch)
        for d, n in zip(users, movies):
            # Local variables
            delta = X[d,n] - (U.dot(Z.T))[d,n]
            Ud, Zn = U[d,:], Z[n,:]
            # Update step
            U[d,:] -= eta*(-delta*Zn + l*Ud)
            Z[n,:] -= eta*(-delta*Ud + l*Zn)

    return U, Z
