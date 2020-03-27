import numpy as np

def als(X, k=50, l=1, epochs=1000):
    '''
    Performs alternating least squares with the goal of recreating the given matrix.

    Parameters:
    X (scipy.sparse.dok_matrix): the data matrix, which should not have been imputed
    k (int): the number of latent features. By default 50
    l (float): the strenght of the regularizer. By default 1
    epochs (int): the number of iterations. By default 1000

    Returns:
    (U, V) (numpy.ndarray, numpy.ndarray): matrices U and V such that X = UV (approximately)
    '''

    # Data matrix dimensions
    m, n = X.shape[0], X.shape[1]

    # Create matrices U, V, and randomly initalize them
    U = np.random.rand(m, k)
    V = np.random.rand(k, n)

    # Get X non-zero indices
    nz_indexes = X.nonzero()
    nzi, nzj = nz_indexes[0], nz_indexes[1]

    # Identity matrix time lambda
    Il = l*np.identity(k)

    # Begin loop
    for epoch in range(epochs):

        # Optimize U
        for i in range(m):
            Ml = np.zeros((k, k))
            vr = np.zeros((k,))
            for j in nzj:
                Ml += V[:,j].dot(V[:,j].T) + Il
                vr += V[:,j]*X[i,j]
            U[i,:] = np.linalg.inv(V[:,j].dot(V[:,j].T) + Il).dot(V[:,j]*X[i,j])

        # Optimize V
        for j in range(n):
            Ml = np.zeros((k, k))
            vr = np.zeros((k,))
            for i in nzi:
                Ml += U[i,:].dot(U[i,:].T) + Il
                vr += U[i,:]*X[i,j]
            V[:,j] = np.linalg.inv(U[i,:].dot(U[i,:].T) + Il).dot(U[i,:]*X[i,j])

    return U, V
