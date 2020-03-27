import numpy as np
import matplotlib.pylab as plt

def svd(X, k=2, plot=False):
    '''
    Performs SVD on the covariance matrix of the given matrix, and set ups matrices for reconstructing the input matrix.

    Parameters:
    X (numpy.ndarray): the data matrix
    k (int): the number of dimensions to select for reconstruction. By default 2
    plot (boolean): whether to plot the singular values. By default False

    Returns:
    U, V (numpy.ndarray, numpy.ndarray): matrices of eigenvectors, properly redimensionalized
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

    if plot == True:
        plt.plot(np.log(S**2))
        plt.show()

    # Select vectors from U, V
    U_k = U[:,:k]
    V_k = V[:,:k]

    return U_k, V_k

def reconstruct(U, V):
    '''
    Reconstructs matrix X based on eigenvectors matrices U and V.
    Note that matrices U, V must be passed in the same order as returned by function svd.

    Parameters:
    (U, V) (numpy.ndarray, numpy.ndarray): the matrices of eigenvalues

    Returns:
    X (np.ndarray): reconstructed matrix
    '''

    X = U.dot(V.T)

    return X
