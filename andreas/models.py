import numpy as np
import pandas as pd

def svd_mean_recommender(data, param):
    
	num_eigen = param
	# 1. Compute data matrix with imputed means

	# Initialize empty
	data_matrix = np.zeros([10000,1000])
	data_matrix[:] = np.nan

	# Fill with observed values
	for i in np.arange(data.shape[0]):
		row_ind = data.iloc[i,0]
		col_ind = data.iloc[i,1]
		pred = data.iloc[i,2]
		data_matrix[row_ind, col_ind] = pred

	# Fill with column means
	item_means = np.apply_along_axis(np.nanmean, 0, data_matrix)
	for i in np.arange(data_matrix.shape[1]):
		data_matrix[np.isnan(data_matrix[:,i]), i] = item_means[i]

	# 2. Perform SVD (this is exercise 2.2 of series 2)

	u, s, vh = np.linalg.svd(data_matrix)

	s_diag = np.concatenate((np.diag(s), np.zeros((9000,1000))), axis = 0) # Represent singular values in matrix
	u_prime = np.matmul(u,np.sqrt(s_diag))
	v_prime = np.matmul(np.sqrt(s_diag), vh.T) # transpose since output vh is V^T

	# Compute prediction matrix. NOTE: currently not using square root matrices (prime) as recommended in exercise.
	# Tried this when first doing the exercise, got strange values and don't fully understand it. 
	predictions = np.matmul(u[:,0:num_eigen], np.matmul(s_diag[0:num_eigen,0:num_eigen], vh[0:num_eigen,:]))

	# Note that this is still complete matrix representation
	return predictions