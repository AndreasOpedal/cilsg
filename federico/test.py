import numpy as np
import utils
import preprocess
import model_selection
import svd
import als
import sgd

X = utils.read_data_as_matrix('../data/data_train.csv')

print('Done reading')

# tups = utils.read_data_as_tuples('../data/data_train.csv')
# print(tups[0])

# X_train = preprocess.knn_impute(X, n_neighbors=5)
#
# print('Done imputing')

# U, V = svd.svd(X_train)
# X_rec = svd.reconstruct(U, V)
# print(np.linalg.norm(X_train-X_rec))

# U, V = als.als(X, iters=3)
# print(np.linalg.norm(X-U.dot(V)))

# X_train, X_test = model_selection.train_test_split(X)
#
# print('Done split')
#
U, Z = sgd.sgd(X, epochs=1000)
print(np.linalg.norm(X-U.dot(Z.T)))
