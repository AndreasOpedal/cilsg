import numpy as np
import utils
import preprocess
import model_selection
import factorization
import metrics

np.random.seed(42)

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

X_train, X_test = model_selection.train_test_split(X, min_user_ratings=2, test_movies_pct=0.5)
print('Done split')
print(len(X_train.nonzero()[0]))
print(len(X_test.nonzero()[0]))

X_pred = factorization.svd_pp(X_train, epochs=1500, eta=0.01, batch_size=50, l=1)
print(metrics.rmse(X_test, X_pred))
