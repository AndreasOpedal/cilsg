# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
get_ipython().system('pip install --user scikit-surprise')
from surprise import AlgoBase, PredictionImpossible, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

import pickle

# %%
data_train_raw = pd.read_csv('../submission/data/data-train.csv')

# parse rows and columns
row_str = data_train_raw['Id'].apply(lambda x: x.split('_')[0])
row_id = row_str.apply(lambda x: int(x.split('r')[1]) - 1)
col_str = data_train_raw['Id'].apply(lambda x: x.split('_')[1])
col_id = col_str.apply(lambda x: int(x.split('c')[1]) - 1)

# apply changes
data_train_raw['row'] = row_id
data_train_raw['col'] = col_id

# dataset as data frame
data_train_df = data_train_raw.loc[:,['row', 'col', 'Prediction']]


# %%
# set up surprise dataset
reader = Reader()
dataset = Dataset.load_from_df(data_train_df[['row', 'col', 'Prediction']], reader)

# now set up training and test set, with a test split of 25%
trainset, testset = train_test_split(dataset, test_size=0.25)


# %%
def sum_column_norm_square(M):
    return np.sum(np.square(np.linalg.norm(M, axis = 0)))

def als_objective(A, P, Q, lamb):
    """
    Least squares with regularization

    A: np.array, target matrix
    """
    observed_id = np.nonzero(A)
    least_squares = np.sum(np.square(A - P.T@Q)[observed_id]) #extract observed values
    regularization = lamb*(sum_column_norm_square(P)+ sum_column_norm_square(Q))
    error = (least_squares + regularization)/observed_id[0].shape[0]
    return error 

def als(trainset, k, lamb, tol, max_iter):
    """
    """
    m, n = trainset.n_users, trainset.n_items
    # Get rating matrix
    A = np.zeros((m, n))
    for u, i, r in trainset.all_ratings():
        A[u,i] = r
    # Initialize P, Q
    P = np.ones((k,m)) #user matrix
    Q = np.ones((k,n)) #item matrix
    Id_k = np.eye(N=k)
    # Alternate to optimize
    num_iter = 0
    while num_iter < max_iter:
        ls_error = als_objective(A, P, Q, lamb)
        if num_iter % 10 == 0:
            print("Iter  {}:  error {}".format(num_iter, ls_error))
        for u in range(m):
            obs_items = np.nonzero(A[u, :])
            sum_qqT = np.sum([Q[:, i]*np.reshape(Q[:, i], (-1, 1)) for i in obs_items[0]], 0)
            P[:, u] = np.squeeze(np.linalg.inv(sum_qqT+lamb*Id_k) @ np.reshape(np.sum([A[u,item]*Q[:, item] for item in obs_items[0]], 0), (-1, 1)))
        for i in range(n):
            obs_users = np.nonzero(A[:, i])
            sum_ppT = np.sum([P[:, u]*np.reshape(P[:, u], (-1, 1)) for u in obs_users[0]], 0)
            Q[:, i] = np.squeeze(np.linalg.inv(sum_ppT+lamb*Id_k) @ np.reshape(np.sum([A[u, i]*P[:, u] for u in obs_users[0]], 0), (-1, 1)))
        if ls_error <= tol:
            break
        num_iter += 1
    return P, Q

class ALS(AlgoBase):
    def __init__(self,k = 10, lamb = 0.1, tol  = 0.001):
        self.k = k
        self.lamb = lamb
        self.tol = tol
        self.max_iter = 50
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.P, self.Q = als(self.trainset, self.k, self.lamb, self.tol, self.max_iter)
                
    def estimate(self, u, i):
        return np.clip(np.dot(self.P[:, u], self.Q[:, i]), 1, 5)


# %%
model = GridSearchCV(ALS, param_grid = {"k":[5, 10, 20, 35, 55, 80], "lamb":[0.1, 0.3, 0.5, 0.7], "tol":[0.25, 0.1, 0.05]})

model.fit(dataset)

with open("./als.pkl") as f:
    pickle.dump([model.best_score, model.best_params], f)

