# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
#get_ipython().system('pip install --user scikit-surprise')
from surprise import AlgoBase, PredictionImpossible, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

import pickle
import time

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

def compute_rmse(target, pred):
    mse = np.mean(np.square(target-pred))
    return np.sqrt(mse)

def als(trainset, k, lamb, max_iter):
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
    best_PQ = [None, None, None]
    obs_idx = np.nonzero(A)
    target = A[obs_idx]
    num_rmse_incr = 0
    while num_iter < max_iter:
        # train rmse
        pred =(P.T@Q)[obs_idx]
        last_train_rmse = compute_rmse(target, pred)
        #ls_error = als_objective(A, P, Q, lamb)
       
        for u in range(m):
            obs_items = np.nonzero(A[u, :])
            num_obs_items = obs_items[0].shape[0]
            sum_qqT = np.sum([Q[:, i]*np.reshape(Q[:, i], (-1, 1)) for i in obs_items[0]], 0)
            P[:, u] = np.squeeze(np.linalg.inv(sum_qqT+lamb*num_obs_items*Id_k) @ np.reshape(np.sum([A[u,item]*Q[:, item] for item in obs_items[0]], 0), (-1, 1)))
        for i in range(n):
            obs_users = np.nonzero(A[:, i])
            num_obs_users = obs_users[0].shape[0]
            sum_ppT = np.sum([P[:, u]*np.reshape(P[:, u], (-1, 1)) for u in obs_users[0]], 0)
            Q[:, i] = np.squeeze(np.linalg.inv(sum_ppT+lamb*num_obs_users*Id_k) @ np.reshape(np.sum([A[u, i]*P[:, u] for u in obs_users[0]], 0), (-1, 1)))
        best_PQ[0],  best_PQ[1], best_PQ[2] = best_PQ[1], best_PQ[2], (P, Q)
        pred = (P.T@Q)[obs_idx]
        train_rmse = compute_rmse(target, pred)        
        if (num_iter+1) % 5 == 0:
            print("Iter  {}: rmse error {}".format(num_iter+1, train_rmse))
        if train_rmse - last_train_rmse >= 0:
            num_rmse_incr += 1
        else:
            num_rmse_incr =  0
        if num_rmse_incr >=  2:
            print("RMSE stops decreasing at iter {}".format(num_iter-1))
            return  best_PQ[0]
        num_iter += 1
    P, Q = best_PQ[2]
    return P, Q

class ALS(AlgoBase):
    def __init__(self,k, lamb, max_iter):
        self.k = k
        self.lamb = lamb
        #self.tol = tol
        self.max_iter = max_iter
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        start = time.time()
        self.P, self.Q = als(self.trainset, self.k, self.lamb, self.max_iter)
        print("Used time: {}".format(time.time()-start))        
    def estimate(self, u, i):
        return np.clip(np.dot(self.P[:, u], self.Q[:, i]), 1, 5)

def read_submission_indexes(file_path):
    '''
    Reads the indexes for which to make the predictions.

    Parameters:
    file_path (string): the file path to read

    Returns:
    indexes (list): a list where each element is a tuple of the form (row, column)
    '''

    data_train_raw = pd.read_csv(file_path).values

    tuples = []

    for i in range(data_train_raw.shape[0]):
        indices, value = data_train_raw[i,0], int(data_train_raw[i,1])
        indices = indices.split('_')
        row, column = int(indices[0][1:])-1, int(indices[1][1:])-1 # indexing in the data starts from 1
        tuples.append((row, column))

    return tuples

def write_predictions_to_csv(predictions, file_path):
    '''
    Writes on a csv file the predictions. The format of the prediction file is the following (second row is example):

    Id, Prediction
    r1_c1, 1

    Parameters:
    predictions (surprise.prediction_algorithms.predictions.Prediction): the object holding the predictions
    file_path (string): the path to the prediction file
    '''

    # Define header
    header = []
    header.append('Id')
    header.append('Prediction')

    # Write file
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for row, col, pred in predictions:
            # Row of the csv file
            file_row = {}
            # Build file row
            file_row['Id'] = 'r'+ str(row+1) + '_c' + str(col+1)
            file_row['Prediction'] = pred
            # Write file row
            writer.writerow(file_row)
# %%
if __name__  == "__main__":
    from datetime import datetime
    import csv
    
    is_refit = True

    if is_refit:
        pred_ids = read_submission_indexes("../submission/data/sample-submission.csv")
	best_model = "./als_20200708_1846.pkl"
        best_params = pickle.load(open(best_model, "rb"))[1]['rmse']
        k, lamb, max_iter = best_params['k'], best_params['lamb'], best_params['max_iter']
        model = ALS(k=k, lamb=lamb, max_iter=max_iter)
        # Fit on all training data
        model.fit(dataset.build_full_trainset())
        submission = []
        for r, c in pred_ids:
            submission.append((r, c, model.estimate(r, c)))
        submission_csv = "../submission/predictions/submission_als_{}.csv".format(datetime.now().strftime('%Y%m%d_%H%M'))
        write_predictions_to_csv(submission, submission_csv)
    else:
        model = GridSearchCV(ALS, param_grid = {"k":[3, 5, 10, 20], "lamb":[0.05, 0.1, 0.5, 1], "max_iter":[10,30,50,100]}, cv = 10, n_jobs = -1, return_train_measures=True)

        model.fit(dataset)
        with open("./als_{}.pkl".format(datetime.now().strftime('%Y%m%d_%H%M')), "wb") as f:
            pickle.dump([model.best_score, model.best_params, model], f)

