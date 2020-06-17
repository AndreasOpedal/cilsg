import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime

def rating_gaussian_model(rating, mean, var):
    """
    Gaussian distribution density function

    Param
    =========
    rating:
    mean: 
    
    Return  
    =========
    """
    return np.exp(-(rating-mean)**2/(2*var))/(np.sqrt(2*np.pi*var))

def rating_laplacian_model(rating, mean, var):
    b = np.sqrt(var/2)
    return (np.exp(-np.abs(rating-mean)/b))/(2*b)

def df_to_mat(ratings, num_row, num_col):
    result = np.zeros((num_row, num_col))
    for i in range(num_row):
        colidx = ratings[ratings['row'] == i]['col']
        result[i, colidx] = ratings[ratings['row'] == i]['Prediction']
    return result

def normalize_rating(ratings, smooth):
    mu = np.mean(ratings['Prediction'])
    var = np.var(ratings['Prediction'])
    rating_sum_per_user = ratings.groupby(by = 'row')['Prediction'].sum()
    counts = ratings.groupby(by = 'row')['row'].count()
    deviation_per_user = np.multiply(ratings.groupby(by = 'row')['Prediction'].var(), counts)
    usermu = (rating_sum_per_user + smooth*mu)/(counts + smooth)
    uservar = (deviation_per_user + smooth*var)/(counts + smooth)
    result = ratings.copy().sort_values(by='row')
    norm_ratings = np.divide(result['Prediction'].to_numpy() - np.repeat(usermu, counts), np.repeat(np.sqrt(uservar), counts))
    result['Prediction'] = norm_ratings.to_numpy()
    return usermu, uservar, result 

    
def gaussian_pLSA(ratings, num_users, num_items, num_hidden_states = 5, max_iter = 10, is_normalize = False):
    #initialization
    np.random.seed(0)
    p_z = np.random.rand(num_users, num_hidden_states) #P(z|u)
    p_z /= p_z.sum(1)[:,  np.newaxis]
    
    if is_normalize:
        usermu, uservar, ratings = normalize_rating(ratings, 5) #TODO
    
    mu_iz  = np.zeros((num_items, num_hidden_states))  #+ np.mean(ratings['Prediction'])
    sigma2_iz = np.ones((num_items, num_hidden_states)) #+ np.var(ratings['Prediction'])
    p_z_given_uri = np.zeros((num_users, num_items, num_hidden_states))
    rating_mat = df_to_mat(ratings, num_users, num_items)
    # E-step
    for i in range(max_iter):
        print("Iteration {}:".format(i))
        print("Start E-step...")
        st = time.time()
        def helper(row):
            user = np.int(row[0])
            item = np.int(row[1])
            rating = row[2]
            p_rating_item = rating_gaussian_model(rating, mu_iz[item], sigma2_iz[item])#all p(rating, item|z)
            #p_rating_item = rating_laplacian_model(rating, mu_iz[item], sigma2_iz[item])
            denom = np.dot(p_rating_item, p_z[user]) #p(rating,  item|z)p(z|u)
            for z in range(num_hidden_states):
                nom = rating_gaussian_model(rating, mu_iz[item, z], sigma2_iz[item, z])*p_z[user, z]  
                #nom = rating_laplacian_model(rating, mu_iz[item, z], sigma2_iz[item, z])*p_z[user, z] 
                p_z_given_uri[user, item, z] = nom/denom #p(z|user, rating, item;thetahat)
        ratings.apply(helper, 1)
        print("E-step finished using time {}".format(time.time()-st))
        
            
        # M-step
        print("Start M-step...")
        st = time.time()
        for u in range(num_users): # update P(z|u)
            denom = np.sum(p_z_given_uri[u]) # sum_z sum_u P(z|u, v, y;hattheta)
            for z in range(num_hidden_states):
                p_z[u, z] = np.sum(p_z_given_uri[u][:, z])/denom # sum of P(z|user, rating, item; thetahat)
        for i in range(num_items):
            for z in range(num_hidden_states):           
                denom = np.sum(p_z_given_uri[:, i, z]) #sum_y  {(z|user, rating, item)}
                item_rating = rating_mat[:, i]
                mu_iz[i, z] = np.dot(p_z_given_uri[:, i, z], item_rating)/denom
                sigma2_iz[i, z] = np.dot(np.square(item_rating - mu_iz[i,z]), p_z_given_uri[:, i, z])/denom
        print("M-step finished using time {}".format(time.time()-st))
        print("="*20)
    if is_normalize:
        return p_z, mu_iz, usermu, uservar
    return p_z, mu_iz

def predict(p_z, mu_iz, usermu=None, uservar=None):
    pred = np.nan_to_num(p_z)@np.nan_to_num(mu_iz.T)
    if usermu is not None:
        pred = np.add(np.multiply(pred, uservar[:,np.newaxis]), usermu[:, np.newaxis])
    return pred

def SVD_impute(ratings, prediction, num_keep=10):
    nr, nc = prediction.shape
    train_matrix = df_to_mat(ratings, nr, nc)
    train_r, train_c = ratings.loc[:, 'row'], ratings.loc[:, 4'col']
    prediction[train_r, train_c] = train_matrix[train_r, train_c]
    u, s, vh = np.linalg.svd(prediction)
    s = s[:num_keep]
    svdresult = u[:, :s.shape[0]]@np.diag(s)@vh[:s.shape[0], :]
    return svdresult

if __name__ == "__main__":
    
    fold_1 = pd.read_csv('../andreas/fold_1.csv', index_col = 0)
    fold_2 = pd.read_csv('../andreas/fold_2.csv', index_col = 0)
    fold_3 = pd.read_csv('../andreas/fold_3.csv', index_col = 0)
    fold_4 = pd.read_csv('../andreas/fold_4.csv', index_col = 0)
    fold_5 = pd.read_csv('../andreas/fold_5.csv', index_col = 0)
    folds = [fold_1, fold_2, fold_3, fold_4, fold_5]

    param_hidden = [10, 15, 20]#[5, 10, 20, 30, 40, 50, 100]
    rmse = []
    max_iter = 5
    is_normalize = True
    log = open("./log.txt", "a")
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    for z in param_hidden:
        print("{}\n Fitting with {} latent variables".format(ts, z), file = log)
        cv_rmse = []
        st = time.time()
        for i in range(len(folds)):
            train = copy.deepcopy(folds)
            test = train.pop(i)
            train = pd.concat(train)
            if is_normalize:
                pz, mu, usermu, uservar = gaussian_pLSA(train, 10000, 1000, z, max_iter, is_normalize)
                #np.savez("./saved_models/plsa{}_iter{}_testfold{}_{}".format(z, max_iter,i, ts), pz, mu, usermu, uservar)
                plsa_prediction = predict(pz, mu, usermu, uservar)
            else:
                pz, mu = gaussian_pLSA(pd.concat(train), 10000, 1000, z, max_iter, is_normalize)
                #np.savez("./saved_models/plsa{}_iter{}_testfold{}_{}".format(z, max_iter, i, ts), pz, mu)
                plsa_prediction = predict(pz, mu)
            svd_prediction = SVD_impute(train, plsa_prediction)
            svd_prediction = np.clip(svd_prediction, 1, 5)
            # Eval on Validation
            ridx = test.loc[:,'row']
            cidx = test.loc[:, 'col']
            err = np.sqrt(np.mean(np.square(test['Prediction']- svd_prediction[ridx, cidx])))
            cv_rmse.append(err)
            print("RMSE on test fold {}: {}".format(i, err))
            print("RMSE on test fold {}: {}".format(i, err), file = log)
        rmse.append(np.mean(cv_rmse))
        print("RMSE mean {} with std {}".format(np.mean(cv_rmse), np.std(cv_rmse)), file=log)   
        print("Used time: {}".format(time.time() - st), file=log)
        print("*"*30, file = log)
    print(rmse, file = log)
