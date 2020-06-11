import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy

fold_1 = pd.read_csv('../andreas/fold_1.csv', index_col = 0)
fold_2 = pd.read_csv('../andreas/fold_2.csv', index_col = 0)
fold_3 = pd.read_csv('../andreas/fold_3.csv', index_col = 0)
fold_4 = pd.read_csv('../andreas/fold_4.csv', index_col = 0)
fold_5 = pd.read_csv('../andreas/fold_5.csv', index_col = 0)
folds = [fold_1, fold_2, fold_3, fold_4, fold_5]


def rating_gaussian_model(rating, mean, var):
    return np.exp(-(rating-mean)**2/(2*var))/(np.sqrt(2*np.pi*var))

def df_to_mat(ratings, num_row, num_col):
    result = np.zeros((num_row, num_col))
    for i in range(num_row):
        colidx = ratings[ratings['row'] == i]['col']
        result[i, colidx] = ratings[ratings['row'] == i]['Prediction']
    return result

def gaussian_pLSA(ratings, num_users, num_items, num_hidden_states = 5, max_iter = 100):
    #initialization
    np.random.seed(0)
    p_z = np.random.rand(num_users, num_hidden_states) #P(z|u)
    p_z /= p_z.sum(1)[:,  np.newaxis]
    mu_iz  = np.zeros((num_items, num_hidden_states))  + np.mean(ratings['Prediction'])
    sigma2_iz = np.ones((num_items, num_hidden_states)) + np.var(ratings['Prediction'])
    p_z_given_uri = np.zeros((num_users, num_items, num_hidden_states))
    rating_mat = df_to_mat(ratings, num_users, num_items)
    # E-step
    for i in range(max_iter):
        print("Iteration {}:".format(i))
        print("Start E-step...")
        st = time.time()
        def helper(row):
            user = row[0]
            item = row[1]
            rating = row[2]
            p_rating_item = rating_gaussian_model(rating, mu_iz[item], sigma2_iz[item])#all p(rating, item|z)
            denom = np.dot(p_rating_item, p_z[user]) #p(rating,  item|z)p(z|u)
            for z in range(num_hidden_states):
                nom = rating_gaussian_model(rating, mu_iz[item, z], sigma2_iz[item, z])*p_z[user, z]  
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
    return p_z, mu_iz

param_hidden = [10, 20, 30, 40, 50, 100]
rmse = []
max_iter = 5
log = open("./log.txt". "a")
for z in param_hidden:
    print("Fitting with {} latent variables".format{z}, file = log)
    acc_err = 0
    st = time.time()
    for i in range(len(folds)):
        train = copy.deepcopy(folds)
        test = train.pop(i)
        pz, mu = gaussian_pLSA(pd.concat(train), 10000, 1000, z, max_iter)
        np.save("./saved_models/plsa{}_iter{}_testfold{}.npz".format{z, max_iter,i}, pz, mu)
        prediction = np.nan_to_num(pz)@np.nan_to_num(mu.T)
        ridx = test.loc[:,'row']
        cidx = test.loc[:, 'col']
        err = np.sqrt(np.mean(np.square(test['Prediction']- prediction[ridx, cidx])))
        acc_err += err
        print("RMSE on test fold {}: {}".format(i, err), file = log)
    rmse.append(acc_err/5)
    print("Used time: {}".format(time.time() - st), file=log)
    print("*"*30)
    print("*"*30)
print(rmse, file = log)