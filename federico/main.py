import numpy as np
import pandas as pd
from scipy import stats
import utils
from model_selection import grid_search
from surprise import Reader, Dataset, accuracy, SVDpp, SlopeOne
from surprise.model_selection import train_test_split
from surprise.model_selection.search import RandomizedSearchCV
from prediction import SGDbox, SGD, SGDPP, SGDtum, SVDthr, SGDbound
from dnn import NeuMF

def cv():
    np.random.seed(666)
    # Read dataframe
    df = utils.read_data_as_data_frame('../data/data_train.csv')
    # Define reader
    reader = Reader()
    # Set up data
    data = Dataset.load_from_df(df[['row', 'col', 'Prediction']], reader)
    # Create data split
    trainset, testset = train_test_split(data, test_size=0.25)
    # Define model
    # model = SVDpp(n_factors=160, n_epochs=25, lr_pu=0.015, lr_qi=0.015, reg_pu=0.1, reg_qi=0.07, lr_bu=0.04, lr_bi=0.04, lr_yj=0.05, reg_yj=0.05, verbose=True)
    # model = SVDbx(n_factors=2, n_epochs=25, lr_pu=0.001, lr_qi=0.001, reg_pu=0.006, reg_qi=0.004, conf=0.4, verbose=True)
    # model = SGD(n_factors=2, n_epochs=50, lr0=0.01, lr0_b=0.01, reg=0.08, reg_bu=0.02, reg_bi=0.02, decay=0.005, decay_b=0.025, init_bias=True, conf=0.4, verbose=True)
    # model = SGDPP(n_factors=183, n_epochs=34, init_std=0.0110162557627814, lr0=0.0162717539757305, lr0_b=0.012, lr0_yj=0.07, reg=0.0757259675190306, reg_bu=0.191038542203524, reg_bi=0.0550618848336194, reg_yj=0.1, decay=0.365929028624037, decay_b=0.0961457957446339, decay_yj=0.23, init_bias=True, conf=0.45412114687501, verbose=True)
    # model = SGDtum(n_factors=159, n_epochs=26, init_std=0.02, lr0=0.015, lr0_b=0.0475, reg=0.08, reg_bu=0.1, reg_bi=0.1, decay=0.005, decay_b=0.001, alpha=0.008, alpha_b=0.002, init_bias=True, conf=0.35, verbose=True)
    # model = SVDthr(n_epochs=100, lr=3, tao=19500, conf=None, verbose=True)
    # model = SVDbbx(init_bias=True, conf=0.45, verbose=True)
    # model = SGDbound(n_factors=165, n_epochs=100, lr_pu=0.1, lr_qi=0.1, lr_bu=0.01, lr_bi=0.01, reg=1, conf=None, verbose=True)
    model = NeuMF(lr=0.05, n_epochs=25, gmf_u_reg=0.01, gmf_i_reg=0.01, n_features=160, verbose=True)
    # Fit model
    model.fit(trainset)
    # Predictions
    predictions = model.test(testset)
    # Error
    error = accuracy.rmse(predictions)

def grid():
    np.random.seed(666)
    # Read dataframe
    df = utils.read_data_as_data_frame('../data/data_train.csv')
    # Define reader
    reader = Reader()
    # Set up data
    data = Dataset.load_from_df(df[['row', 'col', 'Prediction']], reader)
    # Create data split
    trainset, testset = train_test_split(data, test_size=0.25)
    # Parameter grid
    param_grid = {
        'n_factors': [24, 36],
        'n_epochs': [25, 30, 35, 40, 45, 50, 55],
        'lr_pu': [0.002, 0.005, 0.01],
        'lr_qi': [0.002, 0.005, 0.01],
        'lr_bu': [0.004, 0.008],
        'lr_bi': [0.004, 0.008],
        'max_init_p': [1],
        'max_init_q': [1],
        'conf': [0.45]
    }
    results = grid_search(SGDbound, param_grid, trainset, testset, target_score=1.010, verbose=True)

def random_search():
    np.random.seed(666)
    # Read dataframe
    df = utils.read_data_as_data_frame('../data/data_train.csv')
    # Define reader
    reader = Reader()
    # Set up data
    data = Dataset.load_from_df(df[['row', 'col', 'Prediction']], reader)
    # Define grid
    param_grid_sgdtum = {
        'n_factors': stats.randint(140, 170),
        'n_epochs': stats.randint(20, 50),
        'lmb_bu': stats.uniform(0.001, 2.5),
        'lmb_bi': stats.uniform(0.001, 2.5),
        'lr0_pu': stats.uniform(0.005, 0.1),
        'lr0_qi': stats.uniform(0.005, 0.1),
        'lr0_bu': stats.uniform(0.004, 0.1),
        'lr0_bi': stats.uniform(0.004, 0.1),
        'reg_pu': stats.uniform(0.02, 0.2),
        'reg_qi': stats.uniform(0.02, 0.2),
        'reg_bu': stats.uniform(0.002, 0.45),
        'reg_bi': stats.uniform(0.002, 0.45),
        'decay_pu': stats.uniform(0.002, 0.8),
        'decay_qi': stats.uniform(0.002, 0.8),
        'decay_bu': stats.uniform(0.001, 0.5),
        'decay_bi': stats.uniform(0.001, 0.5),
        'alpha_pu': stats.uniform(0.01, 2),
        'alpha_qi': stats.uniform(0.01, 2),
        'alpha_bu': stats.uniform(0.01, 2),
        'alpha_bi': stats.uniform(0.01, 2),
        'conf': stats.uniform(0.35, 0.12)
    }
    param_grid_sgdpp = {
        'n_factors': stats.randint(140, 170),
        'n_epochs': stats.randint(20, 50),
        'lr0': stats.uniform(0.006, 0.09),
        'lr0_b': stats.uniform(0.004, 0.1),
        'lr0_yj': stats.uniform(0.006, 0.09),
        'reg': stats.uniform(0.02, 0.2),
        'reg_bu': stats.uniform(0.002, 0.45),
        'reg_bi': stats.uniform(0.002, 0.45),
        'reg_yj': stats.uniform(0.02, 0.2),
        'decay': stats.uniform(0.002, 0.8),
        'decay_b': stats.uniform(0.0005, 0.5),
        'decay_yj': stats.uniform(0.0005, 0.5),
        'conf': stats.uniform(0.35, 0.12)
    }
    gs = RandomizedSearchCV(algo_class=SGDPP, param_distributions=param_grid_sgdpp, measures=['rmse'], cv=3, joblib_verbose=100, n_iter=110, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

def dump():
    # Set seed
    np.random.seed(666)
    # Define reader
    reader = Reader()
    # Load training data as dataframe and set up training data
    df_train = utils.read_data_as_data_frame('../data/data_train.csv')
    data_train = Dataset.load_from_df(df_train[['row', 'col', 'Prediction']], reader).build_full_trainset()
    # Read indexes for submission
    indexes = utils.read_submission_indexes('../data/sampleSubmission.csv')
    # Define model
    # model = SGDPP(n_factors=164, n_epochs=30, lr0=0.0075, lr0_b=0.01, lr0_yj=0.007, reg=0.08, reg_bu=0.008, reg_bi=0.0015, reg_yj=0.0015, decay=0.005, decay_b=0.001, decay_yj=0.0015, init_bias=True, conf=0.45, verbose=True)
    # model = SVDthr(n_epochs=100, lr=3, tao=19500, conf=None, verbose=True)
    model = SGDtum(n_factors=169, n_epochs=42, lmb_bu=2.420964387368687, lmb_bi=0.20017849440978214, lr0_pu=0.06978378207487108, lr0_qi=0.013726907314909859, lr0_bu=0.006885696896143731, lr0_bi=0.03199470070000068, reg_pu=0.1590838986249438, reg_qi=0.06880461862509689, reg_bu=0.2049753831654436, reg_bi=0.44256107286152097, decay_pu=0.29941379098382076, decay_qi=0.22599235520721855, decay_bu=0.09498904597959346, decay_bi=0.36910356417846474, alpha_pu=0.44019487747234276, alpha_qi=0.43966762647922475, alpha_bu=0.2903441180407593, alpha_bi=0.14297249003135515, init_bias=True, conf=0.38385301945950434, verbose=True)
    # Fit model
    model.fit(data_train)
    # Predictions
    predictions = []
    for index in indexes:
        u, i = index
        r = model.predict(u, i).est # prediction
        predictions.append((u, i, r))
    # Dump predictions
    utils.write_predictions_to_csv(predictions, 'predictions/sgdtum-33.csv')

if __name__ == '__main__':
    # execute main function
    # cv()
    # grid()
    random_search()
    # dump()
