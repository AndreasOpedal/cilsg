import numpy as np
import pandas as pd
from scipy import stats
import utils
from model_selection import grid_search
from surprise import Reader, Dataset, accuracy, SVDpp, SlopeOne
from surprise.model_selection import train_test_split
from surprise.model_selection.search import RandomizedSearchCV
from prediction import SGD, SGDPP, SGDtum, SVDthr, SGDbound, SGDPPtum, SGDheu
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
    # model = NeuMF(lr=0.05, n_epochs=25, gmf_u_reg=0.01, gmf_i_reg=0.01, n_features=160, verbose=True)
    # model = SGDPPtum(n_factors=164, n_epochs=40, lr0=0.075, lr0_b=0.01, lr0_yj=0.007, reg=0.08, reg_bu=0.008, reg_bi=0.0015, reg_yj=0.0015, decay=0.005, decay_b=0.001, decay_yj=0.0015, alpha=0.71, alpha_b=0.85, alpha_yj=0.01, conf=0.45, init_bias=True, verbose=True)
    # model = SGDheu(n_factors=100, n_epochs=65, init_mean=0, init_std=0.01, lr_pu=0.015, lr_qi=0.015, decay_pu=0.05, decay_qi=0.05, reg_pu=0.075, reg_qi=0.075, lambda_bu=10, lambda_bi=15, conf=0.35, verbose=True)
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
        'n_factors': [110, 130],
        'n_epochs': [65, 70],
        'lr_pu': [0.02, 0.025],
        'lr_qi': [0.02, 0.025],
        'decay_pu': [0.05],
        'decay_qi': [0.05],
        'reg_pu': [0.085, 0.1],
        'reg_qi': [0.085, 0.1],
        'lambda_bu': [10, 15],
        'lambda_bi': [10, 15],
        'conf': [0.45]
    }
    results = grid_search(SGDheu, param_grid, trainset, testset, target_score=None, verbose=True)

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
    param_grid_sgdheu = {
        'n_factors': stats.randint(120, 185),
        'n_epochs': stats.randint(20, 60),
        'init_std': stats.uniform(0.009, 0.09),
        'lr_pu': stats.uniform(0.005, 0.2),
        'lr_qi': stats.uniform(0.005, 0.2),
        'decay_pu': stats.uniform(0.001, 0.1),
        'decay_qi': stats.uniform(0.001, 0.1),
        'reg_pu': stats.uniform(0.025, 0.8),
        'reg_qi': stats.uniform(0.025, 0.8),
        'lambda_bu': stats.randint(1, 30),
        'lambda_bi': stats.randint(1, 30),
        'conf': stats.uniform(0.05, 0.4)
    }
    gs = RandomizedSearchCV(algo_class=SGDheu, param_distributions=param_grid_sgdheu, measures=['rmse'], cv=5, joblib_verbose=100, n_iter=100, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    results = pd.DataFrame.from_dict(gs.cv_results)
    print(results)

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
    # model = SGDheu(n_factors=100, n_epochs=65, init_std=0.009, lr_pu=0.015, lr_qi=0.015, decay_pu=0.05, decay_qi=0.05, reg_pu=0.075, reg_qi=0.075, lambda_bu=10, lambda_bi=15, conf=0.49, verbose=True)
    # model = SGDPP(n_factors=151, n_epochs=45, lr0=0.063692091948985, lr0_b=0.09860293512519518, lr0_yj=0.05399393492907, reg=0.10316897292827662, reg_bu=0.28627871231683566, reg_bi=0.28313286417289973, reg_yj=0.08970518093274805, decay=0.6713569724412978, decay_b=0.14163140243293743, decay_yj=0.31240637031314267, init_bias=True, conf=0.4086156458032375, verbose=True)
    # model = SVDthr(n_epochs=100, lr=3, tao=19500, conf=None, verbose=True)
    # model = SGDtum(n_factors=169, n_epochs=42, lmb_bu=2.420964387368687, lmb_bi=0.20017849440978214, lr0_pu=0.06978378207487108, lr0_qi=0.013726907314909859, lr0_bu=0.006885696896143731, lr0_bi=0.03199470070000068, reg_pu=0.1590838986249438, reg_qi=0.06880461862509689, reg_bu=0.2049753831654436, reg_bi=0.44256107286152097, decay_pu=0.29941379098382076, decay_qi=0.22599235520721855, decay_bu=0.09498904597959346, decay_bi=0.36910356417846474, alpha_pu=0.44019487747234276, alpha_qi=0.43966762647922475, alpha_bu=0.2903441180407593, alpha_bi=0.14297249003135515, init_bias=True, conf=0.38385301945950434, verbose=True)
    # Fit model
    model.fit(data_train)
    # Predictions
    predictions = []
    for index in indexes:
        u, i = index
        r = model.predict(u, i).est # prediction
        predictions.append((u, i, r))
    # Dump predictions
    utils.write_predictions_to_csv(predictions, 'predictions/sgdavg-1.csv')

if __name__ == '__main__':
    # execute main function
    # cv()
    # grid()
    # random_search()
    # dump()
