'''
A file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from factorization import SGDbound, SGD, SGDweighted, SGDPP, SGDPP2
from baseline import ALS

###############################################################################################

# Constants
SEED = 666
TRAIN_DATA_PATH = '../data/data_train.csv'
PREDICTION_INDEXES_PATH = '../data/sampleSubmission.csv'
NEW_PREDICTIONS_DIR = 'predictions/'

###############################################################################################

# Algorithm classes dictionary
algo_classes = {}

# Initialize algo_classes
algo_classes['SGD'] = SGD
algo_classes['SGDPP'] = SGDPP
algo_classes['SGDPP2'] = SGDPP2
algo_classes['ALS'] = ALS

###############################################################################################

# Instances dictionary
instances = {}

# Initialize instances
instances['SGD'] = {}
instances['SGDPP'] = {}
instances['SGDPP2'] = {}
instances['ALS'] = {}

# Index single algorithm classes. Further model instances can manually be added.
instances['SGD'][49] = SGD(n_factors=164, n_epochs=100, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][48] = SGD(n_factors=164, n_epochs=95, init_mean=0.2, init_std=0.05, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0, decay_qi=0.8, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][47] = SGD(n_factors=160, n_epochs=95, init_mean=0, init_std=0.01, lr_pu=0.001, lr_qi=0.001, alpha_pu=0.2, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][46] = SGD(n_factors=162, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.02, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][45] = SGD(n_factors=162, n_epochs=75, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.02, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][44] = SGD(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][43] = SGD(n_factors=192, n_epochs=90, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][42] = SGD(n_factors=192, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][41] = SGD(n_factors=192, n_epochs=75, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][40] = SGD(n_factors=162, n_epochs=20, init_mean=0.2, init_std=0.005, lr_pu=0.02, lr_qi=0.02, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.06, reg_qi=0.06, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGD'][39] = SGD(n_factors=162, n_epochs=16, init_mean=0.2, init_std=0.005, lr_pu=0.021, lr_qi=0.019, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.055, reg_qi=0.055, lambda_bu=25, lambda_bi=2.5, conf=0.49)
instances['SGD'][38] = SGD(n_factors=162, n_epochs=20, init_mean=0.2, init_std=0.005, lr_pu=0.02, lr_qi=0.02, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.06, reg_qi=0.06, lambda_bu=25, lambda_bi=0.5, conf=0.49)
instances['SGD'][37] = SGD(n_factors=100, n_epochs=42, init_mean=0.2, init_std=0.005, lr_pu=0.015, lr_qi=0.015, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.07, reg_qi=0.07, lambda_bu=25, lambda_bi=0.5, conf=0.49)
instances['SGDPP'][1] = SGDPP(n_factors=160, n_epochs=25, init_mean=0, init_std=0.01, lr_pu=0.005, lr_qi=0.005, lr_yj=0.01, alpha_pu=0.45, alpha_qi=0.4, alpha_yj=0.3, decay_pu=0.005, decay_qi=0.005, decay_yj=0.01, reg_pu=0.07, reg_qi=0.07, reg_yj=0.01, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDPP'][2] = SGDPP(n_factors=164, n_epochs=25, init_mean=0, init_std=0.01, lr_pu=0.002, lr_qi=0.002, lr_yj=0.001, alpha_pu=0.45, alpha_qi=0.4, alpha_yj=0.1, decay_pu=0.005, decay_qi=0.005, decay_yj=0.001, reg_pu=0.07, reg_qi=0.07, reg_yj=0.001, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDPP2'][1] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=250, conf=None)
instances['SGDPP2'][2] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=200, conf=None)
instances['SGDPP2'][3] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=150, conf=None)
instances['SGDPP2'][4] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['SGDPP2'][5] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=10, conf=None)
instances['SGDPP2'][6] = SGDPP2(n_factors=192, n_epochs=65, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['SGDPP2'][7] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, impute_strategy='mean', conf=None)
instances['SGDPP2'][8] = SGDPP2(n_factors=164, n_epochs=40, init_mean=0, init_std=0.01, lr_pu=0.0075, lr_qi=0.0075, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.005, decay_qi=0.005, reg_pu=0.08, reg_qi=0.08, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['SGDPP2'][9] = SGDPP2(n_factors=192, n_epochs=105, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['SGDPP2'][10] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=0.48)
instances['SGDPP2'][11] = SGDPP2(n_factors=164, n_epochs=40, init_mean=0, init_std=0.01, lr_pu=0.0075, lr_qi=0.0075, alpha_pu=0, alpha_qi=0, decay_pu=0.005, decay_qi=0.005, reg_pu=0.08, reg_qi=0.08, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['SGDPP2'][12] = SGDPP2(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, impute_strategy='median', conf=None)
instances['SGDPP2'][13] = SGDPP2(n_factors=192, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=80, conf=None)
instances['ALS'][1] = ALS()

###############################################################################################

# Grid for Grid Search
param_grid = {
    'n_factors': [192],
    'n_epochs': [85, 90, 95],
    'init_mean': [0.2],
    'init_std': [0.005],
    'lr_pu': [0.0025, 0.005],
    'lr_qi': [0.0025, 0.005],
    'alpha_pu': [0.3, 0.325],
    'alpha_qi': [0.3, 0.325],
    'decay_pu': [0.01],
    'decay_qi': [0.01],
    'reg_pu': [0.06],
    'reg_qi': [0.065],
    'lambda_bu': [25],
    'lambda_bi': [0.5],
    'conf': [None]
}

###############################################################################################

# Grid for Random Search
dist_grid = {
    'n_factors': stats.randint(160, 165),
    'n_epochs': stats.randint(65, 90),
    'init_mean': stats.uniform(0.2, 0.05),
    'init_std': stats.uniform(0.005, 0.01),
    'lr_pu': stats.uniform(0.012, 0.02),
    'lr_qi': stats.uniform(0.012, 0.02),
    'decay_pu': stats.uniform(0.04, 0.05),
    'decay_qi': stats.uniform(0.04, 0.05),
    'reg_pu': stats.uniform(0.055, 0.03),
    'reg_qi': stats.uniform(0.055, 0.03),
    'lambda_bu': stats.randint(12, 25),
    'lambda_bi': stats.uniform(0.1, 6),
    'conf': stats.uniform(0.47, 0.02)
}
