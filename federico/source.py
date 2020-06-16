'''
This file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from factorization import SGDheu, SGDPP2
from baseline import ALS

###############################################################################################

# Constants
SEED = 666
TRAIN_DATA_PATH = '../data/data_train.csv'
PREDICTION_INDEXES_PATH = '../data/sampleSubmission.csv'
NEW_PREDICTIONS_DIR = 'predictions/'
WEIGHTS_DIR = 'weights/'

###############################################################################################

# Algorithm classes dictionary
algo_classes = {}

# Initialize algo_classes
algo_classes['SGDheu'] = SGDheu
algo_classes['SGDPP2'] = SGDPP2
algo_classes['ALS'] = ALS

###############################################################################################

# Instances dictionary
instances = {}

# Initialize instances
instances['SGDheu'] = {}
instances['SGDPP2'] = {}
instances['ALS'] = {}

# Index single algorithm classes. Further model instances can manually be added.
instances['SGDheu'][44] = SGDheu(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDPP2'][14] = SGDPP2(n_factors=192, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=50, conf=None)
instances['SGDPP2'][18] = SGDPP2(n_factors=160, n_epochs=25, init_mean=0.2, init_std=0.005, lr_pu=0.001, lr_qi=0.005, alpha_pu=0.25, alpha_qi=0.35, decay_pu=0.005, decay_qi=0.001, reg_pu=0.08, reg_qi=0.01, lambda_bu=10, lambda_bi=10, lambda_yj=10, conf=None)
instances['SGDPP2'][19] = SGDPP2(n_factors=192, n_epochs=58, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.1, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, lambda_yj=50, conf=None)
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
    'lambda_bi': [0.5]
}

###############################################################################################

# Grid for Random Search
dist_grid = {
    'n_factors': stats.randint(190, 194),
    'n_epochs': stats.randint(80, 95),
    'init_mean': stats.uniform(0.175, 0.05),
    'init_std': stats.uniform(0.004, 0.002),
    'lr_pu': stats.uniform(0.0012, 0.0007),
    'lr_qi': stats.uniform(0.0012, 0.0007),
    'alpha_pu': stats.uniform(0.2, 0.25),
    'alpha_qi': stats.uniform(0.2, 0.25),
    'decay_pu': stats.uniform(0.01, 0.02),
    'decay_qi': stats.uniform(0.04, 0.03),
    'reg_pu': stats.uniform(0.05, 0.02),
    'reg_qi': stats.uniform(0.055, 0.02),
    'lambda_bu': stats.randint(20, 25),
    'lambda_bi': stats.uniform(0.4, 0.2),
    'lambda_yj': stats.randint(50, 55)
}
