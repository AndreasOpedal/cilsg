'''
This file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from factorization import SVDPP2
from baseline import SVD, ALS
from thresholding import SVDthr
from plsa import PLSA
from ensemble import Ensemble

###############################################################################################

# Constants
TRAIN_DATA_PATH = '../data/data-train.csv'
PREDICTION_INDEXES_PATH = '../data/sample-submission.csv'
PREDICTIONS_DIR = '../predictions/'

###############################################################################################

# Algorithm classes dictionary
algo_classes = {}

# Initialize algo_classes
algo_classes['SVD'] = SVD
algo_classes['ALS'] = ALS
algo_classes['SVDPP2'] = SVDPP2
algo_classes['SVDthr'] = SVDthr
algo_classes['PLSA'] = PLSA
algo_classes['Ensemble'] = Ensemble

###############################################################################################

# Instances dictionary
instances = {}

# Initialize instances
instances['SVD'] = {}
instances['ALS'] = {}
instances['SVDPP2'] = {}
instances['SVDthr'] = {}
instances['PLSA'] = {}
instances['Ensemble'] = {}

# Index single algorithm classes. Further model instances can manually be added.
instances['SVD'][1] = SVD()
instances['ALS'][1] = ALS()
instances['SVDPP2'][1] = SVDPP2()
instances['SVDthr'][1] = SVDthr()
instances['Ensemble'][1] = Ensemble()

###############################################################################################

# Parameter grid dictionary
param_grids = {}

# Parameter grid for SVD
param_grids['SVD'] = {
    'n_factors': [100, 160, 200],
    'impute_strategy': ['mean', 'median']
}

# Parameter grid for ALS
param_grids['ALS'] = {
    'n_factors': [50, 100],
    'n_epochs': [5, 10],
    'init_mean': [0, 0.1],
    'init_std': [0.01, 0.1],
    'reg': [0.5, 1]
}

# Parameter grid for SVDPP2
param_grids['SVDPP2'] = {
    'n_factors': [50, 150],
    'n_epochs': [60, 80],
    'init_mean': [0, 0.2],
    'init_std': [0.01, 0.1],
    'lr_pu': [0.01, 0.02],
    'lr_qi': [0.01, 0.02],
    'alpha_pu': [0.1, 0.3],
    'alpha_qi': [0.1, 0.3],
    'decay_pu': [0.02, 0.03],
    'decay_qi': [0.05, 0.06],
    'reg_pu': [0.06, 0.1],
    'reg_qi': [0.06, 0.1],
    'lambda_bu': [20, 25],
    'lambda_bi': [1, 10],
    'lambda_yj': [40, 45],
    'impute_strategy': ['mean', 'median']
}

# Parameter grid for SVDthr
param_grids['SVDthr'] = {
    'tao': [10000, 15000],
    'step_size': [1.8, 1.99, 2.05],
    'eps': [0.1, 0.2]
}

# Parameter grid for PLSA
param_grids['PLSA'] = {
    'n_latent': [5, 10, 15],
    'n_epochs': [2, 5],
    'to_normalize': [False, True],
    'alpha': [2, 5]
}

###############################################################################################

# Distribution grid dictionary
dist_grids = {}

# Note: for integer types use stats.randint(low, high), while for float types use stats.uniform(start, delta), such that values will
# be in range [start, start+delta]

# Distribution grid for SVD
dist_grids['SVD'] = {
    'n_factors': stats.randint(100, 160)
}

# Distribution grid for ALS
dist_grids['ALS'] = {
    'n_factors': stats.randint(100, 160),
    'n_epochs': stats.randint(5, 10),
    'init_mean': stats.uniform(0, 0.1),
    'init_std': stats.uniform(0.1, 0.1),
}

# Distribution grid for SVDPP2
dist_grids['SVDPP2'] = {
    'n_factors': stats.randint(160, 210),
    'n_epochs': stats.randint(65, 90),
    'init_mean': stats.uniform(0, 0.1),
    'init_std': stats.uniform(0, 0.1),
    'lr_pu': stats.uniform(0.002, 0.01),
    'lr_qi': stats.uniform(0.002, 0.01),
    'alpha_pu': stats.uniform(0.1, 0.3),
    'alpha_qi': stats.uniform(0.1, 0.3),
    'decay_pu': stats.uniform(0.01, 0.03),
    'decay_qi': stats.uniform(00.01, 0.03),
    'reg_pu': stats.uniform(0.05, 0.02),
    'reg_qi': stats.uniform(0.05, 0.02),
    'lambda_bu': stats.uniform(20, 10),
    'lambda_bi': stats.uniform(0.1, 2),
    'lambda_yj': stats.uniform(45, 10)
}

# Distribution grid for SVDthr
dist_grids['SVDthr'] = {
    'tao': stats.uniform(9000, 6000),
    'step_size': stats.uniform(1.5, 1.5),
    'eps': stats.uniform(0.1, 0.3)
}

# Distribution grid for PLSA
dist_grids['PLSA'] = {
    'n_latent': stats.randint(15, 25),
    'n_epochs': stats.randint(2, 5),
}
