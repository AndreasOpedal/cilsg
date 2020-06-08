'''
A file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from prediction import SGDbound, SGDheu, SGDweighted

###############################################################################################

# Constants
SEED = 666
TRAIN_DATA_PATH = '../data/data_train.csv'
PREDICTION_INDEXES_PATH = '../data/sampleSubmission.csv'
NEW_PREDICTIONS_DIR = 'predictions/'

###############################################################################################

# Algo classes dictionary
algo_classes = {}

# Initialize algo_classes
algo_classes['SGDheu'] = SGDheu
algo_classes['SGDweighted'] = SGDweighted
algo_classes['SGDbound'] = SGDbound

###############################################################################################

# Instances dictionary
instances = {}

# Initialize instances
instances['SGDheu'] = {}
instances['SGDweighted'] = {}
instances['SGDbound'] = {}

# Index single algorithm classes. Further model instances can manually be added.
instances['SGDheu'][48] = SGDheu(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0, decay_qi=0.5, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][47] = SGDheu(n_factors=160, n_epochs=95, init_mean=0, init_std=0.01, lr_pu=0.001, lr_qi=0.001, alpha_pu=0.2, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][46] = SGDheu(n_factors=162, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.02, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][45] = SGDheu(n_factors=162, n_epochs=75, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.02, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][44] = SGDheu(n_factors=192, n_epochs=95, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][43] = SGDheu(n_factors=192, n_epochs=90, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][42] = SGDheu(n_factors=192, n_epochs=85, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][41] = SGDheu(n_factors=192, n_epochs=75, init_mean=0.2, init_std=0.005, lr_pu=0.005, lr_qi=0.005, alpha_pu=0.3, alpha_qi=0.3, decay_pu=0.02, decay_qi=0.05, reg_pu=0.06, reg_qi=0.065, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][40] = SGDheu(n_factors=162, n_epochs=20, init_mean=0.2, init_std=0.005, lr_pu=0.02, lr_qi=0.02, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.06, reg_qi=0.06, lambda_bu=25, lambda_bi=0.5, conf=None)
instances['SGDheu'][39] = SGDheu(n_factors=162, n_epochs=16, init_mean=0.2, init_std=0.005, lr_pu=0.021, lr_qi=0.019, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.055, reg_qi=0.055, lambda_bu=25, lambda_bi=2.5, conf=0.49)
instances['SGDheu'][38] = SGDheu(n_factors=162, n_epochs=20, init_mean=0.2, init_std=0.005, lr_pu=0.02, lr_qi=0.02, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.06, reg_qi=0.06, lambda_bu=25, lambda_bi=0.5, conf=0.49)
instances['SGDheu'][37] = SGDheu(n_factors=100, n_epochs=42, init_mean=0.2, init_std=0.005, lr_pu=0.015, lr_qi=0.015, alpha_pu=0.25, alpha_qi=0.25, decay_pu=0.05, decay_qi=0.05, reg_pu=0.07, reg_qi=0.07, lambda_bu=25, lambda_bi=0.5, conf=0.49)
instances['SGDweighted'][2] = SGDweighted(n_factors=160, n_epochs=4, init_mean=0.2, init_std=0.005, alpha_pu=1, alpha_qi=1, reg_pu=1, reg_qi=1, lambda_bu=10, lambda_bi=5, conf=None)
instances['SGDbound'][1] = SGDbound(n_factors=100, n_epochs=50, lr_pu=0.015, lr_qi=0.015, reg_pu=10000, reg_qi=10000, conf=None)

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
