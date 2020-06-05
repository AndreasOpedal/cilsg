'''
A file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from prediction import SGDbound, SGDheu, SGDweighted

###############################################################################################

# Constants
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
instances['SGDheu'][33] = SGDheu(n_factors=183, n_epochs=50, init_mean=0, init_std=0.01, lr_pu=0.016, lr_qi=0.016, alpha_pu=0.7, alpha_qi=0.7, decay_pu=0.4, decay_qi=0.4, reg_pu=0.075, reg_qi=0.075, lambda_bu=25, lambda_bi=1, conf=0.49)
instances['SGDheu'][34] = SGDheu(n_factors=164, n_epochs=74, init_mean=0.20290823068583227, init_std=0.006739247315860849, lr_pu=0.013978749173528232, lr_qi=0.014193006564689255, decay_pu=0.048568935943306406, decay_qi=0.0721391003797361, reg_pu=0.08465510910679358, reg_qi=0.06437173876894195, lambda_bu=19, lambda_bi=4.905163341379891, conf=0.4869291460385132)
instances['SGDweighted'][1] = SGDweighted(n_factors=100, n_epochs=20, init_mean=0, init_std=0.1, lr_pu=0.01, lr_qi=0.01, decay_pu=0.05, decay_qi=0.05, reg_pu=0.075, reg_qi=0.075, lambda_bu=25, lambda_bi=0.5, conf=0.49)
instances['SGDbound'][1] = SGDbound(n_factors=100, n_epochs=50, lr_pu=0.015, lr_qi=0.015, reg_pu=1, reg_qi=1, conf=0.49)

###############################################################################################

# Grid for Grid Search
param_grid = {
    'n_factors': [190],
    'n_epochs': [65],
    'init_mean': [0.2],
    'init_std': [0.005],
    'lr_pu': [0.015],
    'lr_qi': [0.015],
    'decay_pu': [0.055],
    'decay_qi': [0.055],
    'reg_pu': [0.075],
    'reg_qi': [0.075],
    'lambda_bu': [25, 26, 27, 28],
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
