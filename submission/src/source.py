'''
This file which contains constants, algorithm classes, model instances, and parameter grids.
'''

from scipy import stats
from factorization import SVDPP2
from baseline import Mean, SVD, ALS
from thresholding import SVDthr
from plsa import pLSA

###############################################################################################

# Constants
TRAIN_DATA_PATH = '../data/data-train.csv'
PREDICTION_INDEXES_PATH = '../data/sample-submission.csv'
NEW_PREDICTIONS_DIR = 'predictions/'

###############################################################################################

# Algorithm classes dictionary
algo_classes = {}

# Initialize algo_classes
algo_classes['Mean'] = Mean
algo_classes['SVD'] = SVD
algo_classes['ALS'] = ALS
algo_classes['SVDPP2'] = SVDPP2
algo_classes['SVDthr'] = SVDthr
algo_classes['pLSA'] = pLSA

###############################################################################################

# Instances dictionary
instances = {}

# Initialize instances
instances['Mean'] = {}
instances['SVD'] = {}
instances['ALS'] = {}
instances['SVDPP2'] = {}
instances['SVDthr'] = {}
instances['pLSA'] = {}

# Index single algorithm classes. Further model instances can manually be added.
instances['Mean'][1] = Mean()
instances['SVD'][1] = SVD()
instances['ALS'][1] = ALS()
instances['SVDPP2'][1] = SVDPP2()
instances['SVDthr'][1] = SVDthr()
instances['pLSA'][1] = pLSA()

###############################################################################################

# Example parameter grid for Grid Search, for the pLSA algorithm
param_grid = {
    'n_latent': [5, 10, 15],
    'n_epochs': [2, 5],
    'to_normalize': [False, True],
    'alpha': [2, 5]
}

###############################################################################################

# Example parameter distributions for Random Search, for the pLSA algorithm
# Note: for float types, use stats.uniform(start, delta), such that random values will be in range [start, start+delta]
dist_grid = {
    'n_latent': stats.randint(5, 10),
    'n_epochs': stats.randint(2, 5)
}
