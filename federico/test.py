import numpy as np
import utils
import preprocess
from postprocess import Intifier
import model_selection
import factorization
import metrics

np.random.seed(42)

# Whether to do grid search
TO_GRID = False

# Whether to do k-fold crossvalidation or write predictions
TO_KF_CV = True

# Whether to do crossvalidation or write predictions
TO_CV = False

# Whether to impute or not
TO_IMPUTE = False

# Whether to build folds from scratch (k-fold crossvalidation)
TO_BUILD_FOLDS = False

# Whether to postprocess or not
TO_POSTPROCESS = True

# Whether to apply post-processing after computing

# Prediction file name
PRED_FILE_NAME = './predictions/svd-pp-12.csv'

# Imputer to use
imputer = preprocess.knn_impute

# The postprocesser to use
postprocesser = Intifier(conf=0.4)

# Model to use
model = factorization.SVDPP(k=100, l=0, eta=0.01, batch_size=20, epochs=100)

# Parameters to use for grid search
# svdpp_setup = []
# svdpp_setup.append([25, 0.8, 0.01, 100, 1500])
svdb_setup = []
svdb_setup.append([20, 0.5, 0.01, 300, 1500, 1, 5])
svdb_setup.append([20, 0.5, 0.02, 300, 1500, 1, 5])
svdb_setup.append([20, 0.1, 0.01, 300, 1500, 1, 5])
svdb_setup.append([20, 0.5, 0.01, 500, 1500, 1, 5])

if TO_GRID:
    model = factorization.SVDBox()
    results = model_selection.grid_search(svdb_setup, model=model, metric=metrics.rmse, postprocesser=postprocesser, dir_path='./cv-5-fold/')
    # Print results
    for result in results:
        params, mean, var = result
        print('Mean: % 5.9f, Variance: % 5.9f, Parameters: %s' % (mean, var, params))
elif TO_KF_CV:
    if TO_BUILD_FOLDS:
        print('Creating folds...')
        X_df = utils.read_data_as_data_frame('../data/data_train.csv')
        folds = model_selection.kfold_split(X_df, save=True)
        print('Done creating folds')
    mean, var = model_selection.kfold_cv(model=model, metric=metrics.rmse, postprocesser=postprocesser, dir_path='./cv-5-fold/')
    print('Mean: % 5.9f, Variance: % 5.9f' % (mean, var))
elif TO_CV:
    print('Reading data...')
    X = utils.read_data_as_matrix('../data/data_train.csv')
    print('Done reading data')
    X_train, X_test = model_selection.train_test_split(X, min_user_ratings=2, test_movies_pct=0.5)
    model.fit(X_train)
    X_pred = model.transform()
    if TO_POSTPROCESS:
        X_pred = postprocesser.process(X_pred)
    error = metrics.rmse(X_test, X_pred)
    print('Error: % 5.9f' % error)
else:
    print('Reading data...')
    X = utils.read_data_as_matrix('../data/data_train.csv')
    tups = utils.read_submission_indexes('../data/sampleSubmission.csv')
    print('Done reading data')
    if TO_IMPUTE:
        print('Imputing...')
        X = imputer(X, n_neighbors=5)
        print('Done imputing')
    model.fit(X)
    X_pred = model.transform()
    if TO_POSTPROCESS:
        X_pred = postprocesser.process(X_pred)
    print('Writing predictions...')
    utils.write_predictions_to_csv(X_pred, tups, PRED_FILE_NAME)
    print('Done writing predictions')
