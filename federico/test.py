import numpy as np
import utils
import preprocess
from postprocess import intify_prediction
import model_selection
import factorization
import metrics

np.random.seed(42)

# Whether to do crossvalidation or write predictions
TO_DO_CV = True

# Whether to impute or not
TO_IMPUTE = False

# Whether to build folds from scratch (k-fold crossvalidation)
TO_BUILD_FOLDS = False

# Prediction file name
PRED_FILE_NAME = './predictions/pred-4.csv'

# Imputer to use
imputer = preprocess.knn_impute

# Model to use
model = factorization.SVDPP(epochs=1500, eta=0.01, batch_size=20, l=2)

if TO_DO_CV:
    if TO_BUILD_FOLDS:
        print('Creating folds...')
        X_df = utils.read_data_as_data_frame('../data/data_train.csv')
        folds = model_selection.kfold_split(X_df, save=True)
        print('Done creating folds')
    mean, var = model_selection.kfold_cv(model=model, metric=metrics.rmse, postprocess=intify_prediction, dir_path='./cv-5-fold/')
    print('Mean: % 5.9f, Variance: % 5.9f' % (mean, var))
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
    print('Writing predictions...')
    utils.write_predictions_to_csv(X_pred, tups, PRED_FILE_NAME)
    print('Done writing predictions')
