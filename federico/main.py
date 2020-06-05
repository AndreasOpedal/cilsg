import argparse
import numpy as np
import pandas as pd
import utils
import source
from model_selection import targeting_rmse
from preprocess import synthetic_ratings, build_weights
from surprise import Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection.search import RandomizedSearchCV, GridSearchCV
from prediction import SGDPP, SGDtum, SGDheu, SGDweighted

def cv(model, data):
    '''
    Performs cross-validation, by training on 75% of the provided data.
    The test metric is RMSE.

    Parameters:
    model (surprise.AlgoBase): the model to test
    data (surprise.Dataset): the data to use
    '''
    # Create data split
    trainset, testset = train_test_split(data, test_size=0.25)
    # Fit model
    model.fit(trainset)
    # Predictions
    predictions = model.test(testset)
    # Error
    error = accuracy.rmse(predictions)

def target_cv(model, data):
    '''
    Performs cross-validation on the ratings, i.e. computes the RMSE for each rating {1,2,3,4,5}.
    The model in trained on 75% of the provided data.

    Parameters:
    model (surprise.AlgoBase): the model to test
    data (surprise.Dataset): the data to use
    '''
    # Create data split
    trainset, testset = train_test_split(data, test_size=0.25)
    # Fit model
    model.fit(trainset)
    # Target RMSEs
    targeting_rmse(model, testset)

def kfold(model, data, k=10):
    '''
    Perform k-fold cross-validation. The test metric is RMSE.

    Parameters:
    model (surprise.AlgoBase): the model to test
    data (surprise.Dataset): the data to use
    k (int): the number of folds. By default 10
    '''
    # Set up kfold
    dict = cross_validate(model, data, measures=['rmse'], cv=k, n_jobs=-1, verbose=True)

def grid(algo_class, data, k=10):
    '''
    Performs parameter grid search. See source.py for the parameter grid. The test metric is RMSE.

    Parameters:
    algo_class (surprise.AlgoBase (class)): the class of algorithm to use in the search
    data (surprise.Dataset): the data to use
    k (int): the number of folds. By default 10
    '''
    # Initialize search
    gs = GridSearchCV(algo_class=algo_class, param_grid=source.param_grid, measures=['rmse'], cv=k, joblib_verbose=100, n_jobs=-1)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    results = pd.DataFrame.from_dict(gs.cv_results)
    print(results)

def random_search(algo_class, data, k=10, n_iters=100):
    '''
    Performs parameter random search. See source.py for the parameter distribution. The test metric is RMSE.

    Parameters:
    algo_class (surprise.AlgoBase (class)): the class of algorithm to use in the search
    data (surprise.Dataset): the data to use
    k (int): the number of folds. By default 10
    n_iters (int): the number of iterations. By default 100.
    '''
    # Initialize search
    rs = RandomizedSearchCV(algo_class=algo_class, param_distributions=source.dist_grid, measures=['rmse'], cv=k, joblib_verbose=100, n_iter=n_iters, n_jobs=-1)
    rs.fit(data)
    print(rs.best_score['rmse'])
    print(rs.best_params['rmse'])
    results = pd.DataFrame.from_dict(rs.cv_results)
    print(results)

def dump(model, data, indexes, file_name):
    '''
    Dumps the predictions of the selected model on a csv file.
    Predictions are made on the whole training set.

    Parameters:
    model (surprise.AlgoBase): the model whose predictions need to be computed
    data (surprise.Dataset): the training data
    indexes (list): a list of tuples, where each tuple contains the indexes (u,i) which need to be predicted
    file_name (str): the name of the csv file
    '''
    # Set up (whole) training set
    data_all = data.build_full_trainset()
    # Fit model
    model.fit(data_all)
    # Predictions
    predictions = []
    for index in indexes:
        u, i = index
        r = model.predict(u, i).est
        predictions.append((u, i, r))
    # Dump predictions
    utils.write_predictions_to_csv(predictions, file_name)

if __name__ == '__main__':
    # Argparser parameters
    parser = argparse.ArgumentParser(description='Collaborative Filtering')
    parser.add_argument('computation', type=str, metavar='computation', help='the computation to perform (options: cv, target_cv, kfold, grid, random_search, dump)')
    parser.add_argument('algo_name', type=str, metavar='algo_name', help='the name of the algorithm to use (see prediction.pyx)')
    parser.add_argument('--model_num', type=int, default=-1, help='the number of the model (instance of an algo_class) to use (default: -1)')
    parser.add_argument('--synth', type=bool, default=False, help='whether to add synthetic ratings (default: false)')
    parser.add_argument('--k', type=int, default=10, help='the k for kfold cross-validation (default: 10)')
    parser.add_argument('--n_iters', type=int, default=100, help='the number of iterations to perform in random search (default: 100)')
    parser.add_argument('--verbose', type=bool, default=False, help='whether the algorithm should be verbose (default: false)')

    # Parse arguments
    args = parser.parse_args()

    # Set seed
    np.random.seed(666)

    # Read data as a DataFrame
    df = utils.read_data_as_data_frame(source.TRAIN_DATA_PATH)
    # Read list of indexes (to predict)
    indexes = utils.read_submission_indexes(source.PREDICTION_INDEXES_PATH)

    # Add synthetic ratings (if selected)
    if args.synth:
        df = synthetic_ratings(df, indexes)

    # Set up training set
    reader = Reader()
    training_set = Dataset.load_from_df(df[['row', 'col', 'Prediction']], reader)

    # Select algo class
    algo_class = source.algo_classes[args.algo_name]

    # Select model (if any)
    if args.model_num > 0:
        model = source.instances[args.algo_name][args.model_num]
    else:
        model = None

    # Set verbosity of model (if selected)
    if model is not None:
        model.verbose = args.verbose

    # Perform requested computation
    if args.computation == 'cv':
        cv(model, training_set)
    elif args.computation == 'target_cv':
        target_cv(model, training_set)
    elif args.computation == 'kfold':
        kfold(model, training_set, k=args.k)
    elif args.computation == 'grid':
        grid(algo_class, training_set, k=args.k)
    elif args.computation == 'random_search':
        random_search(algo_class, training_set, k=args.k, n_iters=args.n_iters)
    elif args.computation == 'dump':
        file_name = source.NEW_PREDICTIONS_DIR + args.algo_name.lower() + '-' + str(args.model_num) + '.csv'
        dump(model, data, indexes, file_name)
