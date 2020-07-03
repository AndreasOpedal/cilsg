import argparse
import numpy as np
import pandas as pd
import utils
import source
from surprise import Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection.search import RandomizedSearchCV, GridSearchCV
from surprise.model_selection.split import KFold

def cv(model, data):
    '''
    Performs cross-validation by training on 75% of the provided data.
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

def kfold(model, data, k=10):
    '''
    Perform k-fold cross-validation. The test metric is RMSE.

    Parameters:
    model (surprise.AlgoBase): the model to test
    data (surprise.Dataset): the data to use
    k (int): the number of folds. By default 10
    load (bool): whether the weights of the model should be loaded
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

def dump(model, data, indexes, file_name, load_dir):
    '''
    Dumps the predictions of the selected model on a csv file.
    The model is trained on the whole training set.

    Parameters:
    model (surprise.AlgoBase): the model whose predictions need to be computed
    data (surprise.Trainset): the training data
    indexes (list): a list of tuples, where each tuple contains the indexes (u,i) which need to be predicted
    file_name (str): the name of the csv file
    load_dir (str): the directory from which to load the weights
    '''
    # Fit model
    model.fit(data)
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
    parser.add_argument('exec_mode', type=str, metavar='execution mode', help='the execution mode (options: cv, kfold, grid, random_search, dump, save)')
    parser.add_argument('algo_class', type=str, metavar='algorithm class', help='the algorithm class to use (see names of classes in factorization.pyx and baseline.py)')
    parser.add_argument('--model_num', type=int, default=1, help='the number of the model (instance of an algo_class) to use (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='the k for kfold cross-validation (default: 10)')
    parser.add_argument('--n_iters', type=int, default=10, help='the number of iterations to perform in random search (default: 10)')
    parser.add_argument('--verbose', type=bool, default=False, help='whether the algorithm should be verbose (default: False)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (default: 0)')

    # Parse arguments
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Read data as a DataFrame
    df = utils.read_data_as_data_frame(source.TRAIN_DATA_PATH)
    # Read list of indexes (to predict)
    indexes = utils.read_submission_indexes(source.PREDICTION_INDEXES_PATH)

    # Set up dataset
    reader = Reader()
    dataset = Dataset.load_from_df(df[['row', 'col', 'Prediction']], reader)

    # Select algo class
    algo_class = source.algo_classes[args.algo_class]

    # Select model instance
    model = source.instances[args.algo_class][args.model_num]

    # Set verbosity of model
    if args.verbose:
        model.verbose = args.verbose

    # Perform requested computation
    if args.exec_mode == 'cv':
        cv(model, dataset)
    elif args.exec_mode == 'kfold':
        kfold(model, dataset, k=args.k)
    elif args.exec_mode == 'grid':
        grid(algo_class, dataset, k=args.k)
    elif args.exec_mode == 'random_search':
        random_search(algo_class, dataset, k=args.k, n_iters=args.n_iters)
    elif args.exec_mode == 'dump':
        file_name = source.NEW_PREDICTIONS_DIR + args.algo_class.lower() + '-' + str(args.model_num) + '.csv'
        training_set = dataset.build_full_trainset()
        dump(model, training_set, indexes, file_name, weights_file_path)
    else:
        print('Invalid computation selected.')
