import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
import csv

def read_data_as_matrix(file_path):
    '''
    Reads the data from the given file path and return a sparse matrix holding the data.

    Parameters:
    file_path (string): the file path to read

    Returns:
    X (scipy.sparse.dok_matrix): sparse matrix representing the data
    '''

    data_train_raw = pd.read_csv(file_path).values

    X = dok_matrix((10000, 1000))

    for i in range(0, data_train_raw.shape[0]):
        indices, value = data_train_raw[i,0], int(data_train_raw[i,1])
        indices = indices.split('_')
        row, column = int(indices[0][1:])-1, int(indices[1][1:])-1 # indexing in the data starts from 1
        X[row, column] = value

    return X

def read_submission_indexes(file_path):
    '''
    Reads the indexes for which to make the predictions.

    Parameters:
    file_path (string): the file path to read

    Returns:
    indexes (list): a list where each element is a tuple of the form (row, column)
    '''

    data_train_raw = pd.read_csv(file_path).values

    tuples = []

    for i in range(data_train_raw.shape[0]):
        indices, value = data_train_raw[i,0], int(data_train_raw[i,1])
        indices = indices.split('_')
        row, column = int(indices[0][1:])-1, int(indices[1][1:])-1 # indexing in the data starts from 1
        tuples.append((row, column))

    return tuples

def read_data_as_data_frame(file_path):
    '''
    Reads the data from the given file path as a data frame. Each row of the data frame has the
    form (id, row, columns, value).

    Parameters:
    file_path (string): the file path to read

    Returns:
    tuples (pandas.DataFrame): array of tuples representing the data, where each tuple t_i has the format (row_i, column_i, value_i)
    '''

    data_train_raw = pd.read_csv(file_path)

    # Parse rows and columns
    row_str = data_train_raw['Id'].apply(lambda x: x.split('_')[0])
    row_id = row_str.apply(lambda x: int(x.split('r')[1]) - 1)
    col_str = data_train_raw['Id'].apply(lambda x: x.split('_')[1])
    col_id = col_str.apply(lambda x: int(x.split('c')[1]) - 1)

    # Apply changes
    data_train_raw['row'] = row_id
    data_train_raw['col'] = col_id

    # Training data as data frame
    data_train_df = data_train_raw.loc[:,['row', 'col', 'Prediction']]

    return data_train_df

def data_frame_to_sparse_matrix(df):
    '''
    Converts the given data frame into a sparse matrix

    Parameters:
    df (pandas.DataFrame): the data frame to transform

    Returns:
    X (scipy.sparse.dok_matrix): the corresponding sparse matrix
    '''

    X = dok_matrix((10000, 1000))

    # print(df.shape[0])

    for i in np.arange(df.shape[0]):
        row, col = int(df.iloc[i,1]), int(df.iloc[i,2])
        pred = int(df.iloc[i,3])
        X[row, col] = pred

    return X

def write_predictions_to_csv(X_pred, indexes, file_path):
    '''
    Writes on a csv file the predictions. The format of the prediction file is the following (second row is example):

    Id, Prediction
    r1_c1, 1

    Parameters:
    X_pred (numpy.ndarray): the prediction matrix
    indexes (list): the list of the indexes. Each element is a tuple of the form (row, column)
    file_path (string): the path to the prediction file
    '''

    # Define header
    header = []
    header.append('Id')
    header.append('Prediction')

    # Write file
    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for index in indexes:
            # Row of the csv file
            file_row = {}
            # Extract indexes
            row, col = index
            # Build file row
            file_row['Id'] = 'r'+ str(row) + '_c' + str(col)
            file_row['Prediction'] = X_pred[row,col]
            # Write file row
            writer.writerow(file_row)
