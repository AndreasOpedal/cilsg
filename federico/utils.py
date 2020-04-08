import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix

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

def read_data_as_tuples(file_path):
    '''
    Reads the data from the given file path as an array of tuples holding the indexes and values of the data.

    Parameters:
    file_path (string): the file path to read

    Returns:
    tuples (numpy.array): array of tuples representing the data, where each tuple t_i has the format (row_i, column_i, value_i)
    '''

    data_train_raw = pd.read_csv(file_path).values

    tuples = np.array((data_train_raw.shape[0], 1))

    for i in range(data_train_raw.shape[0]):
        indices, value = data_train_raw[i,0], int(data_train_raw[i,1])
        indices = indices.split('_')
        row, column = int(indices[0][1:])-1, int(indices[1][1:])-1 # indexing in the data starts from 1
        tuples[i] = (row, column, value)

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
