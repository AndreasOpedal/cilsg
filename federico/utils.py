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
    Reads the data from the given file path an array of tuples holding the indexes and values of the data.
    
    Parameters: 
    file_path (string): the file path to read
    
    Returns: 
    tuples (numpy.array): array of tuples representing the data, where each tuple t_i has the format (row_i, column_i, value_i)
    '''
    
    data_train_raw = pd.read_csv(file_path).values
    
    tuples = np.array((data_train_raw.shape[0], 1))
    
    for i in range(0, data_train_raw.shape[0]):
        indices, value = data_train_raw[i,0], int(data_train_raw[i,1])
        indices = indices.split('_')
        row, column = int(indices[0][1:])-1, int(indices[1][1:])-1 # indexing in the data starts from 1
        tuples[i] = (row, column, value)
        
    return tuples