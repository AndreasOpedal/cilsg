import pandas as pd
import numpy as np
import csv
import math
import os

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

def write_predictions_to_csv(predictions, file_path):
    '''
    Writes on a csv file the predictions. The format of the prediction file is the following (second row is example):

    Id, Prediction
    r1_c1, 1

    Parameters:
    predictions (surprise.prediction_algorithms.predictions.Prediction): the object holding the predictions
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
        for row, col, pred in predictions:
            # Row of the csv file
            file_row = {}
            # Build file row
            file_row['Id'] = 'r'+ str(row+1) + '_c' + str(col+1)
            file_row['Prediction'] = pred
            # Write file row
            writer.writerow(file_row)
