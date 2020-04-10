import numpy as np
import numpy.matlib
import pandas as pd
from tqdm.auto import tqdm
import math
import copy
from sklearn.metrics import mean_squared_error

def separate_in_folds(df, k = 5):
	'''
	Separate data in k folds to be used for cross validation
	Data is shuffled by user to make sure that each user is represented in each fold to the extent possible
	'''

	# Create empty df to fill with shuffled data
	shuffled_df = pd.DataFrame(columns = ["row", "col", "Prediction"])

    # Shuffle data by row
	for i in tqdm(np.arange(10000)):
		num_rows = df[df['row'] == i].shape[0]
		row_sample = df[df['row'] == i].sample(n=num_rows)
		shuffled_df = shuffled_df.append(row_sample)

	# Assign folds to rows
	folds = np.append(np.matlib.repmat(np.arange(k)+1, 1, math.floor(shuffled_df.shape[0] / k)),
		(np.arange(k)+1)[0:shuffled_df.shape[0] % k])

	shuffled_df["Fold"] = folds

	# Return separate fold data frames in list, drop fold column
	fold_list = []
	for j in np.arange(k)+1:
		fold_list.append(shuffled_df[shuffled_df["Fold"] == j].drop('Fold', axis = 1))

	return fold_list



def clean_df(df):
	'''
	Cleans initial representation to separate rows (users) and columns (movies) into columns with integer values
	'''
	row_str = df["Id"].apply(lambda x: x.split("_")[0])
	row_id = row_str.apply(lambda x: int(x.split("r")[1]) - 1)
	col_str = df["Id"].apply(lambda x: x.split("_")[1])
	col_id = col_str.apply(lambda x: int(x.split("c")[1]) - 1)

	data_df = pd.DataFrame(data = {'row': row_id, 'col': col_id, 'Prediction': df.loc[:,'Prediction']})

	return data_df


def k_fold_cv(folds, param_array, model_function):
	'''
	Performs k fold cross validation-

	Input:
	folds: list of k folds as data frames
	param_array: array containing hyperparameter values to test
	model_function: model function to test. Needs to have data and param input and predictions 
	in full matrix representation as output

	Output:
	Array of mean RMSE scores corresponding to values in param_array

	'''

	RMSE = np.array([])
	for val in param_array:
	    
	    
		# Cross-validation loop

		# Initialize scores array for RMSE's
		scores = np.array([])

		for i in np.arange(len(folds)): 
			# Make a copy of folds list to be able to use pop()
			fold_copy = copy.deepcopy(folds)

			test_set = fold_copy.pop(i)
			train_set = pd.concat([x for j,x in enumerate(fold_copy)])

			# Now train on training set
			# Insert any model for training here
			pred_matrix = model_function(data = train_set, param = val)

			# Predict on test_set

			# Extract predictions
			row_ind = test_set.loc[:,'row']
			col_ind = test_set.loc[:,'col']
			preds = pred_matrix[row_ind, col_ind]

			# Add column to test_set with predictions from model
			test_set['Model Predictions'] = preds

			# Compute RMSE and add value to scores array
			scores = np.append(scores, 
				np.sqrt(mean_squared_error(test_set.loc[:,'Prediction'], test_set.loc[:,'Model Predictions'])))
		
		# Compute mean of scores for estimate of model performance
		RMSE = np.append(RMSE, np.mean(scores))

	return RMSE




def predict_scores(test_set, pred_matrix):
	'''
	Retrieve predictions for test set from prediction matrix

	Input:
	test_set: data frame with columns: row, col and Prediction
	pred_matrix: full rows x cols matrix with predictions outputted from model

	Output:
	test_set: test set with updated predictions in Predictions column
	'''

	row_ind = test_set.loc[:,'row']
	col_ind = test_set.loc[:,'col']
	preds = pred_matrix[row_ind, col_ind]
	test_set['Prediction'] = preds

	return test_set








