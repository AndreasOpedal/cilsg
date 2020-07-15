import numpy as np
import pandas as pd
import source

class Ensemble:
    '''
    Implementation of an ensemble model, which makes predictions based on the average of SVDPP2, PLSA, and VAE.
    '''

    def __init__(self, svdpp2_file='../predictions/svdpp2-1.csv', plsa_file='../predictions/plsa-1.csv', vae_file='../predictions/vae-1.csv'):
        '''
        Initializes the class with the given parameters.

        Parameters:
        svdpp2_file (str): the file holding the predictions for the SVDPP2 model. By default ../predictions/svdpp2-1.csv
        plsa_file (str): the file holding the predictions for the PLSA model. By default ../predictions/plsa-1.csv
        vae_file (str): the file holding the predictions for the VAE model. By default ../predictions/vae-1.csv
        '''

        self.svdpp2_file = svdpp2_file
        self.plsa_file = plsa_file
        self.vae_file = vae_file

        self.df_submission = None

    def clean_df(self, df):
        '''
        Cleans initial representation to separate rows (users) and columns (movies) into columns with integer values

        Parameters:
        df (pandas.DataFrame): the dataframe to clean

        Returns:
        data_df (pandas.DataFrame): the cleaned dataframe
        '''

        row_str = df["Id"].apply(lambda x: x.split("_")[0])
        row_id = row_str.apply(lambda x: int(x.split("r")[1]) - 1)
        col_str = df["Id"].apply(lambda x: x.split("_")[1])
        col_id = col_str.apply(lambda x: int(x.split("c")[1]) - 1)

        data_df = pd.DataFrame(data = {'row': row_id, 'col': col_id, 'Prediction': df.loc[:,'Prediction']})

        return data_df

    def average(self):
        '''
        Computes the average ratings for the three base models.
        '''

        # Read predictions
        df_svdpp2 = pd.read_csv(self.svdpp2_file)
        df_plsa = pd.read_csv(self.plsa_file)
        df_vae = pd.read_csv(self.vae_file)

        # Read submission indexes
        self.df_submission = pd.read_csv(source.PREDICTION_INDEXES_PATH)

        # Clean dataframes
        df_svdpp2 = self.clean_df(df_svdpp2)
        df_plsa = self.clean_df(df_plsa)
        df_vae = self.clean_df(df_vae)
        preds_submission = self.clean_df(self.df_submission)

        # Sort dataframes in the same way
        df_svdpp2.sort_values(by=['col','row'], inplace=True)
        df_plsa.sort_values(by=['col','row'], inplace=True)
        df_vae.sort_values(by=['col','row'], inplace=True)
        preds_submission.sort_values(by=['col','row'], inplace=True)

        # Compute average
        # preds = np.mean(np.array([df_svdpp2['Prediction'].values, df_plsa['Prediction'].values, df_vae['Prediction'].values]), axis=0)
        preds = np.mean(np.array([df_svdpp2['Prediction'].values, df_vae['Prediction'].values]), axis=0)


        # Write predictions into prediction dataframe
        preds_submission['Prediction'] = preds

        # Rewrite predictions to submission dataframe
        self.df_submission['Prediction'] = preds_submission['Prediction']
