import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.initializers import RandomNormal
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Dense, Reshape, Concatenate, Flatten, Dropout, Multiply
from keras.regularizers import l2
from surprise import AlgoBase, Dataset, PredictionImpossible
from tqdm import tqdm

class NeuMF(AlgoBase):
    '''
    Implementation of the NeuMF model.
    '''

    def __init__(self, n_features=10, n_epochs=10, lr=0.001, gmf_u_reg=0.05, gmf_i_reg=0.05, mlp_u_reg=0.05, mlp_i_reg=0.05, batch_size=32, verbose=True):
        '''
        Initializes the class with the given parameters.

        Parameters:
        n_features (int): the number of latent features. By default 10
        n_epochs (int): the number of iterations. By default 10
        lr (float): the learning rate. By default 0.001
        gmf_u_reg (float): the regularization for the user bias in the GMF embedding layer. By default 0.05
        gmf_i_reg (float): the regularization for the item bias in the GMF embedding layer. By default 0.05
        mlp_u_reg (float): the regularization for the user bias in the MLP embedding layer. By default 0.05
        mlp_i_reg (float): the regularization for the item bias in the MLP embedding layer. By default 0.05
        batch_size (int): the batch size. By default 256
        verbose (boolean): whether the algorithm should be verbose. By default False
        '''

        AlgoBase.__init__(self)

        self.n_features = n_features
        self.n_epochs = n_epochs
        self.lr = lr
        self.gmf_u_reg = gmf_u_reg
        self.gmf_i_reg = gmf_i_reg
        self.mlp_u_reg = mlp_u_reg
        self.mlp_i_reg = mlp_i_reg
        self.batch_size = batch_size
        self.verbose = verbose

        self.dense_layers = [256, 128, 64, 64, 64, 32, 16, 8]
        self.reg_activ = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.reg_bias = [0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.02]

        self.model = None

    def build_network(self):
        '''
        Builds the deep neural network.
        '''

        user_input = tf.compat.v1.keras.Input(shape=(1,), dtype='int32', name='user_input')
        item_input = tf.compat.v1.keras.Input(shape=(1,), dtype='int32', name='item_input')

        gmf_u_embedding = tf.compat.v1.keras.layers.Embedding(input_dim=10000, output_dim=self.n_features, name='gmf_u_embedding', embeddings_initializer=tf.random.normal, embeddings_regularizer=l2(self.gmf_u_reg), input_length=1)
        gmf_i_embedding = tf.compat.v1.keras.layers.Embedding(input_dim=1000, output_dim=self.n_features, name='gmf_i_embedding', embeddings_initializer=tf.random.normal, embeddings_regularizer=l2(self.gmf_i_reg), input_length=1)

        mlp_u_embedding = tf.compat.v1.keras.layers.Embedding(input_dim=10000, output_dim=int(self.dense_layers[0]/2), name='mlp_u_embedding', embeddings_initializer=tf.random.normal, embeddings_regularizer=l2(self.mlp_u_reg), input_length=1)
        mlp_i_embedding = tf.compat.v1.keras.layers.Embedding(input_dim=1000, output_dim=int(self.dense_layers[0]/2), name='mlp_i_embedding', embeddings_initializer=tf.random.normal, embeddings_regularizer=l2(self.mlp_i_reg), input_length=1)

        # GMF vectors
        gmf_u_latent = tf.compat.v1.keras.layers.Flatten()(gmf_u_embedding(user_input))
        gmf_i_latent = tf.compat.v1.keras.layers.Flatten()(gmf_i_embedding(item_input))
        gmf_vector = tf.compat.v1.keras.layers.Multiply()([gmf_u_latent, gmf_i_latent])

        # MLP vectors
        mlp_u_latent = tf.compat.v1.keras.layers.Flatten()(mlp_u_embedding(user_input))
        mlp_i_latent = tf.compat.v1.keras.layers.Flatten()(mlp_i_embedding(item_input))
        mlp_vector = tf.compat.v1.keras.layers.Concatenate()([mlp_u_latent, mlp_i_latent])

        # MLP layers
        for i in range(len(self.dense_layers)):
            layer = tf.compat.v1.keras.layers.Dense(self.dense_layers[i], activation='relu', use_bias=True, bias_regularizer=l2(self.reg_bias[i]), activity_regularizer=l2(self.reg_activ[i]))
            mlp_vector = layer(mlp_vector)

        # Predict vector
        predict_vector = tf.compat.v1.keras.layers.Concatenate()([gmf_vector, mlp_vector])

        # Prediction layer
        result = tf.compat.v1.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='result')

        # Define model
        self.model = tf.compat.v1.keras.Model(inputs=[user_input, item_input], outputs=result(predict_vector))

        # Compile model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss='binary_crossentropy', metrics=['mse'])

    def fit(self, trainset):
        '''
        Fits the model to the provided dataset.

        Parameters:
        trainset (surprise.Trainset): the training set to be fitted
        '''

        AlgoBase.fit(self, trainset)

        # Read training set
        self.trainset = trainset

        # Build neural network
        self.build_network()

        # Model summary
        if self.verbose:
            print(self.model.summary())

        # Execute neural network
        self.train()

        return self

    def train(self):
        '''
        Optimizes the model by calling the neural network.
        '''

        # Extract training data
        users, items, ratings = [], [], []

        for u, i, r in self.trainset.all_ratings():
            users.append(u)
            items.append(i)
            ratings.append(r)

        # Numpify lists
        users = np.array(users)
        items = np.array(items)
        ratings = np.array(ratings)

        # Train model
        self.model.fit([users, items], ratings, batch_size=self.batch_size, epochs=self.n_epochs, verbose=self.verbose, shuffle=True)

    def estimate(self, u, i):
        '''
        Returns the prediction for the given user and item

        Parameters
        u (int): the user index
        i (int): the item index

        Retuns:
        rui (float): the prediction
        '''

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            ua, ia = np.array(u), np.array(i)
            ua, ia = np.reshape(ua, (1,)), np.reshape(ia, (1,))
            rui = self.model.predict([ua, ia])
        else:
            raise PredictionImpossible('User and item are unknown.')

        return rui
