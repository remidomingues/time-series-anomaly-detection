import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.preprocessing import OneHotEncoder

from models import Seq2Seq
from bnn import BNN


DATA_FOLDER = 'data/'
TRAINING_PERCENT = 0.8


def import_dataset(ds_name):
    if ds_name == 'fisher':
        filename = 'mean-daily-temperature-fisher.csv'
        data = pd.read_csv(DATA_FOLDER + filename, dtype={'date': str, 'temp': float}, parse_dates=['temp'])
        return np.array(data['temp'], dtype=float)[np.newaxis].T
    elif ds_name == 'diseases':
        data = np.array([0, 1, 16, 15, 19, 0, 16, 0, 16, 33, 19, 16, 19, 8, 40, 40, 40, 40, 33, 40, 16, 40, 33, 16, 33, 16, 33, 40, 28, 39, 1, 0, 1, 0, 40, 8, 40, 16, 33, 19, 40, 40, 40, 33, 11, 16, 16, 19, 20, 12, 40, 1, 5, 22, 27, 23, 33, 16, 16, 16, 33, 16, 16, 28, 1, 1, 1, 12, 16, 12, 16, 40, 16, 16, 33, 33, 33, 15, 33, 33, 33, 33, 16, 33, 16, 16, 33, 33, 16, 16, 16, 12, 23, 8, 20, 16, 33, 33, 16, 9, 33, 16, 33, 16, 33, 16, 16, 16, 11, 10, 16, 1, 16, 16, 40, 16, 23, 23, 16, 33, 40, 0, 8, 8, 33, 11, 16, 8, 16, 33, 33, 33, 30, 15, 11, 7, 16, 40, 23, 8, 40, 8, 40, 2, 40, 40, 16, 0, 16, 33, 40, 16, 34, 16, 38, 40, 40, 1, 23, 23, 40, 23, 40, 40, 29, 31, 15, 40, 40, 40, 40, 28, 8, 33, 40, 33, 8, 33, 40, 33, 16, 40, 16, 33, 16, 40, 33, 0, 16, 16, 16, 16, 16, 16, 16, 1, 16, 40, 1, 1, 40, 16, 1, 40, 23, 40, 40, 33, 33, 40, 23, 28, 33, 19, 8, 40, 33, 16, 40, 31, 33, 31, 40, 33, 18, 1, 40, 16, 39, 16, 40, 40, 40, 1, 40, 16, 40, 29, 29, 31, 33, 33, 33, 33, 16, 16, 16, 16, 16, 7, 19, 41, 19, 16, 40, 15, 16, 33, 8, 40, 16, 33, 16, 16, 33, 16, 16, 16, 16, 1, 39, 7, 3, 23, 18, 16, 8, 40, 16, 16, 1, 39, 16, 23, 14, 16, 33, 33, 33, 33, 16, 1, 16, 16, 15, 12, 16, 1, 36, 12, 6, 19, 33, 33, 33, 33, 33, 33, 33, 31, 16, 16, 16, 28, 16, 33, 33, 28, 16, 31, 23, 20, 40, 16, 16, 38, 16, 1, 35, 16, 7, 19, 8, 16, 16, 15, 40, 39, 7, 1, 18, 36, 41, 0, 16, 16, 40, 35, 31, 32, 8, 16, 33, 8, 40, 16, 33, 16, 33, 33, 12, 33, 16, 16, 33, 16, 0, 1, 1, 0, 31, 1, 1, 36, 1, 0, 20, 33, 33, 33, 18, 40, 33, 20, 33, 16, 16, 40, 1, 15, 35, 39, 29, 16, 33, 33, 33, 16, 16, 16, 16, 40, 16, 31, 15, 1, 16, 16, 16, 40, 16, 40, 40, 16, 16, 16, 8, 16, 16, 16, 38, 0, 1, 16, 19, 40, 40, 23, 12, 40, 40, 40, 40, 40, 33, 16, 16, 16, 33, 16, 16, 36, 18, 16, 16, 1, 31, 18, 40, 40, 8, 40, 40, 33, 40, 31, 40, 40, 16, 40, 37, 16, 1, 27, 1, 19, 7, 16, 16, 40, 16, 19, 16, 25, 16, 1, 15, 38, 28, 18, 0, 40, 13, 40, 40, 16, 16, 40, 16, 18, 16, 31, 16, 25, 19, 17, 16, 16, 28, 1, 19, 19, 38, 16, 37, 17, 0, 29, 40, 0, 16, 19, 20, 28, 40, 16, 8, 23, 18, 29, 17, 20, 18, 40, 33, 16, 29, 16, 39, 8, 36, 19, 38, 40, 16, 16, 8, 16, 23, 29, 33, 10, 40, 39, 16, 29, 29, 17, 17, 31, 36, 38, 11, 37, 18, 40, 16, 25, 25, 25, 23, 40, 33, 16, 8, 33, 16, 26, 20, 37, 38, 29, 18, 16, 29, 29, 11, 29, 40, 2, 16, 16, 16, 0, 16, 19, 18, 17, 16, 17, 16, 33, 16, 21, 18, 8, 16, 8, 40, 33, 16, 16, 16, 36, 1, 1, 28, 16, 16, 16, 33, 29, 40, 16, 33, 33, 1, 16, 16, 18, 16, 12, 29, 38, 39, 20, 16, 33, 13, 8, 40, 16, 38, 16, 16, 25, 23, 1, 16, 13, 8, 8, 40, 38, 5, 0, 1, 16, 1, 16, 16, 16, 37, 29, 16, 8, 16, 16, 7, 0, 38, 40, 15, 40, 20, 16, 8, 16, 1, 16, 16, 25, 37, 38, 16, 13, 16, 8, 16, 13, 16, 18, 1, 1, 1, 31, 1, 37, 18, 16, 40, 39, 16, 40, 21, 19, 36, 16, 16, 16, 38, 38, 16, 1, 28, 16, 33, 35, 39, 16, 28, 1, 28, 16, 18, 23, 23, 8, 40, 16, 16, 16, 16, 16, 20, 16, 31, 19, 36, 37, 16, 16, 31, 16, 16, 16, 23, 35, 16, 40, 19, 16, 20, 40, 1, 4, 16, 38, 16, 16, 1, 16, 38, 28, 28, 16, 23, 40, 16, 16, 16, 16, 16, 7, 7, 31, 1, 40, 23, 8, 39, 16, 16, 16, 40, 40, 20, 40, 40, 16, 0, 16, 16, 8, 16, 16, 38, 16, 16, 40, 39, 40, 16, 1, 16, 38, 38, 28, 16, 16, 16, 16, 1, 31, 23, 23, 40, 8, 39, 39, 16, 16, 1, 16, 36, 22, 8, 20, 16, 16, 16, 1, 16, 1, 28, 40, 16, 16, 16, 39, 1, 28, 38, 16, 23, 40, 39, 16, 1, 38, 40, 1, 8, 38, 40, 20, 31, 16, 16, 20, 1, 24, 21, 31, 38, 38, 20, 28, 36, 39, 39, 13, 38, 16, 1, 1, 39, 23, 3, 38, 38, 39, 20, 2, 20, 16, 38, 38, 39, 23, 23, 16, 16, 38, 16, 16, 40, 16, 40, 38, 39, 31, 16, 16, 17, 31, 3, 38, 23, 39, 36, 1, 28, 38, 28, 27, 24, 1, 8, 38, 39, 39, 20, 9, 16, 19, 20, 31, 16, 16, 39, 36, 33, 16, 23, 23, 20, 16, 8, 2, 40, 8, 0, 15, 16, 31, 16, 12, 16, 16, 12, 8, 40, 16, 16, 16, 16, 31, 35, 13, 11, 16, 16, 33, 16, 16, 12, 16, 16, 16, 1, 20, 40, 8, 31, 1, 31, 20, 12, 12, 28, 23, 1, 23, 16, 33, 1, 40, 28, 38, 27, 33, 39, 16, 16, 40, 12, 33, 2, 27, 16, 16, 40, 9, 16, 1, 16, 37, 11, 16, 16, 9, 38, 39, 40, 39, 31, 28, 3, 16, 16, 38, 16, 16, 23, 15, 28, 16, 13, 38, 16, 31, 16, 33, 39, 8, 16, 31, 39, 1, 28, 38, 8, 16, 38, 38, 21, 31, 31, 15, 38, 21, 16, 31, 31, 38, 24, 33, 16, 28, 16, 39, 27, 16, 31, 31, 21, 1, 40, 38, 40, 2, 8, 33, 39, 39, 36, 12, 38, 39, 16, 28, 39, 24, 21, 39, 1, 16, 31, 31, 39, 39, 31, 31, 16, 31, 31, 27, 31])
        return OneHotEncoder(sparse=False).fit_transform(data[np.newaxis].T)
    return None


class LSTMAEBNN:
    def __init__(self, window_size=10, overlap=0):
        self.sequence_length = window_size
        self.window_overlap = overlap

    def _sliding_batches(self, X, window_size, overlap=0):
        """
        overlap: if 0, X will be chunked into sequences without overlap
        """
        return np.array([X[min(start, X.shape[0]-window_size):start+window_size]
                        for start in range(0, X.shape[0], window_size)])

    def _fit_bnn(self, actions, X, Y, bnn_hdim=10, bnn_hlayer=1, iterations=1000):
        """
        Fit a Bayesian Neural Network in order to predict the next action given the latent representation
        of an action sequence
        """
        self.bnn = BNN(X.shape[0], X.shape[1], len(actions), bnn_hdim, bnn_hlayer)
        self.bnn.fit(X, Y, iterations=iterations)

    def fit(self, X_train, epochs=10, hidden_dim=5, hidden_layers=1, bnn_hdim=10, bnn_hlayer=1, bnn_epochs=1000,
            verbose=False):
        """
        X_train: (n_actions,)
        """
        # Split the training data into overlapping batches, based on a sliding window
        batches = self._sliding_batches(X_train, self.sequence_length, self.window_overlap)

        # Train a LSTM Autoencoder (seq2seq)
        self.model, self.encoder = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim,
                                           output_length=self.sequence_length, output_dim=batches.shape[2], depth=hidden_layers)
        self.model.compile(loss='mse', optimizer='adam')  # 'rmsprop'
        self.model.fit(batches, batches, epochs=epochs)

        # self.model.get_layer('recurrentseq').get_weights()
        if verbose:
            print(self.model.summary())

        # Latent variables (n_samples, n_hidden, d_in)
        # latent = self.encoder.predict(X_train)

        # BNN classification
        # self._fit_bnn(set(X_train), latent[:-1], batches[1:, 0], bnn_hdim=bnn_hdim,
        #               bnn_hlayer=bnn_hlayer, iterations=bnn_epochs)

        return self

    def predict_seq(self, X_test, metric='distance'):
        """
        (n_samples): Return the reconstruction error for each action sequence given in input
        A high reconstruction error denotes an anomaly
        """
        batches = self._sliding_batches(X_test, self.sequence_length, self.window_overlap)
        print(batches.shape)
        out = self.model.predict(batches)
        print(out.shape)
        # TODO: issue, how do we compare the reconstruction error, given that the sequence is a matrix of events,
        # where each event is a binary vector? We could to it, though the approach looks more and more wrong
        if metric == 'distance':
            batches_error = np.array([0.5 * tf.reduce_sum(tf.pow(tf.subtract(i, o), 2.0), 1)
                                     for i, o in zip(X_test, out)])
        elif metric == 'norm':
            batches_error = np.array([np.linalg.norm(i - o, axis=1)
                                     for i, o in zip(X_test, out)])
        else:
            ValueError('Unknown distance metric: {}'.format(metric))
        return batches_error.mean(), batches_error

    def predict_action(self, X_test):
        """
        X_test: (n_samples, sliding_window_size) action sequences of same size
        (n_samples, n_possible_actions): Return the probabilities for the next action
        """
        # TODO: Big drawback here! We only use the sliding window instead of the whole action
        # sequence to predict the next actions
        pass
        # return self.bnn.predict(self.encoder.predict(X_test))


if __name__ == "__main__":
    data = import_dataset('diseases')
    X_train, X_test = data[:int(data.shape[0]*TRAINING_PERCENT)], data[int(data.shape[0]*TRAINING_PERCENT):]
    model = LSTMAEBNN(window_size=10, overlap=0)
    model.fit(X_train, epochs=10, hidden_dim=5, hidden_layers=1, bnn_hdim=100, bnn_hlayer=1, bnn_epochs=1000, verbose=True)
    model.predict_seq(X_test, metric='norm')
