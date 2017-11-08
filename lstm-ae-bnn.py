import numpy as np
import pandas as pd
import math

from models import Seq2Seq
from bnn import BNN


DATA_FOLDER = 'data/'


def import_csv(ds_name):
    if ds_name == 'fisher':
        filename = 'mean-daily-temperature-fisher.csv'
        return pd.read_csv(DATA_FOLDER + filename, dtype={'date': str, 'temp': float}, parse_dates=['temp'])
    return None


class LSTMAEBNN:
    def __init__(self):
        pass

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
        self.bnn = BNN(n_samples=X.shape[0], X.shape[1], len(actions), bnn_hdim, bnn_hlayer)
        self.bnn.fit(X, Y, iterations=iterations)

    def fit(self, X_train, epochs=10, hidden_dim=5, hidden_layers=1, bnn_hdim=10, bnn_hlayer=1, bnn_epochs=1000,
            window_size=10, overlap=0, verbose=False):
        """
        X_train: (n_actions,)
        """
        # Split the training data into overlapping batches, based on a sliding window
        batches = self._sliding_batches(X_train, window_size)

        # Train a LSTM Autoencoder (seq2seq)
        self.model, self.encoder = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim, output_length=window_size,
                        output_dim=batches.shape[2], depth=hidden_layers)
        self.model.compile(loss='mse', optimizer='adam')  # 'rmsprop'
        self.model.fit(batches, batches, epochs=epochs)

        # self.model.get_layer('recurrentseq').get_weights()
        if verbose:
            print(self.model.summary())

        # Latent variables (n_samples, n_hidden, d_in)
        latent = self.encoder.predict(X_train)
        print(latent.shape)

        # BNN classification
        self._fit_bnn(set(X_train), latent[:-1], batches[1:, 0], bnn_hdim=bnn_hdim,
                      bnn_hlayer=bnn_hlayer, iterations=bnn_epochs)

        return self

    def predict_seq(self, X_test, metric='distance'):
        """
        (n_samples): Return the reconstruction error for each action sequence given in input
        A high reconstruction error denotes an anomaly
        """
        # TODO: input data should be formatted in sliding window batches
        out = self.model.predict(X_test)
        if metric == 'distance':
            return 0.5 * tf.reduce_sum(tf.pow(tf.subtract(out, X_test), 2.0), 1)
        elif metric == 'norm':
            return np.linalg.norm(out - X_test, axis=1)

    def predict_action(self, X_test):
        """
        X_test: (n_samples, sliding_window_size) action sequences of same size
        (n_samples, n_possible_actions): Return the probabilities for the next action
        """
        # TODO: Big drawback here! We only use the sliding window instead of the whole action
        # sequence to predict the next actions
        return self.bnn.predict(self.encoder.predict(X_test))


if __name__ == "__main__":
    data = import_csv('fisher')
    t_data = np.array(data['temp'], dtype=float)[np.newaxis].T
    model = LSTMAEBNN().fit(t_data, epochs=100, hidden_dim=5, depth=1, bnn_hdim=100, bnn_hlayer=1, bnn_epochs=1000
                            window_size=10, overlap=3, verbose=False)
