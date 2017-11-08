import numpy as np
import pandas as pd
import math

from models import Seq2Seq


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


    def _fit_bnn(self, X_train):
        """
        Fit a Bayesian Neural Network in order to predict the next action given the latent representation
        of an action sequence
        """
        # use the next action (or actually a binary vector of #number of possible actions) as Y
        # rediction wil lbe the likelihood of the next action


    def fit(self, X_train, epochs=10, hidden_dim=5, depth=1, window_size=10, overlap=0):
        # Split the training data into overlapping batches, based on a sliding window
        batches = self._sliding_batches(X_train, window_size)

        # Train a LSTM Autoencoder (seq2seq)
        self.model = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim, output_length=window_size,
                        output_dim=batches.shape[2], depth=depth)
        self.model.compile(loss='mse', optimizer='adam')  # 'rmsprop'
        self.model.fit(batches, batches, epochs=epochs)

        # self.model.get_layer('recurrentseq').get_weights()
        # print(self.model.summary())

        # Latent variables (n_samples, n_hidden, d_in)
        latent = encoder.predict(X_train)
        print(latent.shape)

        # BNN classification
        self.fit_bnn(latent, batches)

        return self.model

    def predict(self, X_test):
        #Note: how do we predict the likelihood of an entire action sequence?
        #=> We can't, except if we use a variational autoencoder between to encode the latent variables, and then get the likel;ihood
        # Or we can use the reconstruction error, but then why pass latent vars to BNN instead of raw action seq?
        pass


if __name__ == "__main__":
    data = import_csv('fisher')
    t_data = np.array(data['temp'], dtype=float)[np.newaxis].T
    model = LSTMAEBNN().fit(t_data, epochs=100, hidden_dim=5, depth=1, window_size=10, overlap=3)
