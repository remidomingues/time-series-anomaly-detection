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

    def sliding_batches(self, X, window_size, overlap=0):
        """
        overlap: if 0, X will be chunked into sequences without overlap
        """
        return np.array([X[min(start, X.shape[0]-window_size):start+window_size]
                        for start in range(0, X.shape[0], window_size)])


    def fit(self, X_train, epochs=10, hidden_dim=5, depth=1, window_size=10, overlap=0):
        batches = self.sliding_batches(X_train, window_size)

        self.model, self.encoder = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim, output_length=window_size,
                        output_dim=batches.shape[2], depth=depth)
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(batches, batches, epochs=epochs)


        # Temporary check: encoder weights should be equal to model weights (?)
        a = self.model.get_layer('recurrentseq').get_weights()
        b = self.encoder.get_layer('recurrentseq').get_weights()
        assert all([np.allclose(x, y) for x, y in zip(a, b)])

        """ TODO:
        Once the seq2seq (LSTM-AE) is trained, feed testing data, obtain N encoded latent vectors, feed them to the BNN
        as X, use the next action (or actually a binary vector of #number of possible actions) as Y. Prediction wil lbe
        the likelihood of the next action
        """
        # Do that here

        #Note: how do we predict the likelihood of an entire action sequence?
        #=> We can't, except if we use a variational autoencoder between to encode the latent variables, and then get the likel;ihood

        return self.model

    def predict(self, X_test):
        pass


if __name__ == "__main__":
    data = import_csv('fisher')
    t_data = np.array(data['temp'], dtype=float)[np.newaxis].T
    model = LSTMAEBNN().fit(t_data, epochs=100, hidden_dim=5, depth=1, window_size=10, overlap=3)
