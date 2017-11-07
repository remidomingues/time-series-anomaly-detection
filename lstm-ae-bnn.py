import numpy as np
import pandas as pd
import math

import seq2seq
from seq2seq.models import Seq2Seq


DATA_FOLDER = 'data/'


def import_csv(ds_name):
    if ds_name == 'fisher':
        filename = 'mean-daily-temperature-fisher.csv'
        return pd.read_csv(DATA_FOLDER + filename, dtype={'date': str, 'temp': float}, parse_dates=['temp'])
    return None


def sliding_batches(data, window_size, overlap=0):
    """
    overlap: if 0, data will be chunked into sequences without overlap
    """
    return np.array([data[min(start, data.shape[0]-window_size):start+window_size]
                    for start in range(0, data.shape[0], window_size)])


def train(data, epochs=10, hidden_dim=5, depth=1, window_size=10, overlap=0):
    batches = sliding_batches(data, window_size)

    model = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim, output_length=window_size,
                    output_dim=batches.shape[2], depth=depth)
    model.compile(loss='mse', optimizer='adam')
    model.fit(batches, batches, epochs=epochs)

    """ TODO:
    Once the seq2seq (LSTM-AE) is trained, feed testing data, obtain N encoded latent vectors, feed them to the BNN
    as X, use the next action (or actually a binary vector of #number of possible actions) as Y. Prediction wil lbe
    the likelihood of the next action
    """
    # Do that here

    #Note: how do we predict the likelihood of an entire action sequence?
    #=> We can't, except if we use a variational autoencoder between to encode the latent variables, and then get the likel;ihood

    return model


if __name__ == "__main__":
    data = import_csv('fisher')
    t_data = np.array(data['temp'], dtype=float)[np.newaxis].T
    model = train(t_data, epochs=100, hidden_dim=5, depth=1, window_size=10, overlap=3)
