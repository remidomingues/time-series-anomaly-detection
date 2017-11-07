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


def sliding_batches(data, window_size):
    # TODO: IMPROVE!!!! We should have a sliding window, not splitting data!
    # Drops the last elements to have chunks of data of equal size
    trunc_idx = data.shape[0] - data.shape[0]%window_size
    return np.array(np.split(data[:trunc_idx], int(data.shape[0] / window_size)))


def train(data, epochs=10, hidden_dim=5, depth=1, window_size=10):
    batches = sliding_batches(data, window_size)

    model = Seq2Seq(input_dim=batches.shape[2], hidden_dim=hidden_dim, output_length=window_size,
                    output_dim=batches.shape[2], depth=depth)
    model.compile(loss='mse', optimizer='adam')
    model.fit(batches, batches, epochs=epochs)

    return model


if __name__ == "__main__":
    data = import_csv('fisher')
    t_data = np.array(data['temp'], dtype=float)[np.newaxis].T
    model = train(t_data, epochs=100)
