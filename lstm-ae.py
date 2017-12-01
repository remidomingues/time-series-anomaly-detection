import numpy as np
import difflib
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
import math
import pickle

import helpers #for formatting data into batches and generating random sequence data


TRAINING_PERCENT = 0.8


def import_dataset(ds_name):
    if ds_name == 'random':
        return np.array(next(helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=3, vocab_upper=10,
                                   batch_size=10000)))
    return None


class LSTMAE:
    PAD = 0 # Padding character
    BOS = 1 # Begin of sequence, the decoder starts generating events when he gets this character
    EOS = 2 # End of sequence, used to identify a complete sequence generated by the decoder
    ADDITIONAL_CHARS = 3

    MODEL_PATH = 'tmp/seq2seq.ckpt'
    CONFIG_PATH = 'tmp/seq2seq.pickle'


    def __init__(self):
        tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
        self.session = tf.Session()


    def _batches(X, batch_size, dynamic_pad=False, allow_smaller_final_batch=False):
        if dynamic_pad:
            return None
        if allow_smaller_final_batch:
            batches = [X[start:start+batch_size] for start in range(0, len(X), batch_size)]
        else:
            batches = [X[min(start, len(X)-batch_size-1):min(start, len(X)-batch_size-1)+batch_size] for start in range(0, len(X), batch_size)]
            padded_batches, lengths = [], []
            for batch in batches:
                batch_max_length = max([len(seq) for seq in batch])
                b = np.ones((len(batch), batch_max_length), dtype=int) * LSTMAE.PAD
                for i, seq in enumerate(batch):
                    b[i][:len(seq)] = seq

                lengths.append([batch_max_length] * len(batch))
                padded_batches.append(b.swapaxes(0, 1))
                # The two previous lines, or the 4 following
            #     for seq in batch:
            #         for _ in range(batch_max_length - len(seq)):
            #             seq.append(LSTMAE.PAD)
            # batches = [np.array(batch)]
        return padded_batches, lengths


    def _encode_sequences(self, X, batch_size, fit=False):
        """
        X: list of sequences of variable length
        out: batches of sequences. The sequence length is fixed within a given batch
        """
        if fit:
            sets = [set(x) for x in X]
            alphabet = sets[0]
            for s in sets:
                alphabet = alphabet.union(s)
            self.data_encoder = {x: i+LSTMAE.ADDITIONAL_CHARS for i, x in enumerate(alphabet)}
            self.data_encoder[LSTMAE.PAD] = LSTMAE.PAD
            self.data_encoder[LSTMAE.BOS] = LSTMAE.BOS
            self.data_encoder[LSTMAE.EOS] = LSTMAE.EOS
            self.data_decoder = {i: x for x, i in self.data_encoder.items()}
            self.vocab_size = len(alphabet) + LSTMAE.ADDITIONAL_CHARS  # character length (vector): size of the alphabet, i.e. number of unique actions

        X = [[LSTMAE.BOS] + [self.data_encoder[_x] for _x in x] + [LSTMAE.EOS] for x in X]

        # X is transformed into a list of batches. Padding is then added to have sequences of identical length
        # within each batch
        batches, lengths = LSTMAE._batches(X, batch_size, dynamic_pad=False, allow_smaller_final_batch=False)

        # TODO: We may have to swap axes for each batch...
        # inputs_time_major = inputs_batch_major.swapaxes(0, 1) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return batches, lengths


    def _decode_sequences(self, X):
        """
        X: batches of sequences
        """
        return [[self.data_encoder[_x] for _x in x] for x in X]


    def _build_encoder(self, encoder_hidden_units, embedding_size, encoder_dropout=1.):
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        #contains the lengths for each of the sequence in the batch, we will pad so all the same
        #if you don't want to pad, check out dynamic memory networks to input variable length sequences
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
        #replace one-hot encoded heavy matrix representations
        encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        encoder_cell =  LSTMCell(encoder_hidden_units)
        encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=encoder_dropout, input_keep_prob=1.0, state_keep_prob=1.0)

        ((encoder_fw_outputs, encoder_bw_outputs),
         (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell,
                cell_bw=encoder_cell,
                inputs=encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32, time_major=True)

        #Concatenates tensors along one dimension.
        self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        #letters h and c are commonly used to denote "output value" and "cell state".
        #http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #Those tensors represent combined internal state of the cell, and should be passed together.
        encoder_final_state_c = tf.concat(
            (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

        encoder_final_state_h = tf.concat(
            (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        #TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
        self.encoder_final_state = LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )


    def _build_decoder_cell(self, decoder_hidden_units, attention, decoder_dropout=1.):
        self.decoder_cell = LSTMCell(decoder_hidden_units)
        self.decoder_cell = tf.contrib.rnn.DropoutWrapper(self.decoder_cell, output_keep_prob=decoder_dropout, input_keep_prob=1.0, state_keep_prob=1.0)
        encoder_max_length, self.batch_size = tf.unstack(tf.shape(self.encoder_inputs))

        decoder_inputs = tf.ones([encoder_max_length, self.batch_size], dtype=tf.int32)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)

        self.encoder_final_state_attention = None
        if attention:
            # attention_states: [batch_size, max_time, num_units]
            attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=decoder_hidden_units,
                memory=attention_states,
                #memory_sequence_length=None # default value
            )

            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=self.decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=decoder_hidden_units)

            attention_zero = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.encoder_final_state_attention = attention_zero.clone(cell_state=self.encoder_final_state)


    def _build_decoder(self, train=True):
        self.projection_layer = layers_core.Dense(self.vocab_size, use_bias=True)

        if train:
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.decoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                time_major=True)
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                  embedding=self.embeddings,
                  start_tokens=tf.tile([LSTMAE.BOS], [self.batch_size]),
                  end_token=LSTMAE.EOS)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self.decoder_cell,
            helper=helper,
            initial_state=self.encoder_final_state_attention if self.encoder_final_state_attention else encoder_final_state,
            output_layer=self.projection_layer)

        # Dynamic decoding
        (decoder_outputs, final_state, final_sequence_lengths) = tf.contrib.seq2seq.dynamic_decode(
            decoder, output_time_major=True, maximum_iterations=100)
        if train:
            self.train_decoder_logits = decoder_outputs.rnn_output
            self.train_decoder_prediction = decoder_outputs.sample_id
        else:
            self.infer_decoder_prediction = decoder_outputs.sample_id


    def _build_network(self, train, hidden_units=16, embedding_size=20, attention=True, encoder_dropout=1., decoder_dropout=1.):
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        self._build_encoder(hidden_units, embedding_size, encoder_dropout)
        self._build_decoder_cell(hidden_units * 2, attention, decoder_dropout)  # Twice the size, as the decoder is bidirectional
        self._build_decoder(train)

        if train:
            #cross entropy loss
            #one hot encode the target values so we don't rank just differentiate
            stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
                logits=self.train_decoder_logits,
            )

            #loss function
            self.loss = tf.reduce_mean(stepwise_cross_entropy)
            #train it
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss)

            self.session.run(tf.global_variables_initializer())


    def fit(self, X_train, hidden_units=16, embedding_size=20, attention=True, encoder_dropout=1., decoder_dropout=1.,
            iterations=100000, batch_size=128, display_step=1000, display_samples=3, verbose=True):
        """
        X_train: sequences of variable length, containing action IDs
        """
        # Split the training data into overlapping batches, based on a sliding window
        self.embedding_size = embedding_size  # Size of the vector used as a latent representation for each action (the representation is trained). similar to word2vec representation
        self.hidden_units = hidden_units
        self.attention = attention
        self.batch_size = batch_size
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        np.random.shuffle(X_train)
        batches, lengths = self._encode_sequences(X_train, self.batch_size, fit=True)

        self._build_network(True, self.hidden_units, self.embedding_size, self.attention, self.encoder_dropout, self.decoder_dropout)

        for i in range(iterations):
            batch_idx = i % len(batches)
            args = {
                self.encoder_inputs: batches[batch_idx],
                self.encoder_inputs_length: lengths[batch_idx],
                self.decoder_targets: batches[batch_idx]
            }
            loss = self.session.run(self.train_step, args)

            if i == 0 or (i+1) % display_step == 0 or i == iterations-1:
                predict_, l = self.session.run([self.train_decoder_prediction, self.loss], args)
                print('i={}: loss={}'.format(i+1, l))
                for inp, pred in zip(batches[batch_idx].T[:display_samples], predict_.T[:display_samples]):
                    print('{} => {} ({:.2f}%)'.format(inp, pred, LSTMAE.reconstruction_accurary(inp, pred)*100))

        return self


    def reconstruction_accurary(target, prediction):
        target = ''.join(str(x) for x in target[1:-1].tolist()) # Discard BOS and EOS
        prediction = ''.join(str(x) for x in prediction[1:-1].tolist())
        seq = difflib.SequenceMatcher(None, target, prediction)
        return seq.ratio()


    def predict(self, X):
        """
        X: sequences of variable length, containing action IDs
        out: (n_samples), return the reconstruction accuracy for each action sequence
        A low reconstruction accuracy denotes an anomaly
        """
        batches, lengths = self._encode_sequences(X, 1, fit=False)
        scores, predictions = [], []

        for batch, l in zip(batches, lengths):
            args = {
                self.encoder_inputs: batch,
                self.encoder_inputs_length: l,
            }
            predictions.append(self.session.run(self.infer_decoder_prediction, args))
            scores.append([LSTMAE.reconstruction_accurary(_in, _out) for _in, _out in zip(batch.T, predictions[-1].T)])

        return np.concatenate(scores), np.concatenate(predictions)


    def load(self):
        with open(LSTMAE.CONFIG_PATH, 'rb') as handle:
            (self.vocab_size, self.hidden_units, self.embedding_size, self.attention, self.encoder_dropout,
                self.decoder_dropout, self.data_encoder, self.data_decoder) = pickle.load(handle)

        self._build_network(False, self.hidden_units, self.embedding_size, self.attention)

        tf.train.Saver().restore(self.session, LSTMAE.MODEL_PATH)


    def save(self):
        tf.train.Saver().save(self.session, LSTMAE.MODEL_PATH)
        config = [self.vocab_size, self.hidden_units, self.embedding_size, self.attention,
                  self.encoder_dropout, self.decoder_dropout, self.data_encoder, self.data_decoder]
        with open(LSTMAE.CONFIG_PATH, 'wb') as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data = import_dataset('random')
    X_train, X_test = data[:int(data.shape[0]*TRAINING_PERCENT)], data[int(data.shape[0]*TRAINING_PERCENT):]
    model = LSTMAE()
    # model.fit(X_train, hidden_units=16, embedding_size=20, attention=True, encoder_dropout=1., decoder_dropout=1.,
    #           iterations=2000, batch_size=128, display_step=200, display_samples=3, verbose=True)
    # model.save()
    model.load()
    scores, X_pred = model.predict(X_test)
    print('Prediction accuracy: {}%'.format(np.mean(scores)))
    # for i, o, s in zip(X_test[:10], X_pred[:10], scores[:10]):
    #     print('{} => {} ({:.2f}%)'.format(i, o, s*100))
