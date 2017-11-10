import tensorflow as tf
import numpy as np
import time
from sklearn import metrics

def get_random_block_from_data(data, batch_size):
    if batch_size == len(data):
        return data
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

class VariationalAutoencoder(object):

    def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
        tf.reset_default_graph()
        self.n_input = n_input
        self.n_hidden = n_hidden

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

        # sample from gaussian distribution
        eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

        # cost
        reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + self.latent_loss)
        self.detailed_cost = reconstr_loss + self.latent_loss
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_input, self.n_hidden],
            initializer=tf.contrib.layers.xavier_initializer())
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def fit(self, X, training_epochs=20, batch_size=128, display_step=1, monitoring=-1,
            X_test=None, Y_test=None, outlier_metric='distance'):
        monitoring_data = [[], [], []]  # AP, NLL, time
        total_train_time, prev_monitoring_time, train_time = 0, 0, 0
        first_monitor = True
        current_milli_time = lambda: int(round(time.time() * 1000))
        start_train_time = current_milli_time()
        if monitoring > 0:
            labels = [0 if lb == 0 else 1 for lb in Y_test]

        n_samples = X.shape[0]
        avg_cost = []
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(training_epochs):
            batch_xs = get_random_block_from_data(X, batch_size)

            # Fit training using batch data
            cost = self.partial_fit(batch_xs)
            # Compute average loss
            avg_cost.append(cost)

            total_train_time = current_milli_time() - start_train_time
            if monitoring > 0 and (total_train_time - prev_monitoring_time >= monitoring*1000 or first_monitor):
                train_time += total_train_time - prev_monitoring_time
                scores = np.nan_to_num(self.outlier_score(X_test, outlier_metric))
                ap = metrics.average_precision_score(labels, scores, average='micro')
                nll = self.calc_latent_loss(X_test)
                prev_monitoring_time = current_milli_time() - start_train_time

                monitoring_data[0].append(ap)
                monitoring_data[1].append(np.mean(nll))
                monitoring_data[2].append(train_time/1000.)
                first_monitor = False

            # Display logs per epoch step
            if display_step is not None and i % display_step == 0:
                mean_cost = np.mean(avg_cost)
                print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(mean_cost))
                if np.isnan(mean_cost) or mean_cost > 1e13:
                    return monitoring_data

        return monitoring_data

    def save_model(self):
        self.saver.save(self.sess, 'tmp/vae')

    def restore_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./tmp'))
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)

    def partial_fit(self, X):
        self.save_model()
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        if np.isnan(cost):
            self.restore_model()
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def calc_detailed_cost(self, X):
        return self.sess.run(self.detailed_cost, feed_dict = {self.x: X})

    def calc_latent_loss(self, X):
        return self.sess.run(self.latent_loss, feed_dict = {self.x: X})

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def reconstruction_error(self, X):
        error = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0), 1)
        return self.sess.run(error, feed_dict={self.x: X})

    def outlier_score(self, X, metric='distance'):
        if metric == 'distance':
            return self.reconstruction_error(X)
        elif metric == 'cost':
            return self.calc_detailed_cost(X)
        elif metric == 'latent_loss':
            return self.calc_latent_loss(X)
        elif metric == 'norm':
            return np.linalg.norm(self.reconstruct(X) - X, axis=1)
        else:
            raise ValueError('Invalid metric: {}'.format(metric))

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


class VAE(Detector):
    """ Variational Autoencoder """
    def __init__(self):
        self.training_epochs = 4000
        self.display_step = 100
        self.monitoring = False
        self.outlier_metric = 'norm'  # 'distance', 'norm', 'cost', 'latent_loss'

    def fit(self, X):
        batch_size = min(X.shape[0], 1000)
        self.clf = VariationalAutoencoder(n_input = X.shape[1], n_hidden = 50,
                                          optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))
        return self.clf.fit(X, self.training_epochs, batch_size, self.display_step, monitoring=self.monitoring,
                     X_test=None, Y_test=None, outlier_metric=self.outlier_metric)

    def score_samples(self, X):
        """ An outlier is a data point with a low score """
        return -self.clf.outlier_score(X, metric=self.outlier_metric)
