import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import edward as ed
from edward.models import Normal, OneHotCategorical

class BNN(object):
    """ http://rpubs.com/arowan/bayesian_deep_learning """
    # TODO: batches https://alpha-i.co/blog/MNIST-for-ML-beginners-The-Bayesian-Way.html
    def __init__(self, n_samples, d_in, d_out, hidden_units, hidden_layers):
        self.dims = [(d_in, hidden_units)]
        for _ in range(hidden_units):
            self.dims.append((hidden_units, hidden_units))
        self.dims.append((hidden_units, d_out))

        with tf.name_scope('model'):
            self.weights = []
            for _in, _out in self.dims:
                W_i = Normal(loc=tf.zeros([_in, _out], dtype=tf.float32), scale=tf.ones([_in, _out], dtype=tf.float32))
                b_i = Normal(loc=tf.zeros(_out, dtype=tf.float32), scale=tf.ones(_out, dtype=tf.float32))
                self.weights.append([W_i, b_i])

            self.X = tf.placeholder(tf.float32, [n_samples, d_in], name='X')
            # self.Y = Normal(loc=self._neural_network(self.X), scale=0.1 * tf.ones(n_samples, dtype=tf.float32), name='Y')
            self.Y = OneHotCategorical(logits=self._neural_network(self.X), name='Y')

    def _neural_network(self, x):
        h = x
        for i, (W_i, b_i) in enumerate(self.weights):
            h = tf.matmul(h, W_i) + b_i
            if i != len(self.weights)-1:
                h = tf.tanh(h)
        return h #tf.reshape(h,)

    def _inference(self, X_train, Y_train, iterations):
        with tf.name_scope('posterior'):
            self.session = ed.get_session()
            self.posteriors = []
            for i, (_in, _out) in enumerate(self.dims):
                with tf.name_scope('qW_' + str(i)):
                    qW_i = Normal(loc=tf.Variable(tf.random_normal([_in, _out]), name='loc'),
                                  scale=tf.nn.softplus(tf.Variable(tf.random_normal([_in, _out]), name='scale')))
                with tf.name_scope('qb_' + str(i)):
                    qb_i = Normal(loc=tf.Variable(tf.random_normal([_out]), name='loc'),
                              scale=tf.nn.softplus(tf.Variable(tf.random_normal([_out]), name='scale')))
                self.posteriors.append((qW_i, qb_i))

            inference_dict = {}
            for (W_i, b_i), (qW_i, qb_i) in zip(self.weights, self.posteriors):
                inference_dict[W_i] = qW_i
                inference_dict[b_i] = qb_i

            Y_train = Y_train[np.newaxis].T
            self.oneHotEncoder = OneHotEncoder(sparse=False).fit(Y_train)
            Y_train = self.oneHotEncoder.transform(Y_train)

            self.inference = ed.KLqp(inference_dict, data={self.X: X_train, self.Y: Y_train})

            self.inference.run(n_iter=iterations)
            print(self.session.run(self.inference.loss, feed_dict={self.X: X_train, self.Y: Y_train}))

    def fit(self, X_train, Y_train, iterations=1000):
        self._inference(X_train, Y_train, iterations)

    def predict(self, X):
        # TODO: MC Dropout: apply dropout at test time to obtain mean results and uncertainty over the predictions
        # This allows to represent model uncertainty in deep learning, using dropout as a Bayesian approximation
        # return self.session.run(self.inference.loss, feed_dict={X: X_test, Y: y_test})
        # return self.Y.eval(feed_dict={self.X: X})
        print(self._neural_network(X))
        # return ed.evaluate('Y', data={self.X: X})


def test_multivariate_regression():
    D = 1
    N = 100
    noise_std=0.1
    X = np.concatenate([np.linspace(0, 2, num=N / 2),
                        np.linspace(6, 8, num=N / 2)])
    Y = np.cos(X) + np.random.normal(0, noise_std, size=N)
    X = (X - 4.0) / 4.0
    X = X.reshape((N, D))
    model = BNN(N, D, D, 10, 1)
    model.fit(X, Y, iterations=1000)

def test_classification():
    from sklearn import datasets, metrics
    X, Y = datasets.load_iris(return_X_y=True)
    N, d_in, d_out = X.shape[0], X.shape[1], len(set(Y))
    model = BNN(N, d_in, d_out, 30, 1)
    model.fit(X, Y, iterations=300)
    Y_pred = model.predict(X)
    Y_pred = [np.argmax(pred) for pred in Y_pred] # One-hot decoding
    print(Y)
    print('\n')
    print(Y_pred)
    print(metrics.accuracy_score(Y, Y_pred))

if __name__ == "__main__":
    # test_multivariate_regression()
    test_classification()
