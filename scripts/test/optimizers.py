from __future__ import division

import losses as lss
import numpy as np
import regularizers as reg


class Optimizer(object):

    """
    This class represents a wrapper for a generic optimizer implementing the
    back propagation algorithm as in Deep Learning, p. 205. It provides the
    building blocks for implementing other kinds of optimizers.

    Attributes
    ----------
    delta_W: list
        a list containing the network's weights' gradients

    delta_b: list
        a list containing the network's biases' gradients

    a: list
        a list containing the nets for each network's layer

    h: list
        a list containing the results of the activation functions' application
        to the nets
    """

    def __init__(self, nn):
        self.delta_W = [0 for i in range(nn.n_layers)]
        self.delta_b = [0 for i in range(nn.n_layers)]
        self.a = [0 for i in range(nn.n_layers)]
        self.h = [0 for i in range(nn.n_layers)]

    def forward_propagation(self, nn, x, y):
        for i in range(nn.n_layers):
            self.a[i] = nn.b[i] + (nn.W[i].dot(x.T if i == 0
                                               else self.h[i - 1]))
            self.h[i] = nn.activation[i](self.a[i])

        return lss.mean_squared_error(self.h[-1].T, y)

    def back_propagation(self, nn, x, y):
        g = lss.mean_squared_error(self.h[-1], y.T, gradient=True)

        for layer in reversed(range(nn.n_layers)):
            g = np.multiply(g, nn.activation[layer](self.a[layer], dev=True))
            # update bias, sum over patterns
            self.delta_b[layer] = g.sum(axis=1).reshape(-1, 1)

            # the dot product is summing over patterns
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0
                                        else x)
            # summing over previous layer units
            g = nn.W[layer].T.dot(g)


class SGD(Optimizer):

    """
    This class represents a wrapper for the stochastic gradient descent
    algorithm, as described in Deep Learing pag. 286.

    Attributes
    ----------
    """

    def __init__(self, nn, batch_size, eta=0.1,
                 momentum={'type': 'standard', 'alpha': 0.}, reg_lambda=0.0,
                 reg_method='l2'):
        super(SGD, self).__init__(nn)
        self.error_per_epochs = []
        self.error_per_batch = []
        self.batch_size = batch_size
        self.eta = eta

        assert momentum['type'] in ['standard', 'nesterov']
        self.momentum = momentum

        self.reg_lambda = reg_lambda
        self.reg_method = reg_method
        self.velocity_W = [0 for i in range(nn.n_layers)]
        self.velocity_b = [0 for i in range(nn.n_layers)]

    def optimize(self, nn, X, y, epochs):
        for e in range(epochs):
            error_per_batch = []

            dataset = np.hstack((X, y))
            np.random.shuffle(dataset)
            X, y = np.hsplit(dataset, [X.shape[1]])

            for b_start in np.arange(0, X.shape[0], self.batch_size):
                # BACK-PROPAGATION ALGORITHM ##################################

                x_batch = X[b_start:b_start + self.batch_size, :]
                y_batch = y[b_start:b_start + self.batch_size, :]

                # MOMENTUM CHECK ##############################################

                if self.momentum['type'] == 'nesterov':
                    for layer in range(nn.n_layers):
                        nn.W[layer] += self.momentum['alpha'] * \
                                       self.velocity_W[layer]

                error = super(SGD, self).forward_propagation(nn, x_batch,
                                                             y_batch)
                self.error_per_batch.append(error)
                error_per_batch.append(error)

                super(SGD, self).back_propagation(nn, x_batch, y_batch)

                # WEIGHTS' UPDATE #############################################

                for layer in range(nn.n_layers):
                    weight_decay = reg.regularization(nn.W[layer],
                                                      self.reg_lambda,
                                                      self.reg_method)

                    self.velocity_b[layer] = (self.momentum['alpha'] *
                                              self.velocity_b[layer]) \
                        - (self.eta / x_batch.shape[0]) * \
                        self.delta_b[layer]
                    nn.b[layer] += self.velocity_b[layer]

                    self.velocity_W[layer] = (self.momentum['alpha'] *
                                              self.velocity_W[layer]) \
                        - (self.eta / x_batch.shape[0]) * \
                        self.delta_W[layer]

                    nn.W[layer] += self.velocity_W[layer] - weight_decay

            self.error_per_epochs.append(np.sum(error_per_batch)/X.shape[0])


class CGD(Optimizer):

    """docstring for CGD"""
    def __init__(
                 self, arg):
        super(CGD, self).__init__()
        self.arg = arg

