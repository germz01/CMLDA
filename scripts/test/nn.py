from __future__ import division

import activations as act
import numpy as np
import optimizers as opt
import utils as u


class NeuralNetwork(object):

    """
    """

    def __init__(self, X, y, hidden_sizes=[10], initialization='glorot',
                 activation='sigmoid', task='classifier'):
        self.n_layers = len(hidden_sizes) + 1
        self.topology = u.compose_topology(X, hidden_sizes, y, task)

        self.activation = self.set_activation(activation, task)

        self.W = self.set_weights(initialization)
        self.W_copy = [w.copy() for w in self.W]
        self.b = self.set_bias()
        self.b_copy = [b.copy() for b in self.b]

        assert task in ('classifier', 'regression')
        self.task = task

    def set_activation(self, activation, task):
        to_return = list()

        if type(activation) is list:
            assert len(activation) == self.n_layers

            [to_return.append(act.functions[funct]) for funct in activation]
        elif type(activation) is str:
            assert activation in ['identity', 'sigmoid', 'tanh', 'relu',
                                  'softmax']

            [to_return.append(act.functions[activation])
             for l in range(self.n_layers)]

        if task == 'regression':
            to_return[-1] = act.functions['identity']

        return to_return

    def set_weights(self, initialization):
        """
        """
        assert type(initialization) is str or type(initialization) is dict

        W = []

        for i in range(1, len(self.topology)):

            if type(initialization) is str:
                low = - np.sqrt(6 /
                                (self.topology[i - 1] + self.topology[i]))
                high = np.sqrt(6 /
                               (self.topology[i - 1] + self.topology[i]))

                W.append(np.random.uniform(low, high,
                                           (self.topology[i],
                                            self.topology[i - 1])))
            elif type(initialization) is dict:
                low = dict['low']
                high = dict['high']

                W.append(np.random.uniform(low, high, (self.topology[i],
                                                       self.topology[i - 1])))

        return W

    def set_bias(self):
        """
        """
        b = []

        for i in range(1, len(self.topology)):
            b.append(np.zeros((self.topology[i], 1)))

        return b

    def restore_weights(self):
        """
        """

        self.W = [w.copy() for w in self.W_copy]
        self.b = [b.copy() for b in self.b_copy]

    def update_weights(self, W, b):
        assert len(W) == len(self.W) and len(b) == len(self.b)

        self.W = W
        self.b = b

    def update_copies(self, W=None, bias=None):
        """
        """

        if W is None and bias is None:
            self.W_copy = [w.copy() for w in self.W]
            self.b_copy = [b.copy() for b in self.b]
        else:
            assert len(W) == len(self.W_copy) and len(bias) == len(self.b_copy)

            self.W_copy = [w.copy() for w in W]
            self.b_copy = [b.copy() for b in bias]

    def train(self, X, y, optimizer, epochs=1000, X_va=None, y_va=None,
              **kwargs):
        assert optimizer in ['SGD', 'CGD']

        if optimizer == 'SGD':
            self.optimizer = opt.SGD(self, **kwargs)
            self.optimizer.optimize(self, X, y, X_va, y_va, epochs)
        else:
            self.optimizer = opt.CGD(self, **kwargs)
            self.optimizer.optimize(self, X, y, X_va, y_va, **kwargs)
