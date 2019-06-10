from __future__ import division

import activations as act
import numpy as np
import optimizers as opt
import utils as u


class NeuralNetwork(object):

    """
    This class implements an Artificial Neural Network.

    Attributes
    ----------

    n_layers: int
        the network's number of layers

    topology: list
        the network's topology represented as a list, each list's element
        represents how many units there are in the corresponding layer

    activation: list or str
        if list represents the activation functions that have to setted
        for every network's layer else represents the single activation
        functions that has to be setted for every network's layer

    W: list
        the weights' matrix for each one of the network's layer

    W_copy: list
        a copy of self.W used in case the network has to be resetted

    b: list
        the biases for each one of the network's layers

    b_copy: list
        a copy of self.b used in case the network has to be resetted

    task: str
        the network's task, either 'classifier' or 'regression'

    optimizer: opt.SGD or opt.CGD
        the network's optimizer
    """

    def __init__(self, X, y, hidden_sizes=[10], initialization='glorot',
                 activation='sigmoid', task='classifier'):
        """
        The class' constructor.

        Parameters
        ----------
        X: numpy.ndarray
            the design matrix

        y: numpy.ndarray
            the target column vector

        hidden_sizes: list
            a list of integers. The list's length represents the number of
            neural network's hidden layers and each integer represents the
            number of neurons in a hidden layer
            (Default value = [10])

        initialization: str or dict
            the method that has to used for initializing the network's weights.
            (Default value = 'glorot')

        activation: list or str
            if list represents the activation functions that have to setted
            for every network's layer else represents the single activation
            functions that has to be setted for every network's layer
            (Default value = 'sigmoid')

        task: str
            the task that the neural network has to perform, either
            'classifier' or 'regression'
            (Default value = 'classifier')

        Returns
        -------
        """

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
        """
        This function initializes the list containing the activation functions
        for every network's layer.

        Parameters
        ----------
        activation: list or str
            if list represents the activation functions that have to setted
            for every network's layer else represents the single activation
            functions that has to be setted for every network's layer

        task: str
            the task the network has to pursue, either 'classifier' or
            'regression'

        Returns
        -------
        A list of activation functions
        """

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
        This function initializes the network's weights matrices following
        the rule in Deep Learning, pag. 295

        Parameters
        ----------
        initialization: str or dict
            the method that has to used for initializing the network's weights.
            (Default value = 'glorot')

        Returns
        -------
        A list of weights matrices
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
        This function initializes the network's biases.

        Parameters
        ----------

        Returns
        -------
        A list of biases.
        """
        b = []

        for i in range(1, len(self.topology)):
            b.append(np.zeros((self.topology[i], 1)))

        return b

    def restore_weights(self):
        """
        This functions restores the network's weights and biases to their
        original values.

        Parameters
        ----------

        Returns
        -------
        """

        self.W = [w.copy() for w in self.W_copy]
        self.b = [b.copy() for b in self.b_copy]

    def update_weights(self, W, b):
        """
        This function is used to update the network's weights and biases.

        Parameters
        ----------

        Returns
        -------
        """

        assert len(W) == len(self.W) and len(b) == len(self.b)

        self.W = W
        self.b = b

    def update_copies(self, W=None, bias=None):
        """
        This function is used to update the network's copies of the weights
        and biases.

        Parameters
        ----------

        Returns
        -------
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
        """
        This function implements the neural network's training routine.

        Parameters
        ----------
        X : numpy.ndarray
            the design matrix

        y : numpy.ndarray
            the target column vector

        optimizer: str
            the type of optimizer, either SGD or CGD, that has to be used
            during the training

        epochs: int
            the training routine's number of epochs
            (Default value = 1000)

        X_va: numpy.ndarray
            the design matrix used for the validation
            (Default value = None)

        y_va: numpy.ndarray
            the target column vector used for the validation
            (Default value = None)

        kwargs: dict
            additional parameters for the optimizers' initialization

        Returns
        -------
        """
        assert optimizer in ['SGD', 'CGD']

        if optimizer == 'SGD':
            self.optimizer = opt.SGD(self, **kwargs)
            self.optimizer.optimize(self, X, y, X_va, y_va, epochs)
        else:
            self.optimizer = opt.CGD(self, **kwargs)
            self.optimizer.optimize(self, X, y, X_va, y_va, **kwargs)
