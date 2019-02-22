from __future__ import division

import activations as act
import losses as lss
import numpy as np
import regularizers as reg
import utils as u
import metrics

from tqdm import tqdm


class NeuralNetwork(object):
    """
    This class represents an implementation for a simple neural network.

    Attributes
    ----------
    X: numpy.ndarray
        the design matrix

    hidden_sizes: list
        a list of integers. The list's length represents the number of
        neural network's hidden layers and each integer represents the
        number of neurons in a hidden layer

    n_layers: int
        the network's number of layers

    topology: list
        the network's topology represented as a list

    eta: float
        the learning rate

    alpha: float
        the momentum constant

    epsilon: float
        the early stopping constant, given as a percentage %


    batch_size: int
        the batch size, 'batch' for batch mode

    batch_method: str
        the batch method used during the network's training phase, either
        'batch', 'on-line' or 'minibatch'

    early_stop: str
        the early stop method,
        to be chosen in (None, 'GL', 'PQ', 'testing').
        The 'testing' method computes the exit points without stopping

    early_stop_min_epochs: int
        the threshold for the minimum number of epochs for the network's
        training phase

    reg_lambda: float
        the regularization factor

    reg_method: str
        the regularization method, either l1 or l2 regularization are
        availables

    epochs: int
        the (maximum) number of epochs for which the neural network
        has to be trained

    activation: list
        the activation function to use for each layer, either
        'sigmoid', 'relu', 'tanh', 'identity'. len(hidden_sizes) + 1
        functions must be provided because also the output layer's
        activation function is requested

    w_par: int
        the parameter for initializing the network's weights matrices
        following the rule in Deep Learning, pag. 295

    w_method: str
        the method that has to be used for the weights' initialization,
        either 'DL' or 'uniform'

    W: list
        the weights' matrix for each one of the network's layer

    W_copy: list
        a copy of self.W used in case the network has to be resetted

    b: list
        the biases for each one of the network's layers

    b_copy: list
        a copy of self.b used in case the network has to be resetted

    params: dict
        the network's hyperparameters as a dictionary

    delta_W: list
        a list containing the deltas for the weights' matrices

    delta_b: list
        a list containing the deltas for the biases

    a: list
        a list containing the nets for each one of the network's layers

    h: list
        a list containing the output of the activation functions for each one
        of the network's layers

    task: str
        the network's task, either 'classifier' or 'regression'
    """
    def __init__(self, X, y, hidden_sizes=[10],
                 eta=0.5, alpha=0, epsilon=1,
                 early_stop=None, early_stop_min_epochs=50,
                 batch_size=1, epochs=1000,
                 reg_lambda=0.0, reg_method='l2',
                 w_par=6, w_method='DL',
                 activation='sigmoid',
                 task='classifier'):
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

        eta: float
            the learning rate
            (Default value = 0.5)

        alpha: float
            the momentum constant
            (Default value = 0)

        epsilon: float
            the early stopping constant, given as a percentage %
            (Default value = 1)

        early_stop: str
            the early stop method,
            to be chosen in (None, 'GL', 'PQ', 'testing').
            The 'testing' method computes the exit points without stopping
            (Default value = None)

        early_stop_min_epochs: int
            the threshold for the minimum number of epochs for the network's
            training phase
            (Default value = 50)

        batch_size: int
            the batch size, 'batch' for batch mode
            (Default value = 1)

        epochs: int
            the (maximum) number of epochs for which the neural network
            has to be trained
            (Default value = 1000)

        reg_lambda: float
            the regularization factor
            (Default value = 0.0)

        reg_method: str
            the regularization method, either l1 or l2 regularization are
            availables
            (Default value = 'l2')

        w_par: int
            the parameter for initializing the network's weights matrices
            following the rule in Deep Learning, pag. 295
            (Default value = 6)

        w_method: str
            the method that has to be used for the weights' initialization,
            either 'DL' or 'uniform'
            (Default value = 'DL')

        activation: list
            the activation function to use for each layer, either
            'sigmoid', 'relu', 'tanh', 'identity'. len(hidden_sizes) + 1
            functions must be provided because also the output layer's
            activation function is requested
            (Default value = ['sigmoid', 'sigmoid'])

        task: str
            the task that the neural network has to perform, either
            'classifier' or 'regression'
            (Default value = 'classifier')

        Returns
        -------
        """

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes) + 1
        self.topology = u.compose_topology(X, self.hidden_sizes, y)

        # self.X = X

        self.eta = float(eta)
        self.alpha = alpha
        self.epsilon = epsilon

        if batch_size == 'batch' or batch_size >= X.shape[0]:
            self.batch_size = X.shape[0]
            self.batch_method = 'batch'
        else:
            self.batch_size = batch_size
            if batch_size == 1:
                self.batch_method = 'on-line'
            else:
                self.batch_method = 'minibatch'

        assert early_stop in (None, 'GL', 'PQ', 'testing')
        self.early_stop = early_stop
        self.early_stop_min_epochs = early_stop_min_epochs

        assert reg_method in('l1', 'l2')
        self.reg_method = reg_method
        self.reg_lambda = reg_lambda

        self.epochs = epochs

        self.activation = self.set_activation(activation, task)

        self.w_par = float(w_par)
        self.w_method = w_method

        self.W = self.set_weights(w_par, w_method=self.w_method)
        self.W_copy = [w.copy() for w in self.W]
        self.b = self.set_bias()
        self.b_copy = [b.copy() for b in self.b]

        self.params = self.get_params()

        self.delta_W = [0 for i in range(self.n_layers)]
        self.delta_b = [0 for i in range(self.n_layers)]
        self.a = [0 for i in range(self.n_layers)]
        self.h = [0 for i in range(self.n_layers)]

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
        if type(activation) is list:
            assert len(activation) == self.n_layers

            return activation
        elif type(activation) is str:
            acts = [activation for l in range(self.n_layers)]

            if task == 'regression':
                acts[-1] = 'identity'

            return acts

    def set_weights(self, w_par=6, w_method='DL'):
        """
        This function initializes the network's weights matrices following
        the rule in Deep Learning, pag. 295

        Parameters
        ----------
        w_par: a parameter which is plugged into the formula for estimating
                the uniform interval for defining the network's weights
            (Default value = 6)

        w_method: str
            the method that has to be used for the weights' initialization,
            either 'DL' or 'uniform'
            (Default value = 'DL')

        Returns
        -------
        A list of weights matrices
        """
        W = []

        for i in range(1, len(self.topology)):

            if w_method == 'DL':
                low = - np.sqrt(w_par /
                                (self.topology[i - 1] + self.topology[i]))
                high = np.sqrt(w_par /
                               (self.topology[i - 1] + self.topology[i]))

                W.append(np.random.uniform(low, high,
                                           (self.topology[i],
                                            self.topology[i - 1])))
            elif w_method == 'uniform':
                low = -(w_par)
                high = w_par

                W.append(np.random.uniform(low, high,
                                           (self.topology[i],
                                            self.topology[i - 1])))

        return W

    def get_weights(self):
        """
        This function returns the list containing the network's weights'
        matrices

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(self.n_layers):
            print 'W{}: \n{}'.format(i, self.W[i])

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

    def get_bias(self):
        """
        This function returns the list containing the network's bias'
        matrices

        Parameters
        ----------

        Returns
        -------
        """
        for i in range(len(self.b)):
            print 'b{}: \n{}'.format(i, self.b[i])

    def get_params(self):
        """
        This function returns the network's parameters as a dictionary.

        Parameters
        ----------

        Returns
        -------
        A dictionary of parameters.
        """
        self.params = dict()
        self.params['eta'] = self.eta
        self.params['alpha'] = self.alpha
        self.params['batch_method'] = self.batch_method
        self.params['batch_size'] = self.batch_size
        self.params['hidden_sizes'] = self.hidden_sizes
        self.params['reg_method'] = self.reg_method
        self.params['reg_lambda'] = self.reg_lambda
        self.params['epochs'] = self.epochs
        self.params['activation'] = self.activation
        self.params['epsilon'] = self.epsilon
        self.params['w_par'] = self.w_par
        self.params['w_method'] = self.w_method
        self.params['topology'] = self.topology

        return self.params

    def forward_propagation(self, x, y):
        """
        This function implements the forward propagation algorithm following
        Deep Learning, pag. 205

        Parameters
        ----------
        x: numpy.ndarray
            a record, or batch, from the dataset

        y: numpy.ndarray
            the target array for the batch given in input


        Returns
        -------
        The error between the predicted output and the target one.
        """
        for i in range(self.n_layers):
            self.a[i] = self.b[i] + (self.W[i].dot(x.T if i == 0
                                                   else self.h[i - 1]))

            self.h[i] = act.A_F[self.activation[i]]['f'](self.a[i])

        return lss.mean_squared_error(self.h[-1].T, y)

    def back_propagation(self, x, y):
        """
        This function implements the back propagation algorithm following
        Deep Learning, pag. 206

        Parameters
        ----------
        x: numpy.ndarray
            a record, or batch, from the dataset

        y: numpy.ndarray
            the target array for the batch given in input

        Returns
        -------
        """
        g = lss.mean_squared_error(self.h[-1], y.T, gradient=True)

        for layer in reversed(range(self.n_layers)):
            g = np.multiply(
                g,
                act.A_F[self.activation[layer]]['fdev'](self.a[layer]))
            # update bias, sum over patterns
            self.delta_b[layer] = g.sum(axis=1).reshape(-1, 1)

            # the dot product is summing over patterns
            self.delta_W[layer] = g.dot(self.h[layer - 1].T if layer != 0
                                        else x)
            # summing over previous layer units
            g = self.W[layer].T.dot(g)

    def train(self, X, y, X_va=None, y_va=None):
        """
        This function implements the neural network's training routine.

        Parameters
        ----------
        X : numpy.ndarray
            the design matrix

        y : numpy.ndarray
            the target column vector

        X_va: numpy.ndarray
            the design matrix used for the validation
            (Default value = None)

        y_va: numpy.ndarray
            the target column vector used for the validation
            (Default value = None)

        Returns
        -------
        """
        velocity_W = [0 for i in range(self.n_layers)]
        velocity_b = [0 for i in range(self.n_layers)]

        self.error_per_epochs = []
        self.error_per_epochs_old = []
        self.error_per_batch = []
        self.mee_per_epochs = []
        if X_va is not None:
            self.error_per_epochs_va = []
            self.mee_per_epochs_va = []
        else:
            self.error_per_epochs_va = None
            self.mee_per_epochs_va = None

        if self.task == 'classifier':
            self.accuracy_per_epochs = []
            self.accuracy_per_epochs_va = []

        self.stop_GL = None
        self.stop_PQ = None
        stop_GL = False
        stop_PQ = False

        # for e in tqdm(range(self.epochs), desc='TRAINING'):
        for e in range(self.epochs):
            error_per_batch = []

            dataset = np.hstack((X, y))
            np.random.shuffle(dataset)
            X, y = np.hsplit(dataset, [X.shape[1]])

            for b_start in np.arange(0, X.shape[0], self.batch_size):
                # BACK-PROPAGATION ALGORITHM ##################################

                x_batch = X[b_start:b_start + self.batch_size, :]
                y_batch = y[b_start:b_start + self.batch_size, :]

                error = self.forward_propagation(x_batch, y_batch)
                self.error_per_batch.append(error)
                error_per_batch.append(error)

                self.back_propagation(x_batch, y_batch)

                # WEIGHTS' UPDATE #############################################

                for layer in range(self.n_layers):
                    weight_decay = reg.regularization(self.W[layer],
                                                      self.reg_lambda,
                                                      self.reg_method)

                    velocity_b[layer] = (self.alpha * velocity_b[layer]) \
                        - (self.eta / x_batch.shape[0]) * self.delta_b[layer]
                    self.b[layer] += velocity_b[layer]

                    velocity_W[layer] = (self.alpha * velocity_W[layer]) \
                        - (self.eta / x_batch.shape[0]) * self.delta_W[layer]

                    self.W[layer] += velocity_W[layer] - weight_decay

                ###############################################################

            # COMPUTING OVERALL MSE ###########################################

            self.error_per_epochs_old.append(
                np.sum(error_per_batch)/X.shape[0])

            y_pred = self.predict(X)
            self.error_per_epochs.append(metrics.mse(y, y_pred))
            self.mee_per_epochs.append(metrics.mee(y, y_pred))
            if X_va is not None:
                y_pred_va = self.predict(X_va)
                self.error_per_epochs_va.append(
                    metrics.mse(y_va, y_pred_va))
                self.mee_per_epochs_va.append(
                    metrics.mee(y_va, y_pred_va))

            if self.task == 'classifier':
                y_pred_bin = np.apply_along_axis(
                    lambda x: 0 if x < .5 else 1, 1, y_pred).reshape(-1, 1)

                y_pred_bin_va = np.apply_along_axis(
                    lambda x: 0 if x < .5 else 1, 1, y_pred_va).reshape(-1, 1)

                bin_assess = metrics.BinaryClassifierAssessment(
                    y, y_pred_bin, printing=False)
                bin_assess_va = metrics.BinaryClassifierAssessment(
                    y_va, y_pred_bin_va, printing=False)

                self.accuracy_per_epochs.append(bin_assess.accuracy)
                self.accuracy_per_epochs_va.append(bin_assess_va.accuracy)

            # CHECKING FOR EARLY STOPPING #####################################

            if self.early_stop is not None \
               and e > self.early_stop_min_epochs \
               and (e + 1) % 5 == 0:

                generalization_loss = 100 \
                    * ((self.error_per_epochs_va[e] /
                        min(self.error_per_epochs_va))
                       - 1)

                # GL method
                if generalization_loss > self.epsilon:
                    stop_GL = True

                # PQ method
                if self.early_stop != 'GL':  # PQ or 'testing'

                    min_e_per_strip = min(
                        self.error_per_epochs_va[e - 4:e + 1])
                    sum_per_strip = sum(self.error_per_epochs_va[e - 4:e + 1])
                    progress = 1000 * \
                               ((sum_per_strip / (5 * min_e_per_strip)) - 1)

                    progress_quotient = generalization_loss / progress

                    if progress_quotient > self.epsilon:
                        stop_PQ = True

                # stopping
                if stop_GL and self.stop_GL is None:
                    self.stop_GL = e
                if stop_PQ and self.stop_PQ is None:
                    self.stop_PQ = e

                if self.early_stop != 'testing' and (stop_GL or stop_PQ):
                    break

    def predict(self, x):
        """
        This function predicts the class/regression value for the vector
        given in input

        Parameters
        ----------
        x: np.ndarray
            the input to predict

        Returns
        -------
        A vector of predicted values
        """
        a_pred = [0 for i in range(self.n_layers)]
        h_pred = [0 for i in range(self.n_layers)]

        for layer in range(self.n_layers):

            a_pred[layer] = self.W[layer].dot(x.T if layer == 0 else
                                              h_pred[layer - 1])+self.b[layer]
            h_pred[layer] = act.A_F[self.activation[layer]]['f'](a_pred[layer])

        y_pred = h_pred[-1].T

        return y_pred
        # return lss.mean_squared_error(self.h[-1].T, y)

    def reset(self):
        """
        This function is used in order to reset the neural network inner
        variables. It is mainly used during the validation process.

        Parameters
        ----------

        Returns
        -------
        """
        self.W = [w.copy() for w in self.W_copy]
        self.b = [b.copy() for b in self.b_copy]
        self.delta_W = [0 for i in range(self.n_layers)]
        self.delta_b = [0 for i in range(self.n_layers)]
        self.a = [0 for i in range(self.n_layers)]
        self.h = [0 for i in range(self.n_layers)]
