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

                error = self.forward_propagation(nn, x_batch, y_batch)
                self.error_per_batch.append(error)
                error_per_batch.append(error)

                self.back_propagation(nn, x_batch, y_batch)

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

    """
    """

    def __init__(self, nn):
        """
        """

        super(CGD, self).__init__(nn)
        self.error = np.Inf
        self.error_prev = np.Inf
        self.error_per_epochs = []

    def optimize(self, nn, X, y, max_epochs, error_goal, beta_m,
                 plus=False, strong=False, sigma_1=1e-4, sigma_2=.5,
                 rho=0.0):
        """
        This function implements the optimization procedure following the
        Conjugate Gradient Descent, as described in the paper 'A new conjugate
        gradient algorithm for training neural networks based on a modified
        secant equation.'.

        Parameters
        ----------
        nn: nn.NeuralNetwork
            the neural network which has to be optimized

        X: numpy.ndarray
            the design matrix

        y: numpy.ndarray
            the target column vector

        max_epochs: int
            the maximum number of iterations for optimizing the network

        error_goal: float
            the stopping criteria based on a threshold for the maximum error
            allowed

        beta_m: str
            the method for computing the beta constant

        sigma_1, sigma_2: float
            the hyperparameters for the line search respecting the strong
            Armijo-Wolfe condition

        Returns
        -------
        """

        k = 0
        g_prev = 0

        while self.error >= error_goal and k != max_epochs:
            dataset = np.hstack((X, y))
            np.random.shuffle(dataset)
            X, y = np.hsplit(dataset, [X.shape[1]])

            self.error = self.forward_propagation(nn, X, y)
            self.back_propagation(nn, X, y)
            self.error_per_epochs.append(self.error)

            g = self.flat_weights(self.delta_W, self.delta_b)

            flatten_weights = self.flat_weights(nn.W, nn.b)
            flatted_copies = self.flat_weights(nn.W_copy, nn.b_copy)

            if k == 0:
                self.error_prev = self.error
                g_prev = 0
                if beta_m == 'hs' or 'mhs':
                    d_prev = -g
                    if beta_m == 'mhs':
                        w_prev = 0

            if beta_m == 'fr' or beta_m == 'pr':
                beta = self.get_beta(g, g_prev, beta_m, plus=plus)
            elif beta_m == 'hs':
                beta = self.get_beta(g, g_prev, beta_m, plus=plus,
                                     d_prev=d_prev)
            else:
                beta = self.get_beta(g, g_prev, beta_m, plus=plus,
                                     d_prev=d_prev, error=self.error,
                                     error_prev=self.error_prev,
                                     w=flatten_weights, w_prev=w_prev,
                                     rho=rho)

            d = self.get_direction(k, g, beta)

            eta = self.awls(flatten_weights, g, d, sigma_1, sigma_2, nn, X, y,
                            self.error, strong)

            new_W = flatted_copies + (eta * d)
            nn.W, nn.b = self.unflat_weights(new_W, nn.n_layers, nn.topology)

            nn.W_copy = [w.copy() for w in nn.W]
            nn.b_copy = [b.copy() for b in nn.b]

            k += 1
            g_prev = g

            if beta_m == 'hs' or 'mhs':
                d_prev = d
                if beta_m == 'mhs':
                    w_prev = flatted_copies

    def flat_weights(self, W, b):
        """
        This function flattens the network's biases and weights matrices into
        a single column vector, after concateneting each weights matrix with
        the corresponding bias column vector.

        Parameters
        ----------
        W: list
            a list of weights' matrices

        b: list
            a list of biases column vectors

        Returns
        -------
        A column vector.
        """

        to_return = [np.hstack((b[l], W[l])).flatten() for l in range(len(W))]

        return np.concatenate(to_return).reshape(-1, 1)

    def unflat_weights(self, W, n_layer, topology):
        """
        This functions return the column vector from the optimizer reshaped
        as the original list of weights' matrices.

        Parameters
        ----------
        W: numpy.ndarray
            the column vector containing the network's weights

        n_layer: int
            a number representing the number of network's layer

        topology: list
            a list containing the number of neurons for each network's layer

        Returns
        -------
        A list of weights' matrices.
        """

        to_ret_W, to_ret_b = [], []
        # sommiamo topology[1] perche consideriamo il BIAS
        ws = 0

        for layer in range(1, n_layer + 1):
            to_app_W, to_app_b = [], []

            for i in range(topology[layer]):
                bias_position = (i * topology[layer - 1] + i) if layer == 1 \
                    else ws + (i * topology[layer - 1] + i)
                to_app_b.append(W[bias_position, :])
                to_app_W.append(W[bias_position + 1:bias_position + 1 +
                                topology[layer - 1]])
            to_app_W = np.vstack(to_app_W)
            to_app_b = np.vstack(to_app_b)

            to_ret_W.append(to_app_W.reshape(topology[layer],
                                             topology[layer - 1]))
            to_ret_b.append(to_app_b.reshape(topology[layer], 1))

            ws += (topology[layer - 1] * topology[layer]) + topology[layer]

        return to_ret_W, to_ret_b

    def get_direction(self, k, g, beta, d_prev=0, method='standard'):
        """
        This functions is used in order to get the current descent
        direction.

        Parameters
        ----------
        k: int
            the current epoch

        g: numpy.ndarray
            the vector which contains the gradient for every weight
            in the network

        beta: float
            the beta constant for the current epoch

        d_prev: numpy.ndarray
            the previous epoch's direction
            (Default value = 0)

        method: str
            the choosen method for the direction's calculus as
            suggested in 'A new conjugate gradient algorithm for
            training neural networks based on a modified secant
            equation', either 'standard' or 'plus'
            (Default value = 'standard')

        Returns
        -------
        The gradient descent for epoch k.
        """

        if k == 0:
            return -g

        if method == 'standard':
            return (-g + (beta * d_prev))

        return (-(1 + beta * ((g.T.dot(d_prev)) / np.linalg.norm(g))) * g) \
            + (beta * d_prev)

    def get_beta(self, g, g_prev, method, plus=False, **kwargs):
        """
        This function implements various types of beta.

        Parameters
        ----------
        g: numpy.ndarray
            the vector which contains the gradient for every weight
            in the network

        g_prev: numpy.ndarray
            the vector which contains the gradient for every weight
            in the network for the previous algorithm's iteration

        method: str
            the formula used to compute the beta, either 'hs' or
            'pr' or 'fr' or 'mhs'

        plus: bool
            whether or not to use the modified HS formula
            (Default value = False)

        kwargs: dict
            a dictionary containing the parameters for various betas'
            initialization formulas; the parameters can be the following

            d_prev: numpy.ndarray
                the previous epoch's direction
                (Default value = None)

            error: float
                the error for the current epoch
                (Default value = None)

            error_prev: float
                the error for previous epoch
                (Default value = None)

            w: numpy.ndarray
                the weights' column vector for current epoch
                (Default value = None)

            w_prev: numpy.ndarray
                the weights' column vector for the previous epoch
                (Default value = None)

            rho: float
                an hyperparameter between 0 and 1
                (Default value = None)

        Returns
        -------
        The beta computed with the specified formula.
        """

        assert method in ['hs', 'pr', 'fr', 'mhs']

        beta = 0.0

        if method == 'hs':
            assert 'd_prev' in kwargs
            beta = (g.T.dot(g - g_prev)) / \
                ((g - g_prev).T.dot(kwargs['d_prev']))
        elif method == 'pr':
            beta = (g.T.dot(g-g_prev)) / (np.linalg.norm(g_prev))**2
        elif method == 'fr':
            beta = (np.linalg.norm(g))**2 / (np.linalg.norm(g_prev))**2
        else:
            assert 'error' in kwargs and 'error_prev' in kwargs and \
                'rho' in kwargs and plus is True

            s = kwargs['w'] - kwargs['w_prev']
            teta = (2 * (kwargs['error_prev'] - kwargs['error'])) + \
                (g + g_prev).T.dot(s)
            y_tilde = (g - g_prev) + ((kwargs['rho'] *
                                      ((max(teta, 0)) / s.T.dot(s)))
                                      * s)
            beta = (g.T.dot(y_tilde)) / (kwargs['d_prev'].T.dot(y_tilde))

        return max(beta, 0) if plus else beta

    def awls(self, W, g, d, sigma_1, sigma_2, nn, x, y, error, strong):
        """
        This function implements the searching for an optimal learning rate
        respecting the strong Armijo-Wolfe conditions.

        Parameters
        ----------
        W: numpy.ndarray
            a column vector representing the network's weigths

        g: numpy.ndarray
            a column vector representing the gradients for the network

        d: numpy.ndarray
            a column vector representing the current descent direction

        sigma_1, sigma_2: float
            hyperparameters involved in the Armijo-Wolfe condition

        nn: nn.NeuralNetwork
            the neural network

        x: np.ndarray
            the design set

        y: np.ndarray
            a column vector representing the target values for the design set

        error: float
            the error for the current epoch computed with the unmodified
            weights

        Returns
        -------
        The optimal learning rate.
        """

        # TODO: AGGIUNGERE NP.LINSPACE CON VALORI MIGLIORI
        alphas = np.linspace(0.0, 1., 50, endpoint=False)[1:]
        g_d = g.T.dot(d)

        for alpha in alphas:
            W_search = W.copy()

            new_W, new_b = self.unflat_weights(W_search + (alpha * d),
                                               nn.n_layers, nn.topology)
            nn.W = new_W
            nn.b = new_b

            self.error_prev = self.forward_propagation(nn, x, y)

            if self.error_prev - error <= sigma_1 * alpha * g_d:
                self.back_propagation(nn, x, y)
                new_g = self.flat_weights(self.delta_W, self.delta_b)

                if strong:
                    if np.absolute(new_g.T.dot(d)) <= \
                       sigma_2 * np.absolute(g_d):
                        return alpha
                else:
                    if new_g.T.dot(d) <= sigma_2 * g_d:
                        return alpha
