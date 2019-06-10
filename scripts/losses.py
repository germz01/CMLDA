from __future__ import division

import numpy as np


def mean_squared_error(y_hat, y, gradient=False):
    """
    This functions computes the mean squared error, or its gradient, between
    a target vector and a vector of network's predictions.

    Parameters:
        y_hat: numpy.ndarray
            the network's prediction

        y: numpy.ndarray
            the target vector

        gradient: bool
            whether or not to compute the error's gradient
            (Default value = False)

    Returns
    -------
    A float representing the mean square error or its gradient.
    """
    if gradient:
        return y_hat - y
    else:
        return 0.5 * (np.sum(np.square(y_hat - y)) / y.shape[0])


def mean_euclidean_error(y_hat, y, gradient=False):
    """
    This functions computes the mean euclidean error, or its gradient, between
    a target vector and a vector of network's predictions.

    Parameters:
        y_hat: numpy.ndarray
            the network's prediction

        y: numpy.ndarray
            the target vector

        gradient: bool
            whether or not to compute the error's gradient
            (Default value = False)

    Returns
    -------
    A float representing the mean square error or its gradient.
    """
    if gradient:
        return ((y_hat - y) / np.linalg.norm(y_hat - y))
    else:
        return np.linalg.norm(y_hat - y) / y.shape[0]


# THE FOLLOWING FUNCTIONS ARE FOR TESTING ONLY


def mee(d, y):
    """mean euclidean error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.mean(np.sqrt(np.einsum('pk->p', (d-y)**2)))


def mee_dev(d, y):
    """std. deviation for the euclidean error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.std(np.sqrt(np.einsum('pk->p', (d-y)**2)))


def rmse(d, y):
    """root mean square error

    Parameters
    ----------
    d :

    y :


    Returns
    -------

    """
    p = d.shape[0]
    if len(d.shape) == 1:
        d = d.reshape((p, 1))
    if len(y.shape) == 1:
        y = y.reshape((p, 1))

        return np.sqrt(np.einsum('pk->', (d-y)**2) / p)
