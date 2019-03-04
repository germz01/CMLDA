import matplotlib.pyplot as plt
import numpy as np

from scipy.special import expit


def identity(x, dev=False):
    """
    This function implements the identity function.

    Parameters
    ----------
    x: numpy.ndarray
        the input vector

    dev: bool
        whether or not to compute the function's derivative
        (Default value = False)

    Returns
    -------
    The function's output, or its derivative.
    """
    if dev:
        return np.ones(x.shape)

    return x


def sigmoid(x, dev=False):
    """
    This function implements the sigmoid function.

    Parameters
    ----------
    x: numpy.ndarray
        the input vector

    dev: bool
        whether or not to compute the function's derivative
        (Default value = False)

    Returns
    -------
    The function's output, or its derivative.
    """
    if dev:
        return expit(x) * (1. - expit(x))

    return expit(x)


def tanh(x, dev=False):
    """
    This function implements the hyperbolic tangent function.

    Parameters
    ----------
    x: numpy.ndarray
        the input vector

    dev: bool
        whether or not to compute the function's derivative
        (Default value = False)

    Returns
    -------
    The function's output, or its derivative.
    """
    if dev:
        return 1 - np.tanh(x)**2

    return np.tanh(x)


def relu(x, dev=False):
    """
    This function implements the rectified linear function.

    Parameters
    ----------
    x: numpy.ndarray
        the input vector

    dev: bool
        whether or not to compute the function's derivative
        (Default value = False)

    Returns
    -------
    The function's output, or its derivative.
    """
    if dev:
        return np.where(x < 0, 0, 1)

    return np.where(x < 0, 0, x)


def softmax(x, dev=False):
    """
    This function implements the softmax function.

    Parameters
    ----------
    x: numpy.ndarray
        the input vector

    dev: bool
        whether or not to compute the function's derivative
        (Default value = False)

    Returns
    -------
    The function's output, or its derivative.
    """
    if dev:
        return np.diagflat(softmax(x)) - np.dot(softmax(x), softmax(x).T)

    return np.exp(x)/np.sum(np.exp(x))


def print_fun(fun, x, key):
    print keys[key]
    plt.plot(x, fun(x, False), label=keys[key])
    plt.plot(x, fun(x, True), label=keys[key]+'_derivative')
    plt.grid()
    plt.title('activation: ' + keys[key])
    plt.tight_layout()
    plt.legend()
    plt.savefig(fpath + 'activation_{}.pdf'.format(keys[key]))
    plt.close()


# This dictionary allows an easier access to the activation functions
# during the network's creation. It permits to call an activation function
# simply via functions['function_name'](function_input[, dev_yes_or_no]).

functions = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}


if __name__ == "__main__":
    fpath = '../images/'

    x = np.arange(-5, 5, 0.01)
    keys = ['identity', 'sigmoid', 'tanh', 'relu', 'softmax']

    if raw_input('PLOT ACTIVATION FUNCTIONS?[Y/N] ') == 'Y':
        print_fun(identity, x, 0)
        print_fun(sigmoid, x, 1)
        print_fun(tanh, x, 2)
        print_fun(relu, x, 3)
        #print_fun(softmax, x, 4)
        plt.grid()
        plt.title('Activations')
        plt.tight_layout()
        plt.legend()
        plt.savefig(fpath + 'activations.pdf')
        plt.close()
