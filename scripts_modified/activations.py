import numpy as np

from scipy.special import expit


def identity(x, fdev=False):
    if fdev:
        return np.ones(x.shape)
    return x


def sigmoid(x, fdev=False):
    if fdev:
        return expit(x) * (1. - expit(x))
    return expit(x)


def tanh(x, fdev=False):
    if fdev:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


def relu(x, fdev=False):
    if fdev:
        return np.where(x < 0, 0, 1)
    return np.where(x < 0, 0, x)


def softmax(x, fdev=False):
    if fdev:
        return np.diag(np.diag(x)) - np.dot(x, x.T)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

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
