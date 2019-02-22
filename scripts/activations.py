import numpy as np

from scipy.special import expit

# A_F = {'activation_name': {'f':f(), 'fdev':fdev(), 'range': (a,b) }}

A_F = {
    'identity':
    {
        'f': lambda x: x,
        'fdev': lambda x: np.ones(x.shape),
        'range': (np.NINF, np.Inf)
    },
    'sigmoid':
    {
        'f': lambda x: expit(x),
        'fdev': lambda x: expit(x) * (1. - expit(x)),
        'range': (0, 1)
    },
    'tanh':
    {
        'f': lambda x: np.tanh(x),
        'fdev': lambda x: 1 - np.tanh(x)**2,
        'range': (-1, -1)
    },
    'relu':
    {
        'f': lambda x: np.where(x < 0, 0, x),
        'fdev': lambda x: np.where(x < 0, 0, 1),
        'range': (0, np.Inf)
    }
}

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fpath = '../images/'

    x = np.arange(-5, 5, 0.01)
    # print A_F['relu']['f'](x)

    # computing for x each activation funct and its derivative
    y = dict()
    y_dev = dict()
    for f in A_F.keys():
        y[f] = A_F[f]['f'](x)
        y_dev[f] = A_F[f]['fdev'](x)

    if raw_input('PLOT ACTIVATION FUNCTIONS?[Y/N] ') == 'Y':
        # plot each activation function with its derivative
        for f in y.keys():
            print f
            plt.plot(x, y[f], label=f)
            plt.plot(x, y_dev[f], label=f)
            plt.grid()
            plt.title('activation: ' + f)
            plt.tight_layout()
            plt.legend()
            plt.savefig(fpath + 'activation_{}.pdf'.format(f))
            plt.close()

        # plot all activation functions
        for f in y.keys():
            plt.plot(x, y[f], label=f)

        plt.grid()
        plt.title('Activations')
        plt.tight_layout()
        plt.legend()
        plt.savefig(fpath + 'activations.pdf')
        plt.close()
