from __future__ import division

import activations
import ipdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import optimizers

# CONSTANTS

# This is the path for the directory in which the images are saved.
IMGS = '../images/'


def compose_topology(X, hidden_sizes, y, task):
    """
    This functions builds up the neural network's topology as a list.

    Parameters
    ----------
    X: numpy.ndarray
        the design matrix

    hidden_sizes: list
        a list of integers; every integer represents the number
        of neurons that will compose an hidden layer of the
        neural network

    y: numpy.ndarray
        the target column vector

    task: str
        either 'classifier' or 'regression', the kind of task that the
        network has to pursue

    Returns
    -------
    A list of integers representing the neural network's topology.
    """
    if task == 'classifier':
        return [X.shape[1]] + list(hidden_sizes) + [y.shape[1]]

    return [X.shape[1]] + list(hidden_sizes) + [y.shape[1]]

# PLOTTING RELATED FUNCTIONS


def plot_learning_curve(optimizer, data, test_type, metric, params,
                        fname='../report/img/'):
    assert test_type in ['VALIDATION', 'TEST'] and \
        metric in ['MSE', 'MEE', 'ACCURACY', 'NORM']

    plt.plot(range(len(data[0])), data[0], alpha=0.65, label='TRAIN' if
             len(data) > 1 else None)

    if len(data) > 1:
        plt.plot(range(len(data[1])), data[1], alpha=0.65, label=test_type)
        plt.legend()

    plt.grid()
    plt.title('{} PER EPOCHS'.format(metric))
    plt.xlabel('EPOCHS')
    plt.ylabel(metric)

    plt.tight_layout()
    print params['beta_m']

    saving_str = '../report/img/SGD/' \
        if type(optimizer) is optimizers.SGD else \
        '../report/img/CGD/{}/'.format(params['beta_m'].upper())
    saving_str += 'sgd_' if type(optimizer) is optimizers.SGD else 'cgd_'
    saving_str += metric.lower() + '_' + test_type.lower() + '.pdf'

    if fname != '../report/img/':
        saving_str = saving_str.replace('.pdf', '_{}.pdf'.format(fname))

    plt.savefig(saving_str, bbox_inches='tight')
    plt.close()


def plot_learning_curve_with_info(optimizer, data, test_type, metric, params,
                                  fname='../report/img/'):
    assert test_type in ['VALIDATION', 'TEST'] and \
        metric in ['MSE', 'MEE', 'ACCURACY']

    plt.subplot(211)
    plt.plot(range(len(data[0])), data[0], alpha=0.65, label='TRAIN')
    plt.plot(range(len(data[1])), data[1], alpha=0.65, label=test_type)
    plt.grid()
    plt.title('{} PER EPOCHS'.format(metric))
    plt.xlabel('EPOCHS')
    plt.ylabel(metric)
    plt.legend()

    plt.subplot(212)
    plt.title('FINAL RESULTS AND PARAMETERS')
    plt.text(.25, .25, build_info_string(optimizer, data, test_type, metric,
             params), ha='left', va='center', fontsize=8)
    plt.axis('off')
    plt.tight_layout()

    saving_str = '../report/img/SGD/' \
        if type(optimizer) is optimizers.SGD else '../report/img/CGD/'
    saving_str += 'sgd_' if type(optimizer) is optimizers.SGD else 'cgd_'
    saving_str += metric.lower() + '_' + test_type.lower() + '.pdf'

    if fname != '../report/img/':
        saving_str = saving_str.replace('.pdf', '_{}.pdf'.format(fname))

    plt.savefig(saving_str, bbox_inches='tight')
    plt.close()


def build_info_string(optimizer, data, test_type, metric, params):
    assert test_type in ['VALIDATION', 'TEST'] and \
        metric in ['MSE', 'MEE', 'ACCURACY']

    special_char = {'alpha': r'$\alpha$', 'eta': r'$\eta$',
                    'reg_lambda': r'$\lambda$', 'beta_m': r'$\beta$',
                    'rho': r'$\rho$', 'reg_method': 'Regularization',
                    'momentum_type': 'Momentum',
                    'sigma_1': r'$\sigma_1$', 'sigma_2': r'$\sigma_2$',
                    'max_epochs': 'Max epoch', 'error_goal': 'Error goal'}
    act_list = []

    to_ret = 'OPTIMIZER:\n'
    to_ret += 'Stochastic gradient descent\n' \
        if type(optimizer) is optimizers.SGD else \
        'Conjugate gradient descent\n'

    to_ret += '\nFINAL VALUES:\n'
    to_ret += '{} TRAINING = {}\n{} = {}\n'.\
        format(metric, round(data[0][-1], 4), metric + ' ' + test_type,
               round(data[1][-1], 4))
    to_ret += '\nHYPERPARAMETERS:\n'

    for param in params:
        if param != 'topology' and param != 'activation' \
           and param != 'd_m':
            to_ret += special_char[param] + ' = {}'.format(params[param]) + \
                '\n'

    to_ret += '\nTOPOLOGY:\n'
    to_ret += str(params['topology']).replace('[', '').replace(']', '').\
        replace(', ', ' -> ')

    to_ret += '\n\nACTIVATIONS:\n'
    act_list.append('input')

    for act in params['activation']:
        if act is activations.sigmoid:
            act_list.append('sigmoid')
        elif act is activations.relu:
            act_list.append('relu')
        elif act is activations.tanh:
            act_list.append('tanh')
        else:
            act_list.append('identity')

    to_ret += ' -> '.join(act_list)

    return to_ret


def plot_betas_learning_curves(monk, betas, data, title, metric,
                               fname='../report/img/'):
    assert metric in ['MSE', 'MEE', 'ACCURACY']
    plt.subplot(211)

    for i in range(len(data[0])):
        plt.semilogy(range(len(data[0][i])), data[0][i], label=betas[i],
                     alpha=.65)
    plt.grid()
    plt.title(title + ' WITH STANDARD DIRECTION')
    plt.xlabel('EPOCHS')
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplot(212)

    for i in range(len(data[1])):
        plt.semilogy(range(len(data[1][i])), data[1][i], label=betas[i],
                     alpha=.65)

    plt.grid()
    plt.title(title + ' WITH MODIFIED DIRECTION')
    plt.xlabel('EPOCHS')
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(fname + str(monk) + '_' + metric.lower() + '_betas.pdf',
                bbox_inches='tight')
    plt.close()


def binarize_attribute(attribute, n_categories):
    """
    Binarize a vector of categorical values

    Parameters
    ----------
    attribute : numpy.ndarray or list
         numpy array with shape (p,1) or (p,) or list, containing
         categorical values.

    n_categories : int
        number of categories.
    Returns
    -------
    bin_att : numpy.ndarray
        binarized numpy array with shape (p, n_categories)
    """
    n_patterns = len(attribute)
    bin_att = np.zeros((n_patterns, n_categories), dtype=int)
    for p in range(n_patterns):
        bin_att[p, attribute[p]-1] = 1

    return bin_att


def binarize(X, categories_sizes):
    """
    Binarization of the dataset XWhat it does?

    Parameters
    ----------
    X : numpy.darray
        dataset of categorical values to be binarized.

    categories_sizes : list
        number of categories of each X column

    Returns
    -------
    out : numpy.darray
        Binarized dataset
    """

    atts = list()
    for col in range(X.shape[1]):
        atts.append(binarize_attribute(X[:, col], categories_sizes[col]))

    # h stack of the binarized attributes
    out = atts[0]
    for att in atts[1:]:
        out = np.hstack([out, att])

    return out
