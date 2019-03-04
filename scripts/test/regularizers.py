import numpy as np


def regularization(w, lamb, l_n):
    """
    This functions implements the L1/L2 regularization for the weights' decay
    as described in Deep Learning, pag. 224 and 228.

    Parameters
    ----------
    w : the weights' matrix

    lamb : the regularization constant

    l_n: the type of regularization to apply, either L1 or L2


    Returns
    -------
    The regularization factor
    """
    if l_n == 'l1':
        return lamb * np.sign(w)
    else:
        return lamb * w
