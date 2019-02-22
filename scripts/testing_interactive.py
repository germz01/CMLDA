import nn
import numpy as np
import validation as val
import utils as u

from pprint import pprint
from tqdm import tqdm

X = np.concatenate((np.random.normal(2., 1., (50, 2)),
                    np.random.normal(5., 1., (50, 2))),
                   axis=0)
y = np.hstack((np.ones(50), np.zeros(50))).reshape(100, 1)

dataset = np.hstack((X, y))
np.random.shuffle(dataset)

split = int(X.shape[0]*0.75)

train = dataset[:split, :]
validation = dataset[split:, :]

X_train, y_train = np.hsplit(train, [X.shape[1]])
X_va, y_va = np.hsplit(validation, [X.shape[1]])

neural_net = nn.NeuralNetwork(X_train, y_train,
                              eta=0.1, alpha=0.1, epsilon=0.1,
                              batch_size=1, reg_lambda=0.01)

if raw_input('TESTING NEURAL NETWORK?[Y/N] ') == 'Y':
    neural_net.train(X_train, y_train, X_va, y_va)

    print 'STARTED WITH LOSS {}, ENDED WITH {}'.\
        format(neural_net.error_per_epochs[0], neural_net.error_per_epochs[-1])

    u.plot_learning_curve(neural_net,
                          fname='/Users/gianmarco/Desktop/learning_curve.pdf')

if raw_input('TESTING K-FOLD CROSS VALIDATION?[Y/N] ') == 'Y':
    cross_val = val.KFoldCrossValidation(X, y, neural_net)
    tqdm.write('AGGREGATED RESULTS: \n')
    pprint(cross_val.aggregated_results)

if raw_input('TESTING GRID SEARCH?[Y/N] ') == 'Y':
    param_ranges = dict()
    param_ranges['eta'] = (0.02, 2.0)
    param_ranges['alpha'] = 0.001
    param_ranges['batch_size'] = (1, 100)
    param_ranges['hidden_sizes'] = [(1, 100), (10, 20)]
    param_ranges['reg_lambda'] = (0.0, 0.1)
    param_ranges['reg_method'] = 'l2'
    param_ranges['epochs'] = 200

    grid_size = 10
    grid = val.HyperGrid(param_ranges, size=grid_size)

    selection = val.ModelSelectionCV(grid, repetitions=2)
    selection.search(X, y)
    results = selection.load_results()

    best_model = selection.select_best_model(X, y)
