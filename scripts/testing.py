import nn as NN
import numpy as np
import matplotlib.pyplot as plt
import imp
import utils as u
import pandas as pd
from pprint import pprint
import holdout as holdout
import validation as val
import time


# number of patterns for each class
p_class1 = 70
p_class2 = 30
# attributes/features
n = 10

X = np.vstack((np.random.normal(2., 1., (p_class1, n)),
               np.random.normal(6., 1., (p_class2, n))))

y = np.vstack((np.hstack((np.ones(p_class1), np.zeros(p_class2))),
               np.hstack((np.zeros(p_class1), np.ones(p_class2))))).T
# y = np.hstack((np.ones(p_class1), np.zeros(p_class2))).reshape(-1, 1)

imp.reload(NN)
imp.reload(u)

dataset = np.hstack((X, y))
np.random.shuffle(dataset)

split = int(X.shape[0]*0.9)

train = dataset[:split, :]
validation = dataset[split:, :]

X_train, y_train = np.hsplit(train, [X.shape[1]])
X_va, y_va = np.hsplit(validation, [X.shape[1]])

nn = NN.NeuralNetwork(X_train, y_train, eta=0.4,
                      alpha=0.1,
                      hidden_sizes=[3],
                      reg_method='l2', reg_lambda=0.0,
                      epochs=1000,
                      batch_size=100,
                      activation='sigmoid',
                      task='classifier',
                      w_par=6)
nn.train(X_train, y_train, X_va, y_va)
u.plot_learning_curve(nn)

y_pred = nn.predict(X)

np.abs((np.round(y_pred, 0)-y)).sum()

nn.error_per_epochs_va
plt.close()

###########################################################

# Testing holdout

imp.reload(holdout)

# defining the grid
par_ranges = dict()
par_ranges['eta'] = (0.02, 2.0)
par_ranges['alpha'] = (0.0, 0.5)
par_ranges['batch_size'] = (1, 100)
par_ranges['hidden_sizes'] = [(1, 100)]
par_ranges['reg_lambda'] = (0.0, 0.1)

nn = NN.NeuralNetwork(X, y)

pars = nn.get_params().keys()

grid_size = 100
grid = val.HyperRandomGrid(par_ranges, N=grid_size )

hold = holdout.Holdout(X, y)
# model = hold.model_selection(grid)

# hold.best_index
# pprint(model.get_params())

###########################################################

# Testing Cross Validation

# defining grid
param_ranges = dict()
param_ranges['eta'] = (0.02, 2.0)
param_ranges['alpha'] = 0.001
param_ranges['batch_size'] = (1, 100)
param_ranges['hidden_sizes'] = [(1, 100), (10, 20)]
param_ranges['reg_lambda'] = (0.0, 0.1)
param_ranges['reg_method'] = 'l2'
param_ranges['epochs'] = 200

imp.reload(val)

# uniform grid
grid_size = 3
grid = val.HyperGrid(param_ranges, size=grid_size, random=False)

i=0
for hyperparam in grid:
    pprint(hyperparam)
    i+=1

# random grid
grid_size = 10
grid = val.HyperGrid(param_ranges, size=grid_size, random=True)
len(grid)


selection = val.ModelSelectionCV(grid=grid, repetitions=2)

start = time.time()
selection.search(X, y, nfolds=3)
end = time.time()

print end-start
results = selection.load_results()

pprint(selection.select_best_hyperparams(top=7))

best_model = selection.select_best_model(X, y)
