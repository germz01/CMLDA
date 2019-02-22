import numpy as np
import pandas as pd

import nn as NN
import utils as u
import validation as valid
import metrics

import time
import json
import os.path
import imp

from pprint import pprint

###########################################################

# LOADING DATASET, and splitting immediately an internal test set

fpath = '../data/CUP/'

df_original_training_set = pd.read_csv(fpath+'ML-CUP18-TR.csv', skiprows=10, header=None)
colnames = ['id']+[ 'x{}'.format(i) for i in range(1,11)]+['y_1', 'y_2']

df_original_training_set.columns = colnames

df_original_training_set.head()
df_original_training_set = df_original_training_set.drop(['id'], axis=1)
df_original_training_set.head()

df_original_training_set.describe()

original_training_set = df_original_training_set.values

np.random.seed(27)
np.random.shuffle(original_training_set)

# separo il test set che useremo per la valutazione dei modelli
split_test = int(original_training_set.shape[0]*0.05)

test_set = original_training_set[:split_test, :]
design_set = original_training_set[split_test:, ]

X_test, y_test = np.hsplit(test_set, [10])

test_set.shape
design_set.shape

###########################################################

# babysitting for choosing parameter ranges
X_design, y_design = np.hsplit(design_set, [10])

imp.reload(u)
imp.reload(NN)

np.random.shuffle(design_set)

# splitting training/validation
split_percentage = 0.66
split = int(design_set.shape[0]*split_percentage)

training_set = design_set[:split, :]
validation_set = design_set[split:, :]

X_training, y_training = np.hsplit(training_set, [10])

np.mean(X_training, axis=0)

X_validation, y_validation = np.hsplit(validation_set, [10])

# standardization
X_training_std = (X_training-np.mean(X_training, axis=0))/np.std(X_training, axis=0)
X_validation_std = (X_validation-np.mean(X_validation, axis=0))/np.std(X_validation, axis=0)

training_set.shape
validation_set.shape
X_training.shape
y_training.shape


# babysitting
imp.reload(NN)
nn = NN.NeuralNetwork(
    X_training, y_training,
    # eta=0.004, buono
    eta=0.02,
    # eta=0.00005,
    # eta=0.00002,
    hidden_sizes=[40],
    alpha=0.9,
    reg_method='l2', reg_lambda=0.002,
    epochs=2000,
    batch_size='batch', #'batch',
    activation='relu',
    task='regression',
    early_stop=None,  # '', # 'testing',  # 'testing',
    epsilon=1,
    early_stop_min_epochs=500,
    w_method='DL',
    w_par=6,
    # w_method='uniform',
    # w_par=1./17
)
nn.train(X_training, y_training, X_validation, y_validation)


y_final = nn.mee_per_epochs_va[-1]
y_final
y_min = np.min(nn.mee_per_epochs_va)
y_min

y_final/y_min

import matplotlib.pyplot as plt
imp.reload(u)
imp.reload(NN)
epochs_plot_start = 100
epochs_plot_end = len(nn.error_per_epochs)
u.plot_learning_curve_info(
    nn.mee_per_epochs[epochs_plot_start:epochs_plot_end],
    nn.mee_per_epochs_va[epochs_plot_start:epochs_plot_end],
    nn.get_params(),
    accuracy=False,
    task='validation',
    fname='../images/cup_learning_curve')

y_min


y_pred_test = nn.predict(X_test)
metrics.mee(y_test, y_pred_test)

'''

###########################################################
# TOPOLOGY GRID

# [2**i for i in range(5, 20)]
'''
topologies = [int((3./2)**i) for i in range(8, 20)]
topologies

topologies.reverse()




for hidden in topologies:

    ###########################################################
    # EXPERIMENTAL SETUP

    grid_size = 10

    nfolds = 3
    ntrials = 1

    # eta=0.0005,
    # eta=0.00002,

    param_ranges = {
        'eta': (0.0005, 0.005),
        'hidden_sizes': [hidden],
        'alpha': 0.9,
        'reg_method': 'l2', 'reg_lambda': 0.0,
        'epochs': 1000,
        'batch_size': 128,
        'activation': 'relu',
        'task': 'regression',
        # 'early_stop': None,  # 'testing',
        'epsilon': 5,
        'w_method': 'DL',
        'w_par': 6.0}

    info = "Informazioni/appunti/scopo riguardo l'esperimento in corso"

    info = "infos"

    experiment_params = {
        'nfolds': nfolds,
        'ntrials': ntrials,
        'grid_size': grid_size,
        'info': info,
        'param_ranges': param_ranges
    }


    ###########################################################
    # EXPERIMENT GRID SEARCH

    # controllo nomi files
    fpath = '../data/CUP/results/exp1/'

    check_files = True
    experiment = 1

    while(check_files):
        fname_results = 'cup_experiment_{}_results.json.gz'.format(
            experiment)
        fname_params = 'cup_experiment_{}_parameters.json'.format(
            experiment)

        fres = fpath+fname_results
        fpar = fpath+fname_params

        if os.path.isfile(fres) or os.path.isfile(fpar):
            experiment += 1
        else:
            check_files = False


    print '--------------------'
    print 'STARTING EXPERIMENT'
    print 'saving results in:'
    print fres
    print fpar
    print '--------------------'
    # if raw_input('Starting search ?[Y/N] ') == 'Y':

    # save experiment setup
    experiment_params['experiment'] = experiment
    with open(fpar, 'w') as f:
        json.dump(experiment_params, f, indent=4)


    ###########################################################
    # starting search

    grid = valid.HyperGrid(param_ranges, grid_size, random=True)
    selection = valid.ModelSelectionCV(grid, fname=fres)

    selection.search(X_design, y_design, nfolds=nfolds, ntrials=ntrials)

    ###########################################################
'''


###########################################################
# EXPERIMENTAL SETUP

grid_size = 200

nfolds = 5
ntrials = 1

# eta=0.0005,
# eta=0.00002,

param_ranges = {
    'eta': (0.004, 0.02),
    'hidden_sizes': [(2, 100)],
    'alpha': 0.90,
    'reg_method': 'l2',
    'reg_lambda': 0.00,
    'epochs': 2000,
    'batch_size': 'batch',
    'activation': 'relu',
    'task': 'regression',
    'early_stop': 'GL',
    'early_stop_min_epochs' : 500,
    'epsilon': 1,
    'w_method': 'DL',
    'w_par': 6.0}

info = "Informazioni/appunti/scopo riguardo l'esperimento in corso"

info = "batch with early stopping"

experiment_params = {
    'nfolds': nfolds,
    'ntrials': ntrials,
    'grid_size': grid_size,
    'info': info,
    'param_ranges': param_ranges
}


###########################################################
# EXPERIMENT GRID SEARCH

# controllo nomi files
fpath = '../data/CUP/results/grid1/'

check_files = True
experiment = 1

while(check_files):
    fname_results = 'cup_experiment_{}_results.json.gz'.format(
        experiment)
    fname_params = 'cup_experiment_{}_parameters.json'.format(
        experiment)

    fres = fpath+fname_results
    fpar = fpath+fname_params

    if os.path.isfile(fres) or os.path.isfile(fpar):
        experiment += 1
    else:
        check_files = False


print '--------------------'
print 'STARTING EXPERIMENT'
print 'saving results in:'
print fres
print fpar
print '--------------------'
# if raw_input('Starting search ?[Y/N] ') == 'Y':

# save experiment setup
experiment_params['experiment'] = experiment
with open(fpar, 'w') as f:
    json.dump(experiment_params, f, indent=4)


###########################################################
# starting search

grid = valid.HyperGrid(param_ranges, grid_size, random=True)
selection = valid.ModelSelectionCV(grid, fname=fres)

selection.search(X_design, y_design, nfolds=nfolds, ntrials=ntrials)

###########################################################
