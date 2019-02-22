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
import random
import string

from pprint import pprint

###########################################################
# EXPERIMENTAL SETUP

dataset = 3

grid_size = 300

nfolds = 5
ntrials = 7

param_ranges = {
    'eta': (0.2, 10.0),
    'hidden_sizes': [(2, 4)],
    'alpha': (0.5, 0.9),
    'reg_method': 'l2', 'reg_lambda': 0.0,
    'epochs': 500,
    'batch_size': 'batch',
    'activation': 'sigmoid',
    'task': 'classifier',
    # 'early_stop': None,  # 'testing',
    'epsilon': 5,
    'w_method': 'DL',
    'w_par': 6.0}

info = "Informazioni/appunti/scopo riguardo l'esperimento in corso"

info = "batch - no early - sigmoid"

experiment_params = {
    'dataset': dataset,
    'nfolds': nfolds,
    'ntrials': ntrials,
    'grid_size': grid_size,
    'info': info,
    'param_ranges': param_ranges
}

###########################################################
# LOADING DATASET

fpath = '../data/monks/'
preliminary_path = '../images/monks_preliminary_trials/'

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']

datasets = {name: pd.read_csv(fpath+name+'_bin.csv').values
            for name in names}

design_set = datasets['monks-{}_train'.format(dataset)]
test_set = datasets['monks-{}_test'.format(dataset)]

y_design, X_design = np.hsplit(design_set, [1])
y_test, X_test = np.hsplit(test_set, [1])

# simmetrized X_design:
X_design = (X_design*2-1)
X_test = (X_test*2-1)
design_set = np.hstack((y_design, X_design))
test_set = np.hstack((y_test, X_test))

###########################################################
# babysitting for choosing initial param ranges
# and plot_learning_curve

imp.reload(u)
imp.reload(NN)

np.random.shuffle(design_set)
# splitting training/validation
split_percentage = 0.8
split = int(design_set.shape[0]*split_percentage)

training_set = design_set[:split, :]
validation_set = design_set[split:, :]

y_training, X_training = np.hsplit(training_set, [1])
y_validation, X_validation = np.hsplit(validation_set, [1])

training_set.shape
validation_set.shape

b_s = 'batch'

nn = NN.NeuralNetwork(
    X_design, y_design,
    eta=0.5,
    hidden_sizes=[3],
    alpha=0.5,
    reg_method='l2', reg_lambda=0.01,
    epochs=1000,
    batch_size=b_s, #'batch',  # 'batch',
    activation='relu',
    task='classifier',
    # early_stop='testing',  # 'testing',
    epsilon=5,
    # w_method='DL',
    # w_par=6,
    w_method='uniform',
    w_par=1./17
)
nn.train(X_design, y_design, X_test, y_test)
nn.w_par
epochs_plot = 1000

preliminary_name = preliminary_path + 'monks_{}_mb_{}_{}.pdf'.\
    format(dataset,
           b_s,
           ''.join([random.choice(string.ascii_letters + string.digits)
                   for n in xrange(4)]))

u.plot_learning_curve_info(
    nn.error_per_epochs[:epochs_plot],
    nn.error_per_epochs_va[:epochs_plot],
    nn.get_params(),
    task='validation',
    accuracy_h_plot=True,
    accuracy_per_epochs=nn.accuracy_per_epochs[:epochs_plot],
    accuracy_per_epochs_va=nn.accuracy_per_epochs_va[:epochs_plot],
    fname=preliminary_name)


# u.plot_learning_curve(nn, fname='../images/monks_learning_curve.pdf')
# u.plot_learning_curve(nn, fname='../images/monks_{}_{}_{}.pdf'.format(dataset, 'stochastic', 'notearly', 'relu'))

y_pred = nn.predict(X_test)
y_pred = np.apply_along_axis(lambda x: 0 if x < .5 else 1, 1,
                                         y_pred).reshape(-1, 1)
# y_pred = np.round(y_pred)
bca = metrics.BinaryClassifierAssessment(y_pred, y_test,
                                         printing=True)

y_pred_test = np.round(nn.predict(X_test))
metrics.BinaryClassifierAssessment(y_test, y_pred_test)

###########################################################

nn.h[0].shape
nn.h[1]

nn.W[0]


np.round(nn.h[0],2)





###########################################################
# EXPERIMENT GRID SEARCH

# controllo nomi files
fpath = '../data/monks/results/monk_{}/'.format(dataset)

check_files = True
experiment = 1

while(check_files):
    fname_results = 'monks_{}_experiment_{}_results.json.gz'.format(
        dataset, experiment)
    fname_params = 'monks_{}_experiment_{}_parameters.json'.format(
        dataset, experiment)

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

float(np.sum(y_training))/len(y_training)
float(np.sum(y_validation))/len(y_validation)

###########################################################
# starting search

grid = valid.HyperGrid(param_ranges, grid_size, random=True)
selection = valid.ModelSelectionCV(grid, fname=fres)

start = time.time()
selection.search(X_design, y_design, nfolds=nfolds, ntrials=ntrials)
end = time.time()

###########################################################
