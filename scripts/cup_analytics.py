import ipdb
import json
import nn
import numpy as np
import pandas as pd
import utils as u
import validation as val
import warnings

warnings.filterwarnings("ignore")

###############################################################################
# EXPERIMENTAL SETUP ##########################################################

nfolds = 3
grid_size = 20
split_percentage = 0.7
epochs = 500

###############################################################################
# LOADING DATASET #############################################################

fpath = '../data/CUP/'

ds_tr = pd.read_csv(fpath + 'ML-CUP18-TR.csv', skiprows=10, header=None)
ds_ts = pd.read_csv(fpath + 'ML-CUP18-TS.csv', skiprows=10, header=None)
ds_tr.drop(columns=0, inplace=True)
ds_ts.drop(columns=0, inplace=True)
ds_tr, ds_ts = ds_tr.values, ds_ts.values

X_design, y_design = np.hsplit(ds_tr, [10])
X_test, y_test = np.hsplit(ds_ts, [10])

design_set = np.hstack((y_design, X_design))
test_set = np.hstack((y_test, X_test))

###############################################################################
# DATASET PARTITIONING ########################################################

np.random.shuffle(design_set)

training_set = design_set[:int(design_set.shape[0]*split_percentage), :]
validation_set = design_set[int(design_set.shape[0]*split_percentage):, :]

y_training, X_training = np.hsplit(training_set, [2])
y_validation, X_validation = np.hsplit(validation_set, [2])

###############################################################################
# NETWORK INITIALIZATION ######################################################

testing, testing_betas, validation = False, False, False

if raw_input('TESTING OR VALIDATION[testing/validation]? ') == 'validation':
    validation = True
else:
    testing, testing_betas = True, True \
        if raw_input('TESTING BETAS[Y/N]? ') == 'Y' else False

neural_net, initial_W, initial_b = None, None, None

if testing or testing_betas:
    neural_net = nn.NeuralNetwork(X_training, y_training, hidden_sizes=[10],
                                  activation='sigmoid', task='regression')
    initial_W, initial_b = neural_net.W, neural_net.b

###############################################################################
# PRELIMINARY TRAINING ########################################################
#
pars = {}
betas = ['hs', 'mhs', 'fr', 'pr']
errors, errors_std = [], []
opt = raw_input("OPTIMIZER[SGD/CGD]: ")

if testing:
    if opt == 'SGD':
        pars = {'epochs': epochs,
                'batch_size': X_training.shape[0],
                'eta': 0.5,
                'momentum': {'type': 'nesterov', 'alpha': 0.9},
                'reg_lambda': 0.0,
                'reg_method': 'l2'}

        neural_net.train(X_training, y_training, opt, X_va=X_validation,
                         y_va=y_validation, **pars)
    else:
        pars = {'max_epochs': epochs,
                'error_goal': 1e-4,
                'strong': True,
                'rho': 0.5}
        if testing_betas:
            for beta in betas:
                pars['beta_m'] = beta

                for d_m in ['standard', 'modified']:
                    print 'TESTING BETA {} WITH DIRECTION {}'.\
                        format(beta.upper(), d_m.upper())

                    pars['plus'] = True
                    pars['d_m'] = d_m

                    neural_net.train(X_training, y_training, opt,
                                     X_va=X_validation, y_va=y_validation,
                                     **pars)
                    neural_net.update_weights(initial_W, initial_b)
                    neural_net.update_copies()

                    if d_m == 'standard':
                        errors_std.\
                            append(neural_net.optimizer.error_per_epochs)
                    else:
                        errors.\
                            append(neural_net.optimizer.error_per_epochs)

        else:
            pars['plus'] = True
            pars['beta_m'] = 'mhs'
            neural_net.train(X_training, y_training, opt, X_va=X_validation,
                             y_va=y_validation, **pars)

    if testing_betas:
        u.plot_betas_learning_curves('CUP', betas, [errors_std, errors],
                                     'ERRORS', 'MEE')
    else:
        print '\n'
        print 'INITIAL ERROR: {}'.\
            format(neural_net.optimizer.error_per_epochs[0])
        print 'FINAL ERROR: {}'.\
            format(neural_net.optimizer.error_per_epochs[-1])
        print 'INITIAL VALIDATION ERROR: {}'.format(neural_net.optimizer.
                                                    error_per_epochs_va[0])

        print 'FINAL VALIDATION ERROR: {}'.format(neural_net.optimizer.
                                                  error_per_epochs_va[-1])
        print 'EPOCHS OF TRAINING {}'.format(len(neural_net.optimizer.
                                                 error_per_epochs))
        print '\n'

        u.plot_learning_curve_with_info(
            neural_net.optimizer,
            [neural_net.optimizer.error_per_epochs,
             neural_net.optimizer.error_per_epochs_va], 'VALIDATION',
            'MEE', neural_net.optimizer.params)

###############################################################################
# VALIDATION ##################################################################

if validation:
    experiment = 1
    param_ranges = {}

    if opt == 'SGD':
        param_ranges['eta'] = (0.3, 7.)

        type_m = raw_input('MOMENTUM TYPE[standard/nesterov]: ')
        assert type_m in ['standard', 'nesterov']
        param_ranges['type'] = type_m

        param_ranges['alpha'] = (0.5, 0.9)
        param_ranges['reg_method'] = 'l2'
        param_ranges['reg_lambda'] = 0.0
        param_ranges['epochs'] = epochs
    else:
        beta_m = raw_input('CHOOSE A BETA[hs/mhs/fr/pr]: ')
        assert beta_m in ['hs', 'mhs', 'fr', 'pr']
        param_ranges['beta_m'] = beta_m

        d_m = raw_input('CHOOSE A DIRECTION METHOD[standard/modified]: ')
        assert d_m in ['standard', 'modified']
        param_ranges['d_m'] = d_m

        param_ranges['max_epochs'] = epochs
        param_ranges['error_goal'] = 1e-4
        param_ranges['strong'] = True
        param_ranges['plus'] = True
        param_ranges['sigma_2'] = (0.1, 0.4)
        param_ranges['rho'] = (0., 1.)

    param_ranges['optimizer'] = opt
    param_ranges['hidden_sizes'] = [4, 8]
    param_ranges['activation'] = 'sigmoid'
    param_ranges['task'] = 'regression'

    grid = val.HyperGrid(param_ranges, grid_size, random=True)
    selection = val.ModelSelectionCV(grid,
                                     fname=fpath +
                                     'experiment_{}_results.json.gz'.
                                     format(experiment))
    selection.search(X_design, y_design, nfolds=nfolds)
    best_hyperparameters = selection.select_best_hyperparams(error='mee')

    with open('../data/final_setup/CUP_best_hyperparameters_{}.json'.
              format(opt.lower()),
              'w') as json_file:
        json.dump(best_hyperparameters, json_file, indent=4)


