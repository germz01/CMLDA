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
epochs = 2000
beta_m = None
momentum = None
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

# STANDARDIZATION #############################################################

# design_set = (design_set - np.mean(design_set, axis=0)) / \
#     np.std(design_set, axis=0)
# test_set = (test_set - np.mean(test_set, axis=0)) / \
#     np.std(test_set, axis=0)


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
    neural_net = nn.NeuralNetwork(X_training, y_training,
                                  hidden_sizes=[16, 32],
                                  activation='sigmoid', task='regression')
    initial_W, initial_b = neural_net.W, neural_net.b

###############################################################################
# PRELIMINARY TRAINING ########################################################
#
pars = {}
betas = ['hs', 'mhs', 'fr', 'pr']
errors, errors_std = [], []
opt = raw_input("OPTIMIZER[SGD/CGD]: ")
betas = ['hs', 'mhs', 'pr']
errors, errors_std = [], []
mse_tr, mse_ts = list(), list()
convergence_ts, ls_ts = list(), list()   # mod
tot, bw, ls, dr = list(), list(), list(), list()   # mod
bw_p, ls_p, dr_p = list(), list(), list()   # mod

statistics = pd.DataFrame(columns=['DATASET', 'MEAN_MSE_TR', 'STD_MSE_TR',
                                   'MEAN_MSE_TS', 'STD_MSE_TS',
                                   'CONVERGENCE', 'LS'])   # mod

statistics_time = pd.DataFrame(columns=['DATASET', 'TOT', 'BACKWARD', 'LS',
                                        'DIRECTION'])

if testing:
    if opt == 'SGD':
        momentum = raw_input("MOMENTUM[nesterov/standard]: ")
        pars = {'epochs': epochs,
                'batch_size': X_training.shape[0],
                'eta': 0.1,
                'momentum': {'type': momentum, 'alpha': .9},
                'reg_lambda': 0.01,
                'reg_method': 'l2'}

        neural_net.train(X_training, y_training, opt, X_va=X_validation,
                         y_va=y_validation, **pars)
    else:
        beta_choice = raw_input("BETA[mhs/hs/pr]: ")
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
            pars['beta_m'] = beta_choice
            pars['d_m'] = 'modified'
            pars['sigma_2'] = 0.9
            neural_net.train(X_training, y_training, opt, X_va=X_validation,
                             y_va=y_validation, **pars)

    if testing_betas:
        u.plot_betas_learning_curves('CUP', betas, [errors_std, errors],
                                     'ERRORS', 'MSE')
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

        path = '../data/final_setup/analytics/CUP/' + str(opt)

        if momentum is not None:
            path += '/' + str(momentum)
        if opt == 'CGD':
            path += '/' + str(beta_choice)

        with open(path + '/CUP_curves_{}.json'.
                  format(opt.lower()), 'w') as json_file:
            curves_data = {'error': neural_net.optimizer.
                           error_per_epochs,
                           'error_va': neural_net.optimizer.
                           error_per_epochs_va,
                           'gradient_norm': neural_net.optimizer.
                           gradient_norm_per_epochs,
                           'time': neural_net.optimizer.
                           time_per_epochs}
            json.dump(curves_data, json_file, indent=4)

        mse_tr.append(neural_net.optimizer.error_per_epochs[-1])
        mse_ts.append(neural_net.optimizer.error_per_epochs_va[-1])
        convergence_ts.append(neural_net.optimizer.statistics['epochs'])
        ls_ts.append(neural_net.optimizer.statistics['ls'])   # mod
        tot.append(neural_net.optimizer.statistics['time_train'])
        bw.append(neural_net.optimizer.statistics['time_bw'])
        ls.append(neural_net.optimizer.statistics['time_ls'])
        dr.append(neural_net.optimizer.statistics['time_dr'])

        statistics.loc[statistics.shape[0]] = ['CUP_',
                                               np.mean(mse_tr), np.std(mse_tr),
                                               np.mean(mse_ts), np.std(mse_ts),
                                               np.mean(convergence_ts),  # mod
                                               np.mean(ls_ts)]

        statistics_time.loc[statistics_time.shape[0]] = \
            ['CUP_', np.mean(tot), np.mean(bw), np.mean(ls),
                np.mean(dr)]

        file_name = None

        if opt == 'SGD':
            file_name = path + \
                '/{}_CUP_statistics.csv'.format(momentum)
            file_name_time = path + \
                '/{}_CUP_time_statistics.csv'.format(momentum)
        else:
            file_name = path + \
                '/CUP_statistics.csv'
            file_name_time = path + \
                '/CUP_time_statistics.csv'

        statistics.to_csv(path_or_buf=file_name, index=False)
        statistics_time.to_csv(path_or_buf=file_name_time, index=False)

        #
        u.plot_all_learning_curves('CUP', ['nesterov'], [[neural_net.optimizer.
                                        gradient_norm_per_epochs],
                               [neural_net.optimizer.
                                time_per_epochs]],
                               'NORM', 'NORM', type='momentum', time=True,
                               fname=path)
        u.plot_all_learning_curves('CUP', ['nesterov'], [[neural_net.optimizer.
                                        error_per_epochs],
                               [neural_net.optimizer.
                                time_per_epochs]],
                               'ERROR', 'MSE', type='momentum', time=True,
                               fname=path)
###############################################################################
# VALIDATION ##################################################################

if validation:
    experiment = 1
    param_ranges = {}

    if opt == 'SGD':
        param_ranges['eta'] = (0.01, 0.1)

        type_m = raw_input('MOMENTUM TYPE[standard/nesterov]: ')
        assert type_m in ['standard', 'nesterov']
        param_ranges['type'] = type_m

        param_ranges['alpha'] = (0.7, 0.9)
        param_ranges['reg_method'] = 'l2'
        param_ranges['reg_lambda'] = (0.001, 0.01)
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
        if beta_m == 'mhs':
            param_ranges['rho'] = (0., 1.)
        else:
            param_ranges['rho'] = 0.0
    param_ranges['optimizer'] = opt
    param_ranges['hidden_sizes'] = [16, 32]
    param_ranges['activation'] = 'sigmoid'
    param_ranges['task'] = 'regression'

    grid = val.HyperGrid(param_ranges, grid_size, random=True)
    selection = val.ModelSelectionCV(grid,
                                     fname=fpath +
                                     'experiment_{}_results.json.gz'.
                                     format(experiment))
    selection.search(X_design, y_design, nfolds=nfolds)
    best_hyperparameters = selection.select_best_hyperparams(error='mee')

    path = '../data/final_setup/' + str(opt) + '/CUP'
    if beta_m is not None:
        path += '/' + str(beta_m)
    else:
        path += '/' + type_m
    with open(path + '/CUP_best_hyperparameters_{}.json'.
              format(opt.lower()),
              'w') as json_file:
        json.dump(best_hyperparameters, json_file, indent=4)
