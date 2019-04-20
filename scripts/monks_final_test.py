import json
import ipdb
import nn
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

###############################################################################
# EXPERIMENTAL SETUP ##########################################################

ntrials = 10
split_percentage = 0.8
epochs = 500
path_to_json = '../data/final_setup/'

statistics = pd.DataFrame(columns=['DATASET', 'MEAN_MSE_TR', 'STD_MSE_TR',
                                   'MEAN_MSE_TS', 'STD_MSE_TS',
                                   'MEAN_ACCURACY_TR', 'STD_ACCURACY_TR',
                                   'MEAN_ACCURACY_TS', 'STD_ACCURACY_TS',
                                   'CONVERGENCE'])

###############################################################################
# LOADING DATASET #############################################################

fpath = '../data/monks/'

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']

datasets = {name: pd.read_csv(fpath+name+'_bin.csv').values
            for name in names}

X_designs, y_designs = [], []
X_tests, y_tests = [], []

for ds in [1, 2, 3]:
    design_set = datasets['monks-{}_train'.format(ds)]
    test_set = datasets['monks-{}_test'.format(ds)]

    y_design, X_design = np.hsplit(design_set, [1])
    y_test, X_test = np.hsplit(test_set, [1])

    # simmetrized X_design:
    X_design = (X_design*2-1)
    X_test = (X_test*2-1)
    design_set = np.hstack((y_design, X_design))
    test_set = np.hstack((y_test, X_test))

    X_designs.append(X_design)
    y_designs.append(y_design)
    X_tests.append(X_test)
    y_tests.append(y_test)

###############################################################################
# OPTIMIZER SELECTIONS ########################################################

params, opt = None, raw_input('CHOOSE AN OPTIMIZER[SGD/CGD]: ')

###############################################################################
# PARAMETERS SELECTION AND TESTING ############################################

mse_tr, mse_ts = list(), list()
acc_tr, acc_ts = list(), list()

beta = None

if opt == 'CGD':
    beta = raw_input('CHOOSE A BETA[hs/mhs/fr/pr]: ')
    assert beta in ['hs', 'mhs', 'fr', 'pr']

for ds in [0, 1, 2]:
    if opt == 'SGD':
        hps = path_to_json + \
            'monks_{}_best_hyperparameters_sgd.json'.format(ds + 1)
    else:
        hps = path_to_json + \
            'monks_{}_best_hyperparameters_cgd_{}.json'.format(ds + 1, beta)

    with open(hps) as json_file:
        params = json.load(json_file)

    hidden_sizes = [int(i) for i in
                    params['hyperparameters']['topology'].split(' -> ')]
    hidden_sizes = hidden_sizes[1:-1]

    if opt == 'SGD':
        params['hyperparameters']['momentum'] = \
            {'type': params['hyperparameters']['momentum_type'],
             'alpha': params['hyperparameters']['alpha']}
        params['hyperparameters'].pop('momentum_type')
        params['hyperparameters'].pop('alpha')

    params['hyperparameters'].pop('activation')
    params['hyperparameters'].pop('topology')

    for trial in tqdm(range(ntrials),
                      desc='TESTING DATASET {}'.format(ds + 1)):

        neural_net = nn.NeuralNetwork(X_designs[ds], y_designs[ds],
                                      hidden_sizes=hidden_sizes,
                                      activation='sigmoid', task='classifier')
        neural_net.train(X_designs[ds], y_designs[ds], opt, epochs=epochs,
                         X_va=X_tests[ds],
                         y_va=y_tests[ds], **params['hyperparameters'])

        mse_tr.append(neural_net.optimizer.error_per_epochs[-1])
        mse_ts.append(neural_net.optimizer.error_per_epochs_va[-1])
        acc_tr.append(neural_net.optimizer.accuracy_per_epochs[-1])
        acc_ts.append(neural_net.optimizer.accuracy_per_epochs_va[-1])
        neural_net.restore_weights()

    statistics.loc[statistics.shape[0]] = ['MONKS_{}'.format(ds + 1),
                                           np.mean(mse_tr), np.std(mse_tr),
                                           np.mean(mse_ts), np.std(mse_ts),
                                           np.mean(acc_tr), np.std(acc_tr),
                                           np.mean(acc_ts), np.std(acc_ts),
                                           neural_net.optimizer.convergence]

file_name = None

if opt == 'SGD':
    file_name = fpath + opt.lower() + '_monks_statistics.csv'
else:
    file_name = fpath + opt.lower() + '_' + beta + '_monks_statistics.csv'

statistics.to_csv(path_or_buf=file_name, index=False)
