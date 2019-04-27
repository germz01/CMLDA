import json
import nn
import numpy as np
import pandas as pd
import utils
import warnings
import ipdb
from tqdm import tqdm

warnings.filterwarnings("ignore")

###############################################################################
# EXPERIMENTAL SETUP ##########################################################

ntrials = 10
split_percentage = 0.8
epochs = 500
path_to_json = '../data/final_setup/'

statistics = pd.DataFrame(columns=['DATASET', 'MEAN_MSE_TR', 'STD_MSE_TR',
                                   'MEAN_MSE_TS', 'STD_MSE_TS'])

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

X_training = (X_training - np.mean(X_training, axis=0)) / \
    np.std(X_training, axis=0)
X_validation = (X_validation - np.mean(X_validation, axis=0)) / \
    np.std(X_validation, axis=0)

###############################################################################
# OPTIMIZER SELECTIONS ########################################################

params, opt = None, raw_input('CHOOSE AN OPTIMIZER[SGD/CGD]: ')

###############################################################################
# PARAMETERS SELECTION AND TESTING ############################################

mse_tr, mse_ts = list(), list()

beta = None


if opt == 'CGD':
    beta = raw_input('CHOOSE A BETA[hs/mhs/pr]: ')
    assert beta in ['hs', 'mhs', 'pr']

sample = None if raw_input('SAMPLE A LEARING CURVE?[Y/N] ') == 'N' else \
        np.random.randint(0, ntrials)


if opt == 'SGD':
    hps = path_to_json + \
            'SGD/CUP_best_hyperparameters_sgd.json'
else:
    hps = path_to_json + \
            'CGD/' + str(beta) + '/CUP_best_hyperparameters_cgd.json'

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
                  desc='TESTING DATASET'):
    neural_net = nn.NeuralNetwork(X_training, y_training,
                                  hidden_sizes=hidden_sizes,
                                  activation='sigmoid', task='regression')
    neural_net.train(X_training, y_training, opt, epochs=epochs,
                     X_va=X_validation,
                     y_va=y_validation, **params['hyperparameters'])

    mse_tr.append(neural_net.optimizer.error_per_epochs[-1])
    mse_ts.append(neural_net.optimizer.error_per_epochs_va[-1])
    neural_net.restore_weights()

    if sample is not None and sample == trial:
        saving_str = 'cup' if opt == 'SGD' else \
            '{}_cup'.format(beta)

        utils.plot_learning_curve_with_info(
            neural_net.optimizer,
            [neural_net.optimizer.error_per_epochs,
                neural_net.optimizer.error_per_epochs_va],
            'TEST', 'MSE', neural_net.optimizer.params,
            fname=saving_str)

statistics.loc[statistics.shape[0]] = ['CUP',
                                       np.mean(mse_tr), np.std(mse_tr),
                                       np.mean(mse_ts), np.std(mse_ts)]

file_name = None

if opt == 'SGD':
    file_name = fpath + opt.lower() + '_cup_statistics.csv'
else:
    file_name = fpath + opt.lower() + '_' + beta + '_cup_statistics.csv'

statistics.to_csv(path_or_buf=file_name, index=False)

