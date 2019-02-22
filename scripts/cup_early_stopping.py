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

from tqdm import tqdm
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

imp.reload(NN)

topologies = [int((3./2)**i) for i in range(8, 20)]
topologies.reverse()

topologies = np.array(np.random.uniform(10, 100, 20), dtype=int)
topologies

# topologies = [50]
ntrials = 2

import matplotlib.pyplot as plt
imp.reload(u)

results = []
i=0
for hidden in tqdm(topologies):
    for trial in tqdm(range(ntrials)):
        eta = np.random.uniform(0.004, 0.02)

        nn = NN.NeuralNetwork(
            X_training, y_training,
            eta=eta,
            # eta=0.00005,
            # eta=0.00002,
            hidden_sizes=[hidden],
            alpha=0.90,
            reg_method='l2', reg_lambda=0.000,
            epochs=5000,
            batch_size='batch',
            activation='relu',
            task='regression',
            early_stop='testing',  # 'testing',
            epsilon=1,
            early_stop_min_epochs=500,
            w_method='DL',
            w_par=6,
            # w_method='uniform',
            # w_par=1./17
        )
        nn.train(X_training, y_training, X_validation, y_validation)

        epochs_plot_start = 50
        epochs_plot_end = len(nn.error_per_epochs)
        u.plot_learning_curve_info(
            nn.error_per_epochs[epochs_plot_start:epochs_plot_end],
            nn.error_per_epochs_va[epochs_plot_start:epochs_plot_end],
            nn.get_params(),
            accuracy=False,
            task='validation',
            title='Early stopping comparison',
            fname='../data/CUP/results/early_stop/img/learning_early_{}'.format(i),
            MEE_TR=nn.mee_per_epochs[-1],
            MEE_VL=nn.mee_per_epochs_va[-1],
            stop_GL=nn.stop_GL,
            stop_PQ=nn.stop_PQ,
            figsize=(11, 7)
        )

        i += 1
        out = {
            'hidden_sizes': hidden,
            'eta': eta,
            'MEE_min': np.min(nn.mee_per_epochs_va),
            'MEE_final': nn.mee_per_epochs_va[-1],
            'MEE_GL': (nn.mee_per_epochs_va[nn.stop_GL]
                       if nn.stop_GL is not None else None),
            'MEE_PQ': (nn.mee_per_epochs_va[nn.stop_PQ]
                       if nn.stop_PQ is not None else None),
            'stop_min': np.argmin(nn.error_per_epochs_va),
            'stop_final': nn.epochs,
            'stop_GL': nn.stop_GL,
            'stop_PQ': nn.stop_PQ,
        }
        results.append(out)


df_early = pd.DataFrame(results)
df_early.to_csv('../data/CUP/results/early_stop/df_early_stop.csv')

# a = pd.read_csv('../data/CUP/results/early_stop/df_early_stop.csv')

