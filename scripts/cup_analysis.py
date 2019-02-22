import matplotlib as mpl
mpl.use('TkAgg')
# mpl.use('TkAgg')

import validation as validation
import imp
import numpy as np
import json
import utils as u

import pandas as pd
import matplotlib.pyplot as plt
# import plotnine as gg

import seaborn as sns

from pprint import pprint

###########################################################
###########################################################

'''
exp5/ 
Experiment using batch approach over large number of hidden sizes

'''


nexp = 1
# read experiment parameters
with open('../data/CUP/results/exp5/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp5/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

exp5 = selection.load_results_pandas(flat=False)

exp5.shape
exp5.sort_values(['mee_va'], inplace=True)
exp5.head(20).describe()['mee_va']

# binninng hidden sizes
hbin = 100
exp5['h_cat'] = (np.array(exp5['hidden_sizes'], dtype=int)/hbin)*hbin
exp5['h_cat']


# exp5_filtered = exp5.query('mee_va<1.4')
# exp5_filtered = exp5.query('hidden_sizes>1000')
exp5_filtered = exp5

exp5.describe()['mee_va']

# plt.ion()

plt.close()
sns.lmplot(
    data=exp5_filtered,
    x='h_cat',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp5_filtered['h_cat'], y=exp5_filtered['mee_va'],
            color='gray', alpha=0.3)
plt.grid()

plt.close()
sns.lineplot(
    data=exp5_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
)




# plotting learning curve
exp5.sort_values(['mee_va'], inplace=True)
exp5_filtered.sort_values(['mee_va'], inplace=True)

exp5.head()['mee_va']

exp5.shape

imp.reload(u)
max_epochs = len(exp5.iloc[0]['error_per_epochs'])
min_epochs = 50
for i in range(200):
    row = exp5.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:max_epochs],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:max_epochs],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp5_img/exp_{}_curve_{:02d}.png'.format(
            nexp, i)
    )

###########################################################

'''
adding exp6/
'''
nexp = 1
# read experiment parameters
with open('../data/CUP/results/exp6/cup_experiment_{}_parameters.json'.format(nexp)) as f:
    experiment_params = json.load(f)

pprint(experiment_params)

# load results
selection = validation.ModelSelectionCV(
    grid=None,
    fname='../data/CUP/results/exp6/' +
    'cup_experiment_{}_results.json.gz'.format(nexp))

exp6 = selection.load_results_pandas(flat=False)

exp6.sort_values(['mee_va'], inplace=True)
exp6.head(20).describe()['mee_va']
exp6.shape

plt.close()
sns.lmplot(
    data=exp6_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp6_filtered['hidden_sizes'], y=exp6_filtered['mee_va'],
            color='gray', alpha=0.3)
plt.grid()


###########################################################

'''
Wide grid search

'''

exp56 = pd.concat([exp5, exp6])

exp5.shape
exp6.shape
exp56.shape

# hbin = 250
# exp56['h_cat'] = (np.array(exp56['hidden_sizes'], dtype=int)/hbin)*hbin
# exp56['h_cat']

# exp56_filtered = exp56.query('id_fold!=1')

exp56_filtered = exp56

plt.close()
sns.lmplot(
    data=exp56_filtered,
    x='hidden_sizes',
    y='mee_va',
    hue='id_fold',
    robust=False,
    scatter=False
)
plt.scatter(x=exp56_filtered['hidden_sizes'], y=exp56_filtered['mee_va'],
            color='gray', alpha=0.3)
plt.grid()


# group by each parameter (aggregazione sui fold)
exp56_bygrid = exp56.groupby('id_grid').agg({
    'mee_va': [np.median, np.mean, np.std],
})
exp56_bygrid.columns = ['_'.join(x) for x in exp56_bygrid.columns.ravel()]

exp56_bygrid.head()

exp56_merged = pd.merge(exp56_bygrid, exp56, on='id_grid')
exp56_merged.head()

plt.close()
plt.scatter(exp56_merged['hidden_sizes'], exp56_merged['mee_va_mean'])

# aggrego per gruppi di hidden size, per poi calcolare il minimo
hbin = 50
exp56_merged['h_cat'] = (np.array(exp56_merged['hidden_sizes'],
                                  dtype=int)/hbin)*hbin
exp56_merged['h_cat'].head()

# calcolo il minimo sui gruppi di hidden_sizes
exp56_min = exp56_merged.groupby('h_cat').agg({
    'mee_va_median': [np.min],
    'mee_va_mean': [np.min],
})

exp56_min.columns = ['_'.join(x) for x in exp56_min.columns.ravel()]

exp56_min.head()
exp56_min['h_cat']=exp56_min.index

exp56_min.head()

plt.close()
plt.scatter('h_cat', 'mee_va_mean_amin', data=exp56_min)
plt.grid()

# scelta della griglia fine finale
finer_grid = dict()
finer_grid['hidden_sizes'] = [(1500,2000)]

###########################################################

plt.close()
plt.scatter(x=exp56_min['h_cat'],
            y=exp56_min['mee_va_mean_amin'],
            color='blue', alpha=0.7)

plt.scatter(x=exp56_min['h_cat'],
            y=exp56_min['mee_va_median_amin'],
            color='red', alpha=0.99)
plt.grid()
plt.scatter(x=exp56['h_cat'],
            y=exp56['mee_va'],
            color='gray', alpha=0.3)

'''
# guardiamo un pò le learning curve allora

'''
exp56_500 = exp56.query('hidden_sizes>200 and hidden_sizes<800')

exp56_500.sort_values(['mee_va'], inplace=True)
exp56_500.head()[['mee_va', 'eta']]


toplot=exp56_500
min_epochs = 50
for i in range(10):
    row = toplot.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/exp5_img/exp_{}_curve_{:02d}.png'.format(
            nexp, i)
    )

###########################################################
# scelta parametri griglia fine

filtered = exp56.query('hidden_sizes>1500 and hidden_sizes>2000')
filtered.sort_values(['mee_va'], inplace=True)

filtered['max_epochs'] = np.array([len(filtered.iloc[i]['error_per_epochs']) for i in range(filtered.shape[0]) ])

# filtered.query('mee_va<1.15')
filtered.describe()[['eta', 'hidden_sizes','max_epochs']]

# choosing max_epochs
plt.close()
plt.scatter(x='mee_va', y='max_epochs', data=filtered)
plt.ylabel('y_epochs')
plt.axhline(1500)
plt.axhline(1000)


finer_grid['max_epochs'] = 1200
pprint(finer_grid)

# choosing eta

plt.close()
plt.scatter(x='mee_va', y='eta', data=filtered)
plt.xlabel('errors')
plt.ylabel('eta')
plt.grid()

finer_grid['eta'] = (0.01, 0.025)

toplot=filtered.query('mee_va < 1.150 and eta < 0.01')
toplot.shape

min_epochs = 0
for i in range(50):
    row = toplot.iloc[i]
    u.plot_learning_curve_info(
        error_per_epochs=row['mee_per_epochs'][min_epochs:],
        error_per_epochs_va=row['mee_per_epochs_va'][min_epochs:],
        task='validation',
        hyperparams=row['hyperparams'],
        fname='../data/CUP/results/choose_finer_grid/exp_{}_curve_{:02d}.png'.format(
            nexp, i)
    )


pprint(finer_grid)


# timing estimation

training_time = 80 # seconds
n_folds = 3
grid_size = 30

# ore stimate
overall = training_time*n_folds*grid_size  
overall/3600.



pprint(finer_grid)
