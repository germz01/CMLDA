import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns

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

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 15

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=BIGGER_SIZE)
# plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)



from pprint import pprint
###########################################################

early = pd.read_csv('../data/CUP/results/early_stop/df_early_stop.csv')

early.describe()

early.columns
plt.ion()
plt.close()
early.describe()[['MEE_min', 'MEE_PQ', 'MEE_GL']]
plt.scatter(x=early['MEE_min'], y=early['MEE_GL'])
plt.scatter(x=early['MEE_min'], y=early['MEE_PQ'])

plt.show()

plt.close()
plt.scatter(x=early['stop_min'], y=early['stop_GL'], label='GL')
plt.scatter(x=early['stop_min'], y=early['stop_PQ'], label='PQ')
plt.xlabel('Minimum MEE epochs')
plt.ylabel('Early stopping epochs')


plt.close()
sns.lineplot(x=early['hidden_sizes'], y=early['stop_GL']-early['stop_min'], label='GL', ci=None)
sns.lineplot(x=early['hidden_sizes'], y=early['stop_PQ']-early['stop_min'], label='GL', ci=None)

plt.close()

marker_size=70
plt.scatter(x=early['hidden_sizes'], y=early['stop_PQ']-early['stop_min'], label='PQ', s=marker_size)
plt.scatter(x=early['hidden_sizes'], y=early['stop_GL']-early['stop_min'], label='GL', marker='s', s=marker_size)
plt.legend()
plt.axhline(0)
#plt.axvline(100)
plt.grid()
plt.xlim((0,100))
plt.ylim((-1500,1000))


plt.xlabel('Number of hidden units')
plt.ylabel(' Early Stopping point - VL Minimum')
plt.tight_layout()
plt.grid()
plt.savefig('../report/img/early_stop_point.pdf')



plt.close()
plt.scatter(x=early['hidden_sizes'], y=early['MEE_min'])


###########################################################
'''
mostrare la differenza tra l'errore minimo e l'errore ottenuto 
con i due early stopping. 

uso il rapporto tra gli errori
'''
early['ratio_PQ'] = early['MEE_PQ']/early['MEE_min']
early['ratio_GL'] = early['MEE_GL']/early['MEE_min']


early_filtered = early.query('ratio_PQ<1.04')
early_filtered = early

plt.close()
plt.figure(figsize=(7,5))
plt.scatter('hidden_sizes', y='ratio_PQ',
            data=early_filtered, label='PQ', s=marker_size)
plt.scatter('hidden_sizes', y='ratio_GL', data=early, label='GL',
            s=marker_size, marker='s')
plt.ylabel('MEE early stopping / MEE min')
plt.xlabel('Number of hidden units')
plt.legend()
plt.grid()
plt.axhline(1)
plt.tight_layout()
plt.savefig('../report/img/early_stop.pdf')
