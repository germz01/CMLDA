import ipdb
import numpy as np
import pandas as pd

epochs = raw_input('MAX EPOCHS[N/None]: ')
epochs = int(epochs) if epochs != 'None' else None

tab_columns = ['Task', 'Optimizer', 'Convergence Epoch', 'LS Iterations',
               'Elapsed Time', 'BP Time', 'LS Time', 'Dir Time']

path_to_json = '../data/final_setup/'
from_analytics = True if raw_input('RETRIEVE FROM ANALYTICS[Y/N]? ') == 'Y' \
    else False
data = '../data/final_setup/analytics/' if from_analytics else '../data/monks/'

paths = []

for csv in ['_', '_time_']:
    for opt in ['sgd', 'cgd']:
        methods = ['standard', 'nesterov'] if opt == 'sgd' else \
            ['pr', 'hs', 'mhs']

        for m in methods:
            p = '{}_{}_monks{}statistics.csv'.format(opt, m, csv)

            if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))
            else:
                if from_analytics:
                    p = p.replace('.csv', '_no_max_epochs.csv')

            paths.append(data + opt.upper() + '/' + m + '/' + p if
                         from_analytics else data + p)

datasets = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
            'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
            'mhs': pd.read_csv(paths[4])}

datasets_time = {'cm': pd.read_csv(paths[5]), 'nag': pd.read_csv(paths[6]),
                 'pr': pd.read_csv(paths[7]), 'hs': pd.read_csv(paths[8]),
                 'mhs': pd.read_csv(paths[9])}

table = pd.DataFrame(columns=tab_columns)

for monk in [1, 2, 3]:
    for opt in ['cm', 'nag', 'pr', 'hs', 'mhs']:
        opt_name, opt_type = '', ''
        if opt in ['cm', 'nag']:
            opt_name, opt_type = 'SGD ({})'.format(opt.upper()), 'SGD'
        else:
            opt_name, opt_type = 'CGD ({})'.format(opt.upper()), 'CGD'

        conv_epoch = int(datasets[opt].iloc[monk - 1, 9])
        elapsed_time = np.round(datasets_time[opt].iloc[monk - 1, 1], 2)
        ls_iterations = np.nan if opt in ['cm', 'nag'] else \
            int(np.round(datasets[opt].iloc[monk - 1, 10]))
        bp_time = np.round(datasets_time[opt].iloc[monk - 1, 2], 2)
        ls_time = np.nan if opt in ['cm', 'nag'] else \
            np.round(datasets_time[opt].iloc[monk - 1, 3], 2)
        dir_time = np.nan if opt in ['cm', 'nag'] else \
            np.round(datasets_time[opt].iloc[monk - 1, 4], 2)

        table.loc[table.shape[0]] = \
            ['MONK {}'.format(monk), opt_name, conv_epoch, ls_iterations,
             elapsed_time, bp_time, ls_time, dir_time]

table_name = 'table_time.txt' if epochs is None \
    else 'table_time_max_epochs_{}.txt'.format(epochs)

table.to_latex(buf=data + table_name, index=False, na_rep='-',
               index_names=False,
               column_format=(('| c ' * len(table.columns)) + '|'))

