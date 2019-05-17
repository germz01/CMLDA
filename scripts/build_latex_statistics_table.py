import ipdb
import numpy as np
import pandas as pd

epochs = raw_input('MAX EPOCHS[N/None]: ')
epochs = int(epochs) if epochs != 'None' else None

tab_columns = ['Task', 'Optimizer', 'Convergence Epoch', 'Elapsed Time',
               'LS Iterations', 'BP Time', 'LS Time', 'Dir Time']

path_to_json = '../data/final_setup/'
data = '../data/monks/'

paths = []

for opt in ['sgd', 'cgd']:
    if opt == 'sgd':
        for m in ['standard', 'nesterov']:
            p = '{}_{}_monks_time_statistics.csv'.format(opt, m)

            if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))

            paths.append(data + p)
    else:
        for b in ['pr', 'hs', 'mhs']:
            p = '{}_{}_monks_time_statistics.csv'.format(opt, b)

            if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))

            paths.append(data + p)

datasets_time = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
                 'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
                 'mhs': pd.read_csv(paths[4])}

paths = []

for opt in ['sgd', 'cgd']:
    if opt == 'sgd':
        for m in ['standard', 'nesterov']:
            p = '{}_{}_monks_statistics.csv'.format(opt, m)

            if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))

            paths.append(data + p)
    else:
        for b in ['pr', 'hs', 'mhs']:
            p = '{}_{}_monks_statistics.csv'.format(opt, b)

            if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))

            paths.append(data + p)


datasets = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
            'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
            'mhs': pd.read_csv(paths[4])}

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
        bp_time = np.nan if opt in ['cm', 'nag'] else \
            np.round(datasets_time[opt].iloc[monk - 1, 2], 2)
        ls_time = np.nan if opt in ['cm', 'nag'] else \
            np.round(datasets_time[opt].iloc[monk - 1, 3], 2)
        dir_time = np.nan if opt in ['cm', 'nag'] else \
            np.round(datasets_time[opt].iloc[monk - 1, 4], 2)

        table.loc[table.shape[0]] = \
            ['MONK {}'.format(monk), opt_name, conv_epoch, elapsed_time,
             ls_iterations, bp_time, ls_time, dir_time]

table_name = 'table_time.txt' if epochs is None \
    else 'table_time_max_epochs_{}.txt'.format(epochs)

table.to_latex(buf=data + table_name, index=False, na_rep='-',
               index_names=False,
               column_format=(('| c ' * len(table.columns)) + '|'))

