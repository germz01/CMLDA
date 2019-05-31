import numpy as np
import pandas as pd

tab_columns = ['Optimizer', 'Convergence Epoch', 'LS Iterations',
               'Elapsed Time', 'BP Time', 'LS Time', 'Dir Time']

data = '../data/final_setup/analytics/CUP/'

paths = []

for csv in ['_', '_time_']:
    for opt in ['sgd', 'cgd']:
        methods = ['standard', 'nesterov'] if opt == 'sgd' else \
            ['pr', 'hs', 'mhs']

        for m in methods:
            p = '{}_{}_CUP{}statistics.csv'.format(opt, m, csv)

            paths.append(data + opt.upper() + '/' + m + '/' + p)

datasets = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
            'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
            'mhs': pd.read_csv(paths[4])}

datasets_time = {'cm': pd.read_csv(paths[5]), 'nag': pd.read_csv(paths[6]),
                 'pr': pd.read_csv(paths[7]), 'hs': pd.read_csv(paths[8]),
                 'mhs': pd.read_csv(paths[9])}

table = pd.DataFrame(columns=tab_columns)

for opt in ['cm', 'nag', 'pr', 'hs', 'mhs']:
    opt_name, opt_type = '', ''
    if opt in ['cm', 'nag']:
        opt_name, opt_type = 'SGD ({})'.format(opt.upper()), 'SGD'
    else:
        opt_name, opt_type = 'CGD ({})'.format(opt.upper()), 'CGD'

    conv_epoch = int(datasets[opt].iloc[0, 5])
    elapsed_time = np.round(datasets_time[opt].iloc[0, 1], 2)
    ls_iterations = np.nan if opt in ['cm', 'nag'] else \
        int(np.round(datasets[opt].iloc[0, 6]))
    bp_time = np.round(datasets_time[opt].iloc[0, 2], 2)
    ls_time = np.nan if opt in ['cm', 'nag'] else \
        np.round(datasets_time[opt].iloc[0, 3], 2)
    dir_time = np.nan if opt in ['cm', 'nag'] else \
        np.round(datasets_time[opt].iloc[0, 4], 2)

    table.loc[table.shape[0]] = [opt_name, conv_epoch, ls_iterations,
                                 elapsed_time, bp_time, ls_time, dir_time]

table_name = 'table_time.txt'

table.to_latex(buf=data + table_name, index=False, na_rep='-',
               index_names=False,
               column_format=(('| c ' * len(table.columns)) + '|'))
