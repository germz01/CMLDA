import numpy as np
import pandas as pd

tab_columns = ['Optimizer', 'sigma_1', 'sigma_2', 'rho', 'eta',
               'alpha', 'lambda', 'MSE (TR - TS)']
data = '../data/final_setup/analytics/CUP/'

paths = []

for opt in ['sgd', 'cgd']:
    methods = ['standard', 'nesterov'] if opt == 'sgd' else ['pr', 'hs', 'mhs']

    for m in methods:
        p = '{}_{}_CUP_statistics.csv'.format(opt, m)

        paths.append(data + opt.upper() + '/' + m + '/' + p)

datasets = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
            'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
            'mhs': pd.read_csv(paths[4])}

table = pd.DataFrame(columns=tab_columns)

for opt in ['cm', 'nag', 'pr', 'hs', 'mhs']:
    opt_name, opt_type = '', ''
    if opt in ['cm', 'nag']:
        opt_name, opt_type = 'SGD ({})'.format(opt.upper()), 'SGD'
    else:
        opt_name, opt_type = 'CGD ({})'.format(opt.upper()), 'CGD'

    sigma_1 = 1e-4 if opt in ['pr', 'hs', 'mhs'] else np.nan
    sigma_2 = 0.9 if opt in ['pr', 'hs', 'mhs'] else np.nan
    rho = 0.5 if opt in ['pr', 'hs', 'mhs'] else np.nan
    eta = 0.1 if opt in ['cm', 'nag'] else np.nan
    alpha = 0.9 if opt in ['cm', 'nag'] else np.nan
    lamb = 0.01 if opt in ['cm', 'nag'] else np.nan

    mse = '{:.2e} - {:.2e}'.format(datasets[opt].iloc[0, 1],
                                   datasets[opt].iloc[0, 3])

    table.loc[table.shape[0]] = [opt_name, sigma_1, sigma_2, rho, eta, alpha,
                                 lamb, mse]

table_name = 'table.txt'

table.to_latex(buf=data + table_name, index=False, na_rep='-',
               index_names=False,
               column_format=(('| c ' * len(table.columns)) + '|'))
