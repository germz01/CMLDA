import json
import utils as u
import warnings

warnings.filterwarnings("ignore")

time = raw_input("PLOT TIME/EPOCHS? ")
analytics = raw_input("ANALYTICS or FINAL TEST? a/f")

path_to_json = '../data/final_setup/analytics/' if analytics == 'a'  \
                else '../data/final_setup/'

fpath = '../data/monks/'

betas, momentum = ['hs', 'mhs', 'pr'], ['nesterov', 'standard']
all_methods = ['nesterov', 'standard', 'hs', 'mhs', 'pr']

for ds in [0, 1, 2]:

    hpsn = path_to_json + \
        'SGD/{}/MONK{}_curves_sgd.json'.format(momentum[0], ds + 1)
    hpss = path_to_json + \
        'SGD/{}/MONK{}_curves_sgd.json'.format(momentum[1], ds + 1)
    hpsh = path_to_json + \
        'CGD/{}/MONK{}_curves_cgd.json'.format(betas[0].upper(), ds + 1)
    hpsm = path_to_json + \
        'CGD/{}/MONK{}_curves_cgd.json'.format(betas[1].upper(), ds + 1)
    hpsp = path_to_json + \
        'CGD/{}/MONK{}_curves_cgd.json'.format(betas[2].upper(), ds + 1)

    with open(hpsn) as json_file:
        SGD_nesterov = json.load(json_file)
    with open(hpss) as json_file:
        SGD_standard = json.load(json_file)
    with open(hpsh) as json_file:
        CGD_hs = json.load(json_file)
    with open(hpsm) as json_file:
        CGD_mhs = json.load(json_file)
    with open(hpsp) as json_file:
        CGD_pr = json.load(json_file)

    errors_n = SGD_nesterov['error']
    errors_s = SGD_standard['error']
    errors_h = CGD_hs['error']
    errors_m = CGD_mhs['error']
    errors_p = CGD_pr['error']

    acc_n = SGD_nesterov['accuracy']
    acc_s = SGD_standard['accuracy']
    acc_h = CGD_hs['accuracy']
    acc_m = CGD_mhs['accuracy']
    acc_p = CGD_pr['accuracy']

    time_n = SGD_nesterov['time']
    time_s = SGD_standard['time']
    time_h = CGD_hs['time']
    time_m = CGD_mhs['time']
    time_p = CGD_pr['time']

    norm_n = SGD_nesterov['gradient_norm']
    norm_s = SGD_standard['gradient_norm']
    norm_h = CGD_hs['gradient_norm']
    norm_m = CGD_mhs['gradient_norm']
    norm_p = CGD_pr['gradient_norm']

    if time == 'TIME':
        u.plot_all_learning_curves(ds + 1, all_methods, [[errors_n, errors_s,
                                   errors_h, errors_m, errors_p],
                                   [time_n, time_s, time_h, time_m, time_p]],
                                   'TIME', 'MSE', type='all', time=True,
                                   fname='../report/img/analytics/')
    else:

        u.plot_all_learning_curves(ds + 1, momentum, [[errors_n, errors_s]],
                                   'ERRORS', 'MSE', type='momentum',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, momentum, [[acc_n, acc_s]],
                                   'ACCURACY', 'ACCURACY', type='momentum',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, momentum, [[norm_n, norm_s]],
                                   'NORM', 'NORM', type='momentum',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, betas,
                                   [[errors_h, errors_m, errors_p]],
                                   'ERRORS', 'MSE', type='beta',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, betas, [[acc_h, acc_m, acc_p]],
                                   'ACCURACY', 'ACCURACY', type='beta',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, betas, [[norm_h, norm_m, norm_p]],
                                   'NORM', 'NORM', type='beta',
                                   fname='../report/img/analytics/')

        u.plot_all_learning_curves(ds + 1, all_methods, [[errors_n, errors_s,
                                   errors_h, errors_m, errors_p]],
                                   'ERRORS', 'MSE', type='all',
                                   fname='../report/img/analytics/')

