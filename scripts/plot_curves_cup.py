import json
import utils as u
import warnings

warnings.filterwarnings("ignore")

analytics = raw_input("ANALYTICS or FINAL TEST[a/f]?  ")

path_to_json = '../data/final_setup/analytics/CUP/' if analytics == 'a'  \
                else '../data/final_setup/'

fpath = '../data/CUP/'

betas, momentum = ['hs', 'mhs', 'pr'], ['nesterov', 'standard']
all_methods = ['nesterov', 'standard', 'hs', 'mhs', 'pr']


hpsn = path_to_json + \
    'SGD/{}/CUP_curves_sgd.json'.format(momentum[0])
hpss = path_to_json + \
    'SGD/{}/CUP_curves_sgd.json'.format(momentum[1])
hpsh = path_to_json + \
    'CGD/{}/CUP_curves_cgd.json'.format(betas[0].upper())
hpsm = path_to_json + \
    'CGD/{}/CUP_curves_cgd.json'.format(betas[1].upper())
hpsp = path_to_json + \
    'CGD/{}/CUP_curves_cgd.json'.format(betas[2].upper())

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

time_n = SGD_nesterov['time']
time_s = SGD_standard['time']
time_h = CGD_hs['time']
time_m = CGD_mhs['time']
time_p = CGD_pr['time']

errors_n_va = SGD_nesterov['error_va']
errors_s_va = SGD_standard['error_va']
errors_h_va = CGD_hs['error_va']
errors_m_va = CGD_mhs['error_va']
errors_p_va = CGD_pr['error_va']

# norm_n = SGD_nesterov['gradient_norm']
# norm_s = SGD_standard['gradient_norm']
# norm_h = CGD_hs['gradient_norm']
# norm_m = CGD_mhs['gradient_norm']
# norm_p = CGD_pr['gradient_norm']


if analytics == 'a':
    fname = '../report/img/analytics/CUP/'
else:
    fname = '../report/img/comparisons/'

u.plot_all_learning_curves('CUP', all_methods, [[errors_n, errors_s,
                           errors_h, errors_m, errors_p],
                           [time_n, time_s, time_h, time_m, time_p]],
                           'ERRORS', 'MSE', type='all', time=True,
                           fname=fname)

# u.plot_all_learning_curves(ds + 1, all_methods, [[norm_n, norm_s, norm_h,
#                                                   norm_m, norm_p],
#                            [time_n, time_s, time_h, time_m, time_p]],
#                            'NORM', 'NORM', type='all', time=True,
#                            fname=fname)

u.plot_all_learning_curves('CUP', momentum, [[errors_n, errors_s]],
                           'ERRORS', 'MSE', type='momentum',
                           fname=fname)

u.plot_all_learning_curves('CUP', momentum, [[errors_n, errors_s],
                            [time_n, time_s]],
                           'ERRORS', 'MSE', type='momentum', time=True,
                           fname=fname)


# u.plot_all_learning_curves(ds + 1, momentum, [[norm_n, norm_s]],
#                            'NORM', 'NORM', type='momentum',
#                            fname=fname)

u.plot_all_learning_curves('CUP', betas,
                           [[errors_h, errors_m, errors_p]],
                           'ERRORS', 'MSE', type='beta',
                           fname=fname)

u.plot_all_learning_curves('CUP', betas,
                           [[errors_h, errors_m, errors_p], [time_h, time_m, time_p]],
                           'ERRORS', 'MSE', type='beta',  time=True,
                           fname=fname)

u.plot_all_learning_curves('CUP', ['training', 'validation'],
                           [[errors_m, errors_m_va]],
                           'ERRORS', 'MSE', type='mhs',
                           fname=fname)

u.plot_all_learning_curves('CUP', ['training', 'validation'],
                           [[errors_p, errors_p_va]],
                           'ERRORS', 'MSE', type='pr',
                           fname=fname)

u.plot_all_learning_curves('CUP', ['training', 'validation'],
                           [[errors_h, errors_h_va]],
                           'ERRORS', 'MSE', type='hs',
                           fname=fname)


u.plot_all_learning_curves('CUP', ['training', 'validation'],
                           [[errors_s, errors_s_va]],
                           'ERRORS', 'MSE', type='std',
                           fname=fname)

u.plot_all_learning_curves('CUP', ['training', 'validation'],
                           [[errors_n, errors_n_va]],
                           'ERRORS', 'MSE', type='nesterov',
                           fname=fname)

u.plot_all_learning_curves('CUP', all_methods, [[errors_n, errors_s,
                           errors_h, errors_m, errors_p]],
                           'ERRORS', 'MSE', type='all',
                           fname=fname)

# u.plot_all_learning_curves(ds + 1, all_methods, [[norm_n, norm_s, norm_h,
#                                                   norm_m, norm_p]],
#                            'NORM', 'NORM', type='all', time=False,
#                            fname=fname)

