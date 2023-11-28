#######################################################################
#init=mini_em
#dfupdate = numeric
#dfconstr = cattell
#model = AKJBKQKDK
#######################################################################

import numpy as np
import skfda
import tfunHDDC as tfun
import NOxBenchmark as NOx
import traceback
import csv

datam = NOx.fitNOxBenchmark()['data']
'''
W = skfda.misc.inner_product_matrix(data.basis, data.basis)
W[W < 1.e-15] = 0
W_m = np.linalg.cholesky(W)
dety = np.linalg.det(W)
W_list = {'W': W, 'W_m': W_m, 'dety':dety}
'''
data = []
mats = {'W': [], 'W_m': []}
with open('functionTests/initmypcaTests/W.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

for i in data[1:]:
    mats['W'].append(i[0:15])
    mats['W_m'].append(i[15:30])
    mats['dety'] = i[30]
W = np.array(mats['W']).astype(np.longdouble)
W_m = np.array(mats['W_m']).astype(np.longdouble)
dety = np.longdouble(mats["dety"])

Wlist = {'W':W, 'W_m':W_m, "dety":dety}
noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50.
itermax = 200
mini_nb = [5, 10]

try:
    res = tfun._T_funhddc_main1(datam, Wlist, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'mini-em', None, mini_nb, 0, noise, None, None, d_max, d_set, None)
    print(res.a)
    print(res.b)
    print(res.nux)
    print(res.converged)
except Exception as e:
    traceback.print_exc()