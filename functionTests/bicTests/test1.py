#######################################################################
#init=kmeans
#known
#dfupdate = numeric
#dfconstr = cattell
#model = AKJBKQKDK
#######################################################################

import numpy as np
import sys
import skfda
sys.path.append('../../..')
import TFunHDDC.tfunHDDC as tfun
import TFunHDDC.NOxBenchmark as NOx
import traceback
import scipy.linalg as scil
import csv
import time

datam = NOx.fitNOxBenchmark()
'''
W = skfda.misc.inner_product(datam['data'].basis, datam['data'].basis, _matrix = True)
W[W < 1.e-15] = 0
W_m = scil.cholesky(W)
dety = np.linalg.det(W)
W_list = {'W': W, 'W_m': W_m, 'dety':dety}
'''
data = []
mats = {'W': [], 'W_m': []}
with open('W.csv', newline='') as csvfile:
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

fd = datam['data']

#params = {'model':'AKJBKQKDK','a':[1], 'b':[0,1], 'd':np.array([1,2,3]), 'N':3,'loglik':5., 'posterior':np.array([[1.,2,2], [2,2,2], [2,2,2]]), 'K':3, 'prop':0}
#print(tfun._T_hdclassift_bic(params, 3, 'no'))
noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50.
itermax = 200
kmeans_control = {'max_iter': 10, 'n_init': 1, 'algorithm': 'lloyd'}
vec = np.repeat(2, len(datam['target']))
vec[0:50] = 0
vec[50:100] = 1
known = np.repeat(np.NaN, len(datam['target']))
known[np.arange(len(datam['target'])) % 3 == 0] = datam['target'][np.arange(len(datam['target'])) % 3 == 0]
stats1 = {'bic':0, 'icl': 0, 'cl': {'0':np.zeros(len(datam['target'])), '1':np.zeros(len(datam['target'])), '2':np.zeros(len(datam['target']))}, 'converged': 0}
stats2 = {'bic':0, 'icl': 0, 'cl': {'0':np.zeros(len(datam['target'])), '1':np.zeros(len(datam['target'])), '2':np.zeros(len(datam['target']))}, 'converged': 0}

itTest=1
j=0
while j < itTest:
    print(j)
    try:
        start_time = time.process_time()
        res = tfun._T_funhddc_main1(fd, Wlist, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'vector',vec, None, 0, noise, None, kmeans_control, d_max, d_set, None)
        print(time.process_time() - start_time, 'seconds')
        if not isinstance(res, tfun.TFunHDDC):
            print(res)
            continue

        print(res.cl)
        stats1['bic'] += res.bic
        stats1['icl'] += res.icl
        
        for i, cl in enumerate(res.cl):
            stats1['cl'][f'{cl}'][i] += 1

        if res.converged:
            stats1['converged'] += 1
        j += 1
    except Exception as e:
        traceback.format_exc()
        raise e

print(stats1['bic'])
print(stats1['icl'])
stats1['bic'] = stats1['bic']/100.
stats1['icl'] = stats1['icl']/100.
'''
filename = 'KmeansNumericCattellYes.csv'
with open(filename, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    for r in stats1.items():
        if r[0] == 'cl':
            datawriter.writerow(r[0])
            for e in range(len(r[1])):
                datawriter.writerow(r[1][f'{e}'])
        else:
            datawriter.writerow(r)
'''

vec[np.arange(len(datam['target'])) % 3 == 0] = known[np.arange(len(datam['target'])) % 3 == 0]

j=0
while(j < itTest):
    print(j)
    try:
        start_time = time.process_time()
        res = tfun._T_funhddc_main1(fd, Wlist, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'vector', vec, None, 0, noise, None, kmeans_control, d_max, d_set, known)
        print(time.process_time()-start_time, 'seconds')
        if not isinstance(res, tfun.TFunHDDC):
            continue
        print(res.cl)

        stats2['bic'] += res.bic
        stats2['icl'] += res.icl
        for i, cl in enumerate(res.cl):
            stats2['cl'][f'{cl}'][i] += 1
        if res.converged:
            stats2['converged'] += 1

        j+=1
    except Exception as e:
        traceback.format_exc()
        raise e
print(stats2['bic'])
print(stats2['icl'])
stats2['bic'] = stats2['bic']/100.
stats2['icl'] = stats2['icl']/100.
'''
filename = 'KmeansNumericCattellYesKnown.csv'
with open(filename, 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    for r in stats2.items():
        if r[0] == 'cl':
            datawriter.writerow(r[0])
            for e in range(len(r[1])):
                datawriter.writerow(r[1][f'{e}'])
        else:
            datawriter.writerow(r)
'''