#######################################################################
#init=kmeans
#known
#dfupdate = numeric
#dfconstr = cattell
#model = AKJBKQKDK
#######################################################################

import numpy as np
import skfda
import sys
sys.path.append('../../..')
import TFunHDDC.tfunHDDC as tfun
import TFunHDDC.NOxBenchmark as NOx
import traceback
import csv
import matplotlib.pyplot as plt
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
with open('../initmypcaTests/W.csv', newline='') as csvfile:
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

noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50.
itermax = 200
itTest=50

kmeans_control = {'max_iter': 10, 'n_init': 1, 'algorithm': 'lloyd'}
known = np.repeat(np.NaN, len(datam['target']))
known[np.arange(len(datam['target'])) % 3 == 0] = datam['target'][np.arange(len(datam['target'])) % 3 == 0]
stats1 = {'bic':np.zeros(itTest), 'icl': np.zeros(itTest), 'time': 0., 'cl': {'0':np.zeros(len(datam['target'])), '1':np.zeros(len(datam['target'])), '2':np.zeros(len(datam['target']))}, 'converged': 0}
stats2 = {'bic':np.zeros(itTest), 'icl': np.zeros(itTest), 'time': 0., 'cl': {'0':np.zeros(len(datam['target'])), '1':np.zeros(len(datam['target'])), '2':np.zeros(len(datam['target']))}, 'converged': 0}

j=0
while j < itTest:
    print(j)
    try:
        startTime = time.process_time()

        res = tfun._T_funhddc_main1(fd, Wlist, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'kmeans', None, None, 2, noise, None, kmeans_control, d_max, d_set, None)
            
        if not isinstance(res, tfun.TFunHDDC):
            continue

        print(res.cl)
        stats1['bic'][j] = res.bic
        stats1['icl'][j] = res.icl
        stats1['time'] += (time.process_time() - startTime)
        
        for i, cl in enumerate(res.cl):
            stats1['cl'][f'{cl}'][i] += 1

        if res.converged:
            stats1['converged'] += 1
        j += 1
    except Exception as e:
        continue

stats1['time'] = stats1['time']/float(itTest)

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

j=0
while(j < itTest):
    print(j)
    try:
        startTime=time.process_time()
        res = tfun._T_funhddc_main1(fd, Wlist, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'kmeans', None, None, 2, noise, None, kmeans_control, d_max, d_set, known)
            
        if not isinstance(res, tfun.TFunHDDC):
            continue
        print(res.cl)

        stats2['bic'][j] = res.bic
        stats2['icl'][j] = res.icl
        stats2['time'] += (time.process_time() - startTime)
        for i, cl in enumerate(res.cl):
            stats2['cl'][f'{cl}'][i] += 1
        if res.converged:
            stats2['converged'] += 1

        j+=1
    except Exception as e:
        continue

stats2['time'] = stats2['time']/float(itTest)

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

fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(stats1['bic'])
axs[0,0].set_title('BIC for Kmeans init without Known')

axs[0, 1].boxplot(stats1['icl'])
axs[0,1].set_title('ICL for Kmeans init without Known')

axs[1,0].boxplot(stats2['bic'])
axs[1,0].set_title('BIC for Kmeans init with Known')

axs[1,1].boxplot(stats2['icl'])
axs[1,1].set_title('ICL for Kmeans init with Known')

fig, ax = plt.subplots()
ax.boxplot([stats1['bic'], stats1['icl'], stats2['bic'], stats2['icl']])

plt.show()