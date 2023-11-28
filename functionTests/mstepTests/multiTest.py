import skfda
import numpy as np
import pandas as pd
import csv
import sys
sys.path.append("../../..")
import TFunHDDC.tfunHDDC as tfun
import TFunHDDC.triangleSimulation as tri
import TFunHDDC.W_multigen as multi

noise = 1.e-8
graph = False
threshold = 0.1
K=3
d_set = np.repeat(1, K)
d_max = 100
df_start = 50
methods = ['cattell', 'bic', 'grid']
models = ['AKJBKQKDK', 'AKBQKDK', 'ABQKDK']
dfconstr = ['yes', 'no']
dfupdate = ['approx', 'numeric']

data = tri.genTriangles()['data']
vec = np.repeat(2., data['0'].coefficients.shape[0])
vec[0:int(len(vec)/3)] = 0
vec[int(len(vec)/3):int(2*len(vec)/3)] = 1
'''
Wlist = multi.W_multigen(data)

'''
readin = []
mats = {'W': [], 'W_m': []}
with open('triW.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        readin.append(row[0].split(',')[1:])

for i in readin[1:]:
    mats['W'].append(i[0:50])
    mats['W_m'].append(i[50:100])
    mats['dety'] = i[100]
W = np.array(mats['W']).astype(np.longdouble)
W_m = np.array(mats['W_m']).astype(np.longdouble)
dety = np.longdouble(mats["dety"])

Wlist = {'W':W, 'W_m': W_m, 'dety':dety}

t = np.zeros((data['0'].coefficients.shape[0], K))

for i in range(K):
    t[np.nonzero(vec == i)[0], i] = 1

#test grid, cattell, and bic here

n = np.sum(t, axis=0)
p = data['0'].coefficients.shape[1]
nux = np.repeat(df_start, K)
for model in models:
    for method in methods:
        for constr in dfconstr:
            for update in dfupdate:
                print(f'\n####### {model} # {method} # {constr} # {update} #######')
                res = tfun._T_funhddt_init(data, Wlist, K, t, nux, model, threshold, method, noise, None, d_max, d_set)
                tw = tfun._T_funhddt_twinit(data, Wlist, res, nux)
                print(tfun._T_funhddt_m_step1(data, Wlist, K, t, tw, nux, update, constr, model, threshold, method, noise, None, d_max, d_set))
