import skfda
import numpy as np
import pandas as pd
import NOxBenchmark as NOx
import csv
import sys
sys.path.append("../..")
import TFunHDDC.tfunHDDC as tfun
import TFunHDDC.triangleSimulation as tri

noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50

data = []
mats = {'t': [], 'tw': []}
with open('functionTests/initmypcaTests/t1.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

#print(data)
#print(data[1][0:3])
for i in data[1:]:
    mats['t'].append(i[0:3])
t = np.array(mats['t']).astype(np.longdouble)

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

#print(W)
n = np.sum(t, axis=0)
data = NOx.fitNOxBenchmark().data
p = data.coefficients.shape[1]
K = len(t[0])
ev = np.repeat(0., K*p).reshape((K,p))
nux = np.repeat(df_start, K)

for i in range(K):
    
    donnees = _T_initmypca_fd1(data, Wlist, t[:,i])

    ev[i] = donnees["valeurs_propres"]


initx = _T_funhddt_init(data, Wlist, K, t, nux, "AKJBKQKDK", threshold, "cattell", noise, None, d_max, d_set)
print(initx["Q1"])