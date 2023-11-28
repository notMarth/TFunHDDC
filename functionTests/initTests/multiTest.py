import skfda
import numpy as np
import pandas as pd
import NOxBenchmark as NOx
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

data = tri.genTriangles()['data']
vec = np.repeat(2., data['0'].coefficients.shape[0])
vec[0:int(len(vec)/3)] = 0
vec[int(len(vec)/3):int(2*len(vec)/3)] = 1

Wlist = multi.W_multigen(data)

t = np.zeros((data['0'].coefficients.shape[0], K))

for i in range(K):
    t[np.nonzero(vec == i)[0], i] = 1

#test grid, cattell, and bic here

n = np.sum(t, axis=0)
p = data['0'].coefficients.shape[1]
nux = np.repeat(df_start, K)

for model in models:
    for method in methods:
        print(f'\n####### {model} # {method} #######')
        print(tfun._T_funhddt_init(data, Wlist, K, t, nux, model, threshold, method, noise, None, d_max, d_set)['a'])
