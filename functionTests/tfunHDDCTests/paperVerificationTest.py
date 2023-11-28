import sys
sys.path.append('../..')
import time
import numpy as np
import tfunHDDC as tfun
import NOxBenchmark as NOx
import matplotlib.pyplot as plt
import csv
import sklearn
from copy import *

if __name__ == '__main__':
    K = 2
    coreNum = 16
    inits=['kmeans']
    data = NOx.fitNOxBenchmark()['data']
    labels = NOx.fitNOxBenchmark()['target'].astype(int)
    vec = np.repeat(1, len(data.coefficients))
    vec[0:int(len(data.coefficients)/2)] = 0
    kmeans_control1 = {'max_iter': 20, 'n_init': 1, 'algorithm': 'lloyd'}
    kmeans_control2 = {'max_iter':20, 'n_init': 1, 'algorithm': 'elkan'}
    kmeans_control = [kmeans_control1, kmeans_control2]
    itTest = 1

    cl = {}
    for i in range(K):
        cl[f'{i}'] = np.zeros(len(data.coefficients))

    statsrand = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari': np.zeros(itTest), 'ariStdev':0}
    statsvec = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    statsmini = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    statsk = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    
    for i in inits:
        
        if i == 'kmeans':
            for k in kmeans_control:
                res = tfun.tfunHDDC(data, model=["AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"], K=K, mc_cores=coreNum, threshold=0.6, nb_rep = 20, init=i, init_vector=vec, kmeans_control=k)
                if not isinstance(res, tfun.TFunHDDC):
                    print(res)
                    continue
                print(f"CCR for init={i}, 'algo:{k['algorithm']}")
                print(np.sum(np.diag(sklearn.metrics.confusion_matrix(labels, res.cl, labels=[k for k in range(K)])))/len(data.coefficients))
            
        else:
            res = tfun.tfunHDDC(data, model=["AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"], K=K, mc_cores=coreNum, threshold=0.6, nb_rep = 20, init=i, init_vector=vec)
            if not isinstance(res, tfun.TFunHDDC):
                print(res)
                continue
            print(f"CCR for init={i}")
            print(np.sum(np.diag(sklearn.metrics.confusion_matrix(labels, res.cl, labels=[k for k in range(K)])))/len(data.coefficients))