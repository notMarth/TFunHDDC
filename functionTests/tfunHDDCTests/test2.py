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
    inits = ['random', 'vector', 'mini-em', 'kmeans']
    data = NOx.fitNOxBenchmark()['data']
    labels = NOx.fitNOxBenchmark()['target'].astype(int)
    vec = np.repeat(1, len(data.coefficients))
    vec[0:int(len(data.coefficients)/2)] = 0
    kmeans_control = {'max_iter': 10, 'n_init': 1, 'algorithm': 'lloyd'}
    itTest = 1

    cl = {}
    for i in range(K):
        cl[f'{i}'] = np.zeros(len(data.coefficients))

    statsrand = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari': np.zeros(itTest), 'ariStdev':0}
    statsvec = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    statsmini = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    statsk = {'time': 0, 'cl': deepcopy(cl), 'ccr':np.zeros(itTest), 'ccrStdev':0, 'ari':np.zeros(itTest), 'ariStdev':0}
    
    for i in inits:
        j=0    

        if i == 'random':
            stats = deepcopy(statsrand)
        if i == 'vector':
            stats = deepcopy(statsvec)
        if i == 'mini-em':
            stats = deepcopy(statsmini)
        if i == 'kmeans':
            stats = deepcopy(statsk)
        
        print(i)
        while j < itTest:
            print(j)
            start = time.time()
            try:
                res = tfun.tfunHDDC(data, model=["AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"], K=K, mc_cores=coreNum, threshold=0.6, nb_rep = 20, init=i, init_vector=vec, kmeans_control=kmeans_control)
                if not isinstance(res, tfun.TFunHDDC):
                    print(res)
                    continue

            except Exception as e:
                raise e

            
            stats['time'] += time.time() - start
            for k, cl in enumerate(res.cl):
                stats['cl'][f'{cl}'][k] += 1


            stats['ccr'][j] = np.sum(np.diag(sklearn.metrics.confusion_matrix(labels, res.cl, labels=[k for k in range(K)])))/len(data.coefficients)
            stats['ari'][j] = tfun._T_hddc_ari(res.cl, labels)
            
            j+=1   

        if i == 'random':
            statsrand = stats
        if i == 'vector':
            statsvec = stats
        if i == 'mini-em':
            statsmini = stats
        if i == 'kmeans':
            statsk = stats

    for i in inits:
        if i == 'random':
            stats = deepcopy(statsrand)

        if i == 'vector':
            stats = deepcopy(statsvec)

        if i == 'mini-em':
            stats = deepcopy(statsmini)

        if i == 'kmeans':
            stats = deepcopy(statsk)

        stats['time'] = stats['time']/float(itTest)
        stats['ariStdev'] = np.std(stats['ari'])
        stats['ari'] = np.average(stats['ari'])
        stats['ccrStdev'] = np.std(stats['ccr'])
        stats['ccr'] = np.average(stats['ccr'])
        filename = f'py{i}.csv'
        with open(filename, 'w', newline='') as csvfile:
            datawriter = csv.writer(csvfile)
            for r in stats.items():
                if r[0] == 'cl':
                    datawriter.writerow(r[0])
                    for e in range(len(r[1])):
                        datawriter.writerow(r[1][f'{e}'])
                else:
                    datawriter.writerow(r)

    fig, ax = plt.subplots()
    ax.boxplot([statsrand['ari'], statsvec['ari'], statsmini['ari'], statsk['ari']])
    plt.show()
