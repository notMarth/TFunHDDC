import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx
import modelSimulation as mod
from sklearn import metrics as met
import csv
import cProfile
import pstats
import re
from pstats import SortKey
import time

if __name__ == '__main__':
    data = NOx.fitNOxBenchmark(15)
    #NOx.plot_NOx(data)
    
    '''
    vec = []
    with open('HWcluster.csv', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            vec.append(row[0].split(',')[1:][0])

    vec = np.array(vec[1:], dtype=int)
    vec = vec - 1
    '''
    labels=data['labels']

    it = 100
    cumtime = 0.
    for i in range(it):
        print(i)
        start = time.process_time()
        res = tfun.tfunHDDC(data['data'], min_individuals=4, model='all', K=2, threshold=0.1, nb_rep=20, init='kmeans', dfconstr='no', itermax=200)
        cumtime += (time.process_time() - start)
        print("CCR")
        print(np.sum(np.diag(met.confusion_matrix(res.cl, labels)))/len(labels))
        print(met.confusion_matrix(res.cl, labels))
        print("ARI")
        print(tfun._T_hddc_ari(labels, res.cl))    

    print(f"\nAverage time per run: {cumtime/it} seconds")
    '''
    print("Sim Data:")
    start = time.time()
    res = tfun.tfunHDDC(data, min_individuals=3, model='all', K=3, threshold=thresh, nb_rep=1, init='kmeans')
    print("CCR")
    print(np.sum(np.diag(met.confusion_matrix(res.cl, labels)))/len(labels))
    print(met.confusion_matrix(res.cl, labels))
    print("ARI")
    print(tfun._T_hddc_ari(labels, res.cl))
    print(f"Time taken: {time.time()- start} seconds")
    '''