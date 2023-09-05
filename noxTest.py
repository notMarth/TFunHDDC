import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx
from sklearn import metrics as met
import csv

if __name__ == '__main__':
    data = NOx.fitNOxBenchmark()
    #NOx.plot_NOx(data)

    vec = []
    '''
    with open('HWcluster.csv', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            vec.append(row[0].split(',')[1:][0])
 
    vec = np.array(vec[1:], dtype=int)
    vec = vec - 1
    print(vec)
    '''

    res = tfun.tfunHDDC(data['data'], min_individuals=3, model='all', K=2, threshold=[0.05, 0.2, 0.4, 0.6], nb_rep=20, init='kmeans')
    print(np.sum(np.diag(met.confusion_matrix(res.cl, data['labels'])))/len(data['labels']))
    print(tfun._T_hddc_ari(data['labels'], res.cl))