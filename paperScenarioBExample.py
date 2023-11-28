import sys
sys.path.append('../')
import tfunHDDC as tfun
import funHDDC as fun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

'''
This runs the examples from the paper for which TFunHDDC is based on.

300 curves are simulated with 35 splines for the basis functions. Scenario B 
runs with eta = 5, 50, 15 for groups 1, 2, and 3.

TFunHDDC is run first, then funHDDC. The ARI and the table for the the clustered
data are printed.

'''


if __name__ == '__main__':
    print("Scenario B")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[5,50,15])
    labels = data['labels']

    sim.plotModelFD(data)

    res = tfun.tfunHDDC(data['data'], model='all', K=3, threshold=0.2, nb_rep=5, init='kmeans')
    res1 = fun.funHDDC(data['data'], model='all', K=3, threshold=0.2, nb_rep=5, init='kmeans')

    print("Results for TFunHDDC:")
    print(met.confusion_matrix(res.cl, labels))          
    print(tfun._T_hddc_ari(labels, res.cl))

    print("\nResults for funHDDC:")
    print(met.confusion_matrix(res1.cl, labels))          
    print(tfun._T_hddc_ari(labels, res1.cl))

