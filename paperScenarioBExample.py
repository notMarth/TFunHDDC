import sys
sys.path.append('../')
import tfunHDDC as tfun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

'''
This runs the examples from the paper for which TFunHDDC is based on.

300 curves are simulated with 35 splines for the basis functions. There are 3
scenarios that will be run: scenario B runs with eta = 5, 50, 15 for groups 1,
2, and 3.

For each scenario, the ARI is printed,
alongside the table of classified data against its true label.


'''


if __name__ == '__main__':
    print("Scenario B")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[5,50,15])
    labels = data['labels']

    sim.plotModelFD(data)

    res = tfun.tfunHDDC(data['data'], min_individuals=4, model='all', K=3, threshold=0.05, nb_rep=5, init='kmeans')
    print(met.confusion_matrix(res.cl, labels))          
    print(tfun._T_hddc_ari(labels, res.cl))

