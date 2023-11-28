import sys
sys.path.append('../')
import tfunHDDC as tfun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

'''
This runs the examples from the paper for which TFunHDDC is based on.

300 curves are simulated with 35 splines for the basis functions. There are 3
scenarios that will be run: scenario A runs with eta = 10, 7, 17 for groups 1,
2, and 3.

For each scenario, the CCR and the ARI for each resulting run is printed,
alongside the table of classified data against its true label.


'''


if __name__ == '__main__':
    print("Scenario A")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[10,7,17])
    labels = data['labels']

    sim.plotModelFD(data)

    res = tfun.tfunHDDC(data['data'], min_individuals=4, model='all', K=3, threshold=0.2, nb_rep=50, init='kmeans')
    print(met.confusion_matrix(res.cl, labels))          
    print(tfun._T_hddc_ari(labels, res.cl))

