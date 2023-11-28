import sys
sys.path.append('.')
import tfunHDDC as tfun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

'''
This runs the examples from the paper for which TFunHDDC is based on.

300 curves are simulated with 35 splines for the basis functions. There are 3
scenarios that will be run: scenario A runs with eta = 10, 7, 17 for groups 1,
2, and 3, scenario B runs with eta = 5, 50, 15 for groups 1, 2, and 3, and 
scenario C runs with eta = 100, 70, 170 for groups 1, 2, and 3.

For each scenario, the CCR and the ARI for each resulting run is printed,
alongside the table of classified data against its true label.

Each model is run seperately as the log likelihood may prefer a clustering that
has a higher CCR but a lower ARI. Thus by running all 6 models separately, we
can check each output.
'''


if __name__ == '__main__':
    models = ['akjbkqkdk', 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk']
    print("Scenario A")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[10,7,17])
    labels = data['labels']

    results = {}
    for i in models:
        res = tfun.tfunHDDC(data['data'], min_individuals=4, model=i, K=3, threshold=[0.05, 0.1, 0.2], nb_rep=50, init='kmeans')
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, labels)))/len(labels)]
        results[i].append(met.confusion_matrix(res.cl, labels))        
        results[i].append(tfun._T_hddc_ari(labels, res.cl))

        
    for i in results:
        print(f'{i} {results[i]}') 

    print("\nScenario B")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[5,50,15])
    labels = data['labels']

    results = {}
    for i in models:
        res = tfun.tfunHDDC(data['data'], min_individuals=4, model=i, K=3, threshold=[0.05, 0.1, 0.2], nb_rep=50, init='kmeans')
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, labels)))/len(labels)]
        results[i].append(met.confusion_matrix(res.cl, labels))        
        results[i].append(tfun._T_hddc_ari(labels, res.cl))
        
    for i in results:
        print(f'{i} {results[i]}') 

    print("\nScenario C")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[100,70,170])
    labels = data['labels']
    results = {}
    for i in models:
        res = tfun.tfunHDDC(data['data'], min_individuals=4, model=i, K=3, threshold=[0.05, 0.1, 0.2], nb_rep=50, init='kmeans')
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, labels)))/len(labels)]
        results[i].append(met.confusion_matrix(res.cl, labels))        
        results[i].append(tfun._T_hddc_ari(labels, res.cl))
        
    for i in results:
        print(f'{i} {results[i]}') 
