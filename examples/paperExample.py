import sys
sys.path.append('.')
import tfunHDDC as tfun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

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
