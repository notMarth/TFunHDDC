import sys
sys.path.append('.')
import tfunHDDC as tfun
import numpy as np
import modelSimulation as sim
from sklearn import metrics as met

if __name__ == '__main__':
    models = ['akjbkqkdk', 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk']
    print("Sim Data Clustering:")
    data = sim.genModelFD(ncurves=300, nsplines=35, alpha=[0.9,0.9, 0.9], eta=[10,7,17])
    training = np.concatenate((np.arange(0, 50), np.arange(100, 150), np.arange(200, 250))).astype(int)
    test = np.concatenate((np.arange(50,100), np.arange(150, 200), np.arange(250, 300))).astype(int)
    clm = data['labels']
    known = clm[training].astype(int)
    labels = data['labels']

    results = {}
    for i in models:
        res = tfun.tfunHDDC(data['data'], min_individuals=4, model=i, K=3, threshold=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], nb_rep=20, init='kmeans')
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, known)))/len(known)]
        results[i].append(met.confusion_matrix(res.cl, known))        
        results[i].append(tfun._T_hddc_ari(known, res.cl))

        
    for i in results:
        print(f'{i} {results[i]}') 

    print('\nSim Data Classification and Prediction:')

    results = {}
    for i in models:
        res = tfun.tfunHDDC(data['data'][training], min_individuals=4, model=i, K=3, threshold=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], nb_rep=20, init='kmeans', known = known)
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, known)))/len(known)]
        results[i].append(met.confusion_matrix(res.cl, known))
        
        results[i].append(tfun._T_hddc_ari(known, res.cl))

        pred = res.predict(data['data'][test])
        results[i].append(np.sum(np.diag(met.confusion_matrix(pred['class'], clm[test])))/len(clm[test]))
        results[i].append(met.confusion_matrix(pred['class'], clm[test]))
        results[i].append(tfun._T_hddc_ari(clm[test], pred['class']))

    for i in results:
        print(f'{i} {results[i]}') 

    