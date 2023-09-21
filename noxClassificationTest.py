import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx
from sklearn import metrics as met
import time

if __name__ == '__main__':
    models = ['akjbkqkdk', 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk']
    data = NOx.fitNOxBenchmark()
    training = np.arange(0,50)
    test = np.arange(50,100)
    known = data['labels'][training]
    labels = data['labels']

    results = {}
    for i in models:
        start = time.time()
        res = tfun.tfunHDDC(data['data'][training], min_individuals=4, model=i, K=2, threshold=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], nb_rep=20, init='kmeans', known = known, mc_cores=16)
        results[i] = [np.sum(np.diag(met.confusion_matrix(res.cl, known)))/len(known)]
        results[i].append(met.confusion_matrix(res.cl, known))
        
        results[i].append(tfun._T_hddc_ari(known, res.cl))
        results[i].append(f"Time taken: {time.time() - start} seconds")

        pred = res.predict(data['data'][test])
        results[i].append(np.sum(np.diag(met.confusion_matrix(pred['class'], labels[test])))/len(labels[test]))
        results[i].append(met.confusion_matrix(pred['class'], labels[test]))
        results[i].append(tfun._T_hddc_ari(labels[test], pred['class']))
        
    for i in results.values():
        print(i)