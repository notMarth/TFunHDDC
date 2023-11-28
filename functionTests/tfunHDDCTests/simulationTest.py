import sys
sys.path.append('../..')
import time
import numpy as np
import tfunHDDC as tfun
import modelSimulation as sim
import skfda
import matplotlib.pyplot as plt
import csv
from sklearn import model_selection
from sklearn import metrics
from copy import *

if __name__ == '__main__':
    K = 3
    thresh=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #thresh = [0.1]
    models = ['AKJBKQKDK', 'AKJBQKDK', 'AKBKQKDK',  'AKBQKDK', 'ABKQKDK', 'ABQKDK']
    #models=['AKJBKQKDK']
    coreNum = 16
    inits = ['random', 'vector', 'mini-em', 'kmeans']
    #inits=['kmeans']

    results = {}

    data=sim.genModelFD()
    trainingdata=data['data']
    labels=data['labels']
    vec = np.repeat(2, len(trainingdata.coefficients))
    vec[0:int(len(trainingdata.coefficients)/3)] = 0
    vec[int(len(trainingdata.coefficients)/2):2*int(len(trainingdata.coefficients)/3)]=1


    for i in inits:
        for j in models:
            for k in thresh:
                res = tfun.tfunHDDC(trainingdata, K=K, threshold=k, model=j, init=i, init_vector=vec, nb_rep=30, min_individuals=2, mc_cores=coreNum)
                if isinstance(res, tfun.TFunHDDC):
                    results[f'{i}_{j}_{k}'] = [f'{i}_{j}_{k}', np.sum(np.diag(metrics.confusion_matrix(res.cl,labels)))/len(labels), tfun._T_hddc_ari(res.cl, labels)]
                    print(np.sum(np.diag(metrics.confusion_matrix(res.cl,labels)))/len(labels))
                    print(tfun._T_hddc_ari(res.cl, labels))
                else:
                     results[f'{i}_{j}_{k}'] = [f'{i}_{j}_{k}', 'NA', 'NA']

    filename='simTestRes.csv'
    with open(filename, 'w', newline='') as csvfile:
        headers = ['', 'CCR', 'ARI']
        datawriter = csv.writer(csvfile)
        datawriter.writerow(headers)
        for i in results.keys():
            datawriter.writerow(results[i])
