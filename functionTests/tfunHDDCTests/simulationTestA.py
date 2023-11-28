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
from matplotlib import pyplot as plt

if __name__ == '__main__':
    broke = 0
    K = 3
    thresh=[0.05, 0.1, 0.2]
    #thresh = [0.1]
    models = ['AKJBKQKDK', 'AKJBQKDK', 'AKBKQKDK',  'AKBQKDK', 'ABKQKDK', 'ABQKDK']
    #models=['AKJBKQKDK']
    coreNum = 16
    inits = ['random', 'vector', 'mini-em', 'kmeans']
    #inits=['kmeans']
    nb_rep = 20
    results = {}
    itTest = 100

    for i in range(itTest):

        data=sim.genModelFD(eta=[10, 7, 17])
        trainingdata=data['data']
        labels=data['labels']

        vec = np.repeat(2, len(trainingdata.coefficients))
        vec[0:int(len(trainingdata.coefficients)/3)] = 0
        vec[int(len(trainingdata.coefficients)/2):2*int(len(trainingdata.coefficients)/3)]=1

        for j in inits:
            while True:
                try:
                    start=time.time()
                    print(i, j)

                    res = tfun.tfunHDDC(trainingdata, K=K, threshold=thresh, model=models, init=j, init_vector=vec, nb_rep=nb_rep, min_individuals=2, mc_cores=coreNum)
                    if isinstance(res, tfun.TFunHDDC):
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', tfun._T_hddc_ari(res.cl, labels)]
                        print(f'{i}_{j}_{res.model}_{res.threshold}')
                        print(tfun._T_hddc_ari(res.cl, labels))
                    else:
                        print("Diverged")
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', 0]
                except:
                    broke+=1
                    continue
                print(f'Time taken: {time.time() - start} seconds')
                break

    filename='simARes.csv'
    with open(filename, 'w', newline='') as csvfile:
        headers = ['', 'ARI']
        datawriter = csv.writer(csvfile)
        datawriter.writerow(headers)
        for i in results.keys():
            datawriter.writerow(results[i])
    
    results = {}

    for i in range(itTest):    

        data=sim.genModelFD(eta=[5, 50, 15])
        trainingdata=data['data']
        labels=data['labels']

        vec = np.repeat(2, len(trainingdata.coefficients))
        vec[0:int(len(trainingdata.coefficients)/3)] = 0
        vec[int(len(trainingdata.coefficients)/2):2*int(len(trainingdata.coefficients)/3)]=1

        for j in inits:
            while True:
                try:
                    print(i, j)
                    start=time.time()
                    res = tfun.tfunHDDC(trainingdata, K=K, threshold=thresh, model=models, init=j, init_vector=vec, nb_rep=nb_rep, min_individuals=2, mc_cores=coreNum)
                    if isinstance(res, tfun.TFunHDDC):
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', tfun._T_hddc_ari(res.cl, labels)]
                        print(f'{i}_{j}_{res.model}_{res.threshold}')
                        print(tfun._T_hddc_ari(res.cl, labels))
                    else:
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', 0]
                except:
                    broke +=1
                    continue
                print(f'Time taken: {time.time() - start} seconds')

                break

    filename='simBRes.csv'
    with open(filename, 'w', newline='') as csvfile:
        headers = ['', 'ARI']
        datawriter = csv.writer(csvfile)
        datawriter.writerow(headers)
        for i in results.keys():
            datawriter.writerow(results[i])

    results = {}

    for i in range(itTest):    

        data=sim.genModelFD(eta=[100, 70, 170])
        trainingdata=data['data']
        labels=data['labels']

        vec = np.repeat(2, len(trainingdata.coefficients))
        vec[0:int(len(trainingdata.coefficients)/3)] = 0
        vec[int(len(trainingdata.coefficients)/2):2*int(len(trainingdata.coefficients)/3)]=1

        for j in inits:
            while True:
                try:
                    print(i, j)
                    start=time.time()
                    res = tfun.tfunHDDC(trainingdata, K=K, threshold=thresh, model=models, init=j, init_vector=vec, nb_rep=nb_rep, min_individuals=2, mc_cores=coreNum)
                    if isinstance(res, tfun.TFunHDDC):
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', tfun._T_hddc_ari(res.cl, labels)]
                        print(f'{i}_{j}_{res.model}_{res.threshold}')
                        print(tfun._T_hddc_ari(res.cl, labels))
                    else:
                        results[f'{i}_{j}_{res.model}_{res.threshold}'] = [f'{i}_{j}_{res.model}_{res.threshold}', 0]
                except:
                    broke+=1
                    continue
                print(f'Time taken: {time.time() - start} seconds')

                break

    filename='simCRes.csv'
    with open(filename, 'w', newline='') as csvfile:
        headers = ['', 'ARI']
        datawriter = csv.writer(csvfile)
        datawriter.writerow(headers)
        for i in results.keys():
            datawriter.writerow(results[i])

    print(broke)