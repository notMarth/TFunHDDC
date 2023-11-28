import sys
sys.path.append('../../..')
import time
import numpy as np
import tfunHDDC as tfun
import NOxBenchmark as NOx
import skfda
import matplotlib.pyplot as plt
import csv
from sklearn import model_selection
from sklearn import metrics
from copy import *

if __name__ == '__main__':
    K = 2
    thresh=[0.01, 0.1, 0.2, 0.4]
    #thresh = [0.1]
    models = ['AKJBKQKDK', 'AKJBQKDK', 'AKBKQKDK',  'AKBQKDK', 'ABKQKDK', 'ABQKDK']
    coreNum = 16
    inits = ['random', 'vector', 'mini-em', 'kmeans']
    #inits=['kmeans']
    nox = skfda.datasets.fetch_nox()
    trainingdata = nox['data']
    testingdata = nox['data']
    labels = nox['target']

    # trainingdata.coefficients = trainingdata.coefficients[0:int(len(trainingdata.coefficients)/2)]
    # testingdata.coefficients = testingdata.coefficients[int(len(trainingdata)/2):]    

    trainingLabels = labels[0:int(len(trainingdata)/2)]
    testingLabels = labels[int(len(trainingdata)/2):]

    # training, testing, trainingLabels, testingLabels = model_selection.train_test_split(nox_data, labels, shuffle=False, test_size=0.50, train_size=0.50)
    # if len(trainingdata.coefficients) > len(testingdata.coefficients):
    #     trainingdata.coefficients=trainingdata.coefficients[:len(testingdata.coefficients)]
    #     trainingLabels = trainingLabels[0:len(testingdata.coefficients)]
    # else:
    #     testingdata.coefficients= testingdata.coefficients[:len(trainingdata.coefficients)]
    #     testingLabels = testingLabels[0:len(trainingdata.coefficients)]

    trainingdata = trainingdata[0:int(len(trainingdata)/2)]
    testingdata = testingdata[int(len(testingdata)/2):]

    if len(trainingdata) > len(testingdata):
        trainingdata = trainingdata[0:len(testingdata)]
        trainingLabels = trainingLabels[0:len(testingdata)]
    else:
        testingdata = testingdata[0:len(trainingdata)]
        testingLabels = testingLabels[0:len(trainingdata)]

    basis = skfda.representation.basis.BSplineBasis(domain_range = np.linspace(0.,23., num=2),
                                            n_basis=15)
    trainingdata = trainingdata.to_basis(basis)
    testingdata = testingdata.to_basis(basis)

    trainingdata = NOx.fitNOxBenchmark()['data']
    vec = np.repeat(1, len(trainingdata.coefficients))
    vec[0:int(len(trainingdata.coefficients)/2)] = 0

    data = []
    Rdata = {}
    with open('RPredictTest.csv', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            data.append(row[0].split(',')[1:])

        for i in inits:
            for j in models:
                for k in thresh:
                    Rdata[f'{i}{j}{k}'] = []
                    for a in data[1:]:
                        Rdata[f'{i}{j}{k}'].append(a[0])

    results = {}

    for i in inits:
        for j in models:
            for k in thresh:
                res = tfun.tfunHDDC(trainingdata, K=K, threshold=k, model=j, init=i, init_vector=vec, nb_rep=20, min_individuals=2, mc_cores=coreNum)
                predict = res.predict(trainingdata)['class']
                diff = res.cl - predict
                results[f'{i}{j}{k}'] = np.sum(np.diag(metrics.confusion_matrix(predict['class'],labels)))/len(predict)
    # print(np.sum(np.diag(metrics.confusion_matrix(predict['class'],labels)))/len(predict['class']))
    # print(tfun._T_hddc_ari(predict['class'], labels))
