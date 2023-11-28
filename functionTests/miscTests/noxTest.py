import tfunHDDC as tfun
import funHDDC as fun
import numpy as np
import NOxBenchmark as NOx
from sklearn import metrics as met

import csv

if __name__ == '__main__':
    data = NOx.fitNOxBenchmark()
    #NOx.plot_NOx(data)

    vec = []
    '''
    with open('HWcluster.csv', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in datareader:
            vec.append(row[0].split(',')[1:][0])
 
    vec = np.array(vec[1:], dtype=int)
    vec = vec - 1
    print(vec)
    '''

   # res = tfun.tfunHDDC(data['data'], min_individuals=3, model='all', K=2, threshold=[0.05, 0.2, 0.4, 0.6], nb_rep=20, init='kmeans')
   # print(np.sum(np.diag(met.confusion_matrix(res.cl, data['labels'])))/len(data['labels']))
   # print(tfun._T_hddc_ari(data['labels'], res.cl))

data = NOx.fitNOxBenchmark(15)
NOx.plotNOxBenchmark()
labels = data['labels']
data = data['data']
model1=["AkjBkQkDk", "AkjBQkDk", "AkBkQkDk", "ABkQkDk", "AkBQkDk", "ABQkDk"]
#res = tfun.tfunHDDC(data,K=2,threshold=0.2,model=model1, init="kmeans",nb_rep=1,dfconstr="no")
res1 = fun.funHDDC(data,K=2,threshold=0.2,model=model1, init="kmeans",nb_rep=1)
print(met.confusion_matrix(res1.cl, labels))

