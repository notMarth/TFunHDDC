import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx
from sklearn import metrics as met
data = NOx.fitNOxBenchmark(15)
labels = data['labels']
data = data['data']
model1=["AkjBkQkDk", "AkjBQkDk", "AkBkQkDk", "ABkQkDk", "AkBQkDk", "ABQkDk"]
res = tfun.tfunHDDC(data,k=2,threshold=0.6,model=model1, init="kmeans",nb_rep=20,dfconstr="no" ,min_individuals=3)
print(met.confusion_matrix(res.cl, labels))