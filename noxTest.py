import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx
from sklearn import metrics as met
data = NOx.fitNOxBenchmark(7)
labels = data['target']
data = data['data']
model1=["AkjBkQkDk", "AkjBQkDk", "AkBkQkDk", "ABkQkDk", "AkBQkDk", "ABQkDk"]
res = tfun.tfunHDDC(data,K=2,threshold=0.2,model=model1, init="kmeans",nb_rep=1,dfconstr="no")
print(met.confusion_matrix(res.cl, labels))