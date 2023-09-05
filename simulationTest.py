import tfunHDDC as tfun
import numpy as np
import modelSimulation as modelSim
from sklearn import metrics as met
data = modelSim.genModelFD(nsplines=70)

model1=["AkjBkQkDk", "AkjBQkDk", "AkBkQkDk", "ABkQkDk", "AkBQkDk", "ABQkDk"]
res = tfun.tfunHDDC(data['data'], K=3,threshold=0.2,model=model1, init="kmeans",nb_rep=1,dfconstr="no")
print(met.confusion_matrix(res.cl, data['labels']))