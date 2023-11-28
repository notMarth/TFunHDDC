import sys
sys.path.append('.')
import tfunHDDC as tfun
import triangleSimulation as tri
from sklearn import metrics as met

'''
Multivariate example using an example from the paper which TFunHDDC is based on.
Run triangle simulation from TFunHDDC paper.

After each model is run, the ARI and a table comparing the clustered classes vs
the true labels are outputted.

'''


if __name__ == '__main__':
    data = tri.genTriangleScenario1()
    fd = data['data']
    labels = data['labels']

    tri.plotTriangles(data)

    res = tfun.tfunHDDC(fd, K=4, model='all', threshold=0.2, nb_rep=5)
    print(met.confusion_matrix(labels, res.cl))
    print(tfun._T_hddc_ari(labels, res.cl))