import sys
sys.path.append('.')
import tfunHDDC as tfun
import triangleSimulation as tri
from sklearn import metrics as met

'''
Multivariate example using an example from the paper which TFunHDDC is based on.
Each model is run separately, as the log likelihood may prefer a clustering with
a higher CCR but lower ARI.

After each model is run, the ARI and a table comparing the clustered classes vs
the true labels are outputted.

'''


if __name__ == '__main__':
    data = tri.genTriangleScenario1()
    fd = data['data']
    labels = data['labels']
    models = ['akjbkqkdk', 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk']


    for i in models:
        res = tfun.tfunHDDC(fd, K=6, model=i, threshold=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6], nb_rep=20)
        print(i)
        print(met.confusion_matrix(labels, res.cl))
        print(tfun._T_hddc_ari(labels, res.cl))