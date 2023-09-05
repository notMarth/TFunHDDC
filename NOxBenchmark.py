import skfda
import numpy as np
from matplotlib import pyplot as plt
#------------------------------------------------------------------------------#
def fitNOxBenchmark(nbasis = 15):

    nox = skfda.datasets.fetch_nox()
    nox_data = nox['data']
    basis = skfda.representation.basis.BSplineBasis(domain_range = [0,23], n_basis=nbasis)
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, return_basis=True)
    smooth_fd = smoother.fit_transform(nox_data)
    labels = nox['target'].astype(int)
    return {'data': smooth_fd, 'labels':labels}

def plot_NOx(fdn):
    fig = fdn['data'].plot(group = fdn['labels'], group_colors = ['red', 'blue'])
    plt.show()