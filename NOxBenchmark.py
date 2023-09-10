import skfda
import numpy as np
from matplotlib import pyplot as plt
#------------------------------------------------------------------------------#
def fitNOxBenchmark(nbasis = 15):

    nox = skfda.datasets.fetch_nox()
    nox_data = nox['data']
<<<<<<< HEAD
    basis = skfda.representation.basis.BSplineBasis(domain_range = [0,23], n_basis=nbasis)
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, return_basis=True)
    smooth_fd = smoother.fit_transform(nox_data)
    labels = nox['target'].astype(int)
    return {'data': smooth_fd, 'labels':labels}
=======
   # basis = skfda.representation.basis.BSplineBasis(domain_range = np.linspace(0,23, num=2),
                                           # n_basis=nbasis)
    basis = skfda.representation.basis.FourierBasis(domain_range = np.linspace(0,23, num=2),
                                             n_basis=nbasis)                                           
    nox_fd = nox_data.to_basis(basis)
    labels = nox['target']
    print(nox_data)
    return Bunch(data = nox_fd, target = labels)
>>>>>>> 678edf3241b962359205629f7191fae07a5824b0

def plot_NOx(fdn):
    fig = fdn['data'].plot(group = fdn['labels'], group_colors = ['red', 'blue'])
    plt.show()