from sklearn.utils import Bunch
import skfda
import numpy as np
from matplotlib import pyplot as plt
#------------------------------------------------------------------------------#
def fitNOxBenchmark(nbasis = 15):

    nox = skfda.datasets.fetch_nox()
    nox_data = nox['data']
    basis = skfda.representation.basis.BSplineBasis(domain_range = np.linspace(0.,23., num=2),
                                            n_basis=nbasis)
                                            
    nox_fd = nox_data.to_basis(basis)
    labels = nox['target'].astype(int)
    return Bunch(data = nox_fd, target = labels)

def plot_NOx(fdn):
    fig = fdn['data'].plot(group = fdn['target'], group_colors = ['red', 'blue'])
    plt.show()