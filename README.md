# TFunHDDC
tfunHDDC is an adapatation of funHDDC (Scmutz et al., 2018) that uses t-distributions for robust clustering in the presence of outliers. In addition, funHDDC is also available for use.

# Usage
The main function is `tfunHDDC`. It is designed to take a functional data object along with some parameters for the clustering process.
```
def tfunHDDC(data, K=np.arange(1,11), model='AKJBKQKDK', known=None, dfstart=50., 
             dfupdate='approx', dfconstr='no', threshold=0.1, itermax=200, 
             eps=1.e-6, init='random', criterion='bic', d_select='cattell', 
             init_vector=None, show=True, mini_nb=[5,10], min_individuals=4,
             mc_cores=1, nb_rep=2, keepAllRes=True, 
             kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}, 
             d_max=100, d_range=2, verbose=True, Numba=True)
```

Once clustering is done, a `TFunHDDC` object is returned, containing the parameters used for the clustering and the assigned clusters to each curve.

Prediction can also be done using the `TFunHDDC.predict()` function. This requires another `FDataBasis` object with the same number of basis functions as the `FDataBasis` originally clustered on.

The above also holds for funHDDC as well.
```
def funHDDC(data, K=np.arange(1,11), model='AKJBKQKDK', known=None, threshold=0.1, itermax=200, 
            eps=1.e-6, init='random', criterion='bic', d_select='cattell', 
            init_vector=None, show=True, mini_nb=[5,10], min_individuals=4,
            mc_cores=1, nb_rep=2, keepAllRes=True,
            kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}, d_max=100,
            d_range=2, verbose=True):
```
Instead of a `TFunHDDC` object, `funHDDC` returns a `FunHDDC` object. Note that prediction cannot be done with FunHDDC.

# Examples
There are various examples provided to illustrate how to use TFunHDDC and funHDDC. Examples  are taken from the paper from which TFunHDDC is based on (Anton, C., Smith, I. Model-based clustering of functional data via mixtures of t distributions. Adv Data Anal Classif (2023). https://doi.org/10.1007/s11634-023-00542-w)

paperScenarioAExample.py:
300 curves are simulated with 35 splines for the basis functions. There are 3
scenarios that will be run. Scenario A runs with eta = 10, 7, 17 for groups 1,
2, and 3.

paperScenarioBExample.py:
300 curves are simulated with 35 splines for the basis functions. Scenario B runs with eta = 5, 50, 15 for groups 1,
2, and 3.

paperScenarioCExample.py:
300 curves are simulated with 35 splines for the basis functions. Scenario C runs with eta = 100, 70, 170 for groups 1,
2, and 3.

triangleSimulationMultivariateExample.py:
multivariate example taken from the paper from which TFunHDDC is adapted from.
