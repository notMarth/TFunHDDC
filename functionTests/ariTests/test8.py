import numpy as np
import scipy.special as scip
import pandas as pd

def _T_hddc_ari(x, y):
    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    tab = pd.crosstab(x, y).to_numpy()
    print(tab)
    if np.all(tab.shape == (1,1)): return 1
    a = np.sum(scip.binom(tab, 2))
    print(a)
    b = np.sum(scip.binom(np.sum(tab, axis=1), 2)) - a
    print(b)
    c = np.sum(scip.binom(np.sum(tab, axis=0), 2)) - a
    print(c)
    d = scip.binom(np.sum(tab), 2) - a - b - c
    print(d)
    ari = (a - (a + b) * (a + c)/(a+b+c+d))/((a+b+a+c)/2 - (a+b) * (a + c)/(a+b+c+d))
    return ari

a = np.array([6.64e-55, 45.3e-32, 45.3e-32, 77.689e-69, 6.64e-55, 95.89e-60])
b = np.array([6.63e-45, 45.3e-31, 45.3e-32, 77.689e-69, 6.64e-55, 89.89e-89])
print(_T_hddc_ari(a,b))