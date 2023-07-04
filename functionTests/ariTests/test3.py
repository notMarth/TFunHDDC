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
    c = np.sum(scip.binom(np.sum(tab, axis=0), 2)) - a
    d = scip.binom(np.sum(tab), 2) - a - b - c
    ari = (a - (a + b) * (a + c)/(a+b+c+d))/((a+b+a+c)/2 - (a+b) * (a + c)/(a+b+c+d))
    return ari

a = np.array([66, 3, 77, 53, 10, 4])
b = np.array([10, 3, 76, 52, 3, 4])
print(_T_hddc_ari(a,b))