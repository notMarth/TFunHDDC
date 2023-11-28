import numpy as np
import skfda.misc as mis
import scipy.linalg as scil

def W_multigen(data):
    x = data['0'].coefficients

    for i in range(1, len(data)):
        x = np.c_[x, data[f'{i}'].coefficients]
    p = x.shape[1]

    W_fdobj = []
    for i in range(len(data)):
        W_fdobj.append(mis.inner_product_matrix(data[f'{i}'].basis, data[f'{i}'].basis))

    prow = W_fdobj[-1].shape[0]
    pcol = len(data)*prow
    W1 = np.c_[W_fdobj[-1], np.zeros((prow, pcol-W_fdobj[-1].shape[1]))]
    W_list = {}

    for i in range(1, len(data)):
        W2 = np.c_[np.zeros((prow, (i)*W_fdobj[-1].shape[1])),
                W_fdobj[i],
                np.zeros((prow, pcol - (i+1) * W_fdobj[-1].shape[1]))]
        W_list[f'{i-1}'] = W2

    W_tot = np.concatenate((W1,W_list['0']))
    if len(data) > 2:
        for i in range(1, len(data)-1):
            W_tot = np.concatenate((W_tot, W_list[f'{i}']))

    W_tot[W_tot < 1.e-15] = 0
    W_m = scil.cholesky(W_tot)
    dety = scil.det(W_tot)
    Wlist = {'W':W_tot, 'W_m': W_m, 'dety': dety}

    return Wlist