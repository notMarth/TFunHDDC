import skfda
import sklearn
from sklearn.utils import Bunch
import numpy as np
import warnings
import pandas as pd
import math
import ctypes
from pathlib import *
import os

def _T_funhddt_e_step1(fdobj, Wlist, par, clas=0, known=None, kno=None):

   #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis or type(fdobj == skfda.FDataGrid)):
        x = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    if type(fdobj) == dict:
        #Multivariate
        if len(x.keys() > 1):
            x = fdobj[0].coefficients
            for i in range(0, len(fdobj)):
                x = np.c_[x, fdobj[f'{i}'].coefficients]
        #univariate
        else:
            x = fdobj.coeficients



    p = x.shape[1]
    N = x.shape[0]
    K = par["K"]
    nux = par["nux"]
    a = par["a"]
    b = par["b"]
    mu = par["mu"]
    d = par["d"]
    prop = par["prop"]
    Q = par["Q"]
    Q1 = par["Q1"]
    b[b<1e-6] = 1e-6

    if clas > 0:
        unkno = (kno-1)*(-1)

    t = np.repeat(0, N*K).reshape(N, K)
    tw = np.repeat(0, N*K).reshape(N, K)
    mah_pen = np.repeat(0, N*K).reshape(N,K)
    K_pen = np.repeat(0, N*K).reshape(N,K)
    num = np.repeat(0, N*K).reshape(N,K)
    ft = np.repeat(0, N*K).reshape(N,K)

    s = np.repeat(0, K)

    for i in range(0,K):
        s[i] = np.sum(np.log(a[i, 0:d[i]]))

        Qk = Q1[f"{i}"]
        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:d[i]],np.repeat(1/b[i], p-d[i]) ))))
        muki = mu[i]

        Wki = Wlist["W_m"]
        dety = Wlist["dety"]

        mah_pen[:, i] = _T_imahalanobis(x, muki, Wki, Qk, aki)

        tw[:, i] = (nux[i]+p)/(nux[i] + mah_pen[:,i])

        #Verify this doesn't break order of operations
        K_pen[:,i] = np.log(prop[i]) + math.lgamma((nux[i] + p)/2) - (1/2) * \
        (s[i] + (p-d[i])*np.log(b[i]) - np.log(dety)) - ((p/2) * (np.log(np.pi)\
        + np.log(nux[i])) + math.lgamma(nux[i]/2) + ((nux[i] + p)/2) * \
        (np.log(1+mah_pen[:, i] / nux[i])))

    ft = np.exp(K_pen)
    ft_den = np.sum(ft, axis=1)
    kcon = - np.apply_along_axis(np.max, 1, K_pen)
    K_pen = K_pen + kcon
    num = np.exp(K_pen)
    t = num / np.sum(num, axis=1)

    L1 = np.sum(np.log(ft_den))
    L = np.sum(np.log(np.sum(np.exp(K_pen), axis=1)) - kcon)

    #Why assign these if they get reassigned immediately?
    #trow = N
    #tcol = K
    trow = np.sum(t, axis=1)
    tcol = np.sum(t, axis=0)

    if(np.any(tcol<p)):
        t = (t + 0.0000001) / (trow + (K*0.0000001))

    if (clas > 0):
        t = unkno*t

        for i in range(0,N):
            if (kno[i] == 1):
                t[i, known[i]] = 1

    #Nothing is returned here in R
    # Return this for testing purposes
    return {t: t, tw: tw, L: L}

def _T_imahalanobis(x, muk, wk, Qk, aki):
    
    #C code not working for now, try compiling dll on current machine?
    #so_file = "./src/TFunHDDC.so"
    #c_lib = ctypes.CDLL(so_file)
    #print(type(c_lib.imahalanobis()))

    p = x.shape[1]
    N = x.shape[0]
    
    X = x - muk

    Qi = np.matmul(wk, Qk)

    xQu = np.matmul(X, Qi)

    proj = np.matmul(np.matmul(X, Qi), aki)

    #rowsum
    res = np.sum(proj ** 2, axis=1)

    return res

