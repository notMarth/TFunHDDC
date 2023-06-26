import numpy as np
import pandas as pd
import csv

def _T_funhddt_twinit(fdobj, wlist, par, nux):

    #try this if fdobj is an fdata (Univariate only right now)
    #if(type(fdobj) == skfda.FDataBasis or type(fdobj == skfda.FDataGrid)):
    #    x = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    #Will be changed outside of R testing so that the expected element in the
    #dict or dataframs is an FDataBasis
    if type(fdobj) == dict or type(fdobj) == pd.DataFrame:
        #Multivariate
        #Here in R, the first element will be named '1'
        if len(fdobj.keys()) > 1:
            x = np.transpose(fdobj['1']['coefficients'])
            for i in range(0, len(fdobj)):
                x = np.c_[x, np.transpose(fdobj[f'{i}']['coefficients'])]
        #univariate
        else:
            x = fdobj['coefficients'].T

    p = x.shape[1]
    n = x.shape[0]
    K = par['K']
    a = par['a']
    b = par['b']
    mu = par['mu']
    d = par['d']
    Q = par['Q']
    Q1 = par['Q1']
    W = np.zeros(n*K).reshape((n,K))

    b[b<1e-6]  = 1e-6

    mah_pen = np.zeros(n*K).reshape((n,K))

    for i in range(0, K):
        Qk = Q1[f'{i}']

        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:int(d[i])],np.repeat(1/b[i], p-int(d[i])) ))))
        muki = mu[i]

        wki = wlist['W_m']
        mah_pen[:,i] = _T_imahalanobis(x, muki, wki, Qk, aki)
        W[:, i] = (nux[i] + p) / (nux[i] + mah_pen[:, i])

    return W

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

    res = np.sum(proj ** 2, axis=1)

    return res

w = np.diag(np.repeat(1.,21))
w_m = np.diag(np.repeat(1.,21))
dety = 1.

wlist = {'W': w, 'W_m': w_m, 'dety': dety}

Q = {}
for i in range(0, 3):
    Q[f'{i}'] = np.repeat(i*0.1, 21*21).reshape((21,21))

Q1 = {}
for i in range(0, 3):
    Q1[f'{i}'] = np.repeat((i+1)*0.1, 21*21).reshape((21,21))

nsplines = 21
r1 = np.concatenate((np.array([1.,0,50, 100]), np.repeat(0, nsplines-4)))
r2 = np.concatenate((np.array([0.,0,80,0,40,2]), np.repeat(0, nsplines-6)))
r3 = np.concatenate((np.repeat(0, nsplines-6), np.array([20.,0,80,0,0,100])))
mu = np.concatenate((np.array([r1]), np.array([r2]), np.array([r3])), axis=0)

a = np.zeros((3,3))
a[0] = np.array([5.e-60, 1e-52, 6e-64])
a[1] = np.array([7.5e-33, 7.9e-46, None])
a[2] = np.array([7e-24, 3.2e-53, None])

par = {'K': 3, 'nux':np.array([2.,2,2]), 'a': a, 'b': np.array([3.2e-44,7.2e-60,1.e-50]), 'd':np.array([3,2,2.]), 'mu':mu, 'prop': np.array([1/3,1/3,1/3]), 'Q':Q, 'Q1':Q1}

data = []
with open('data.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

data = {'coefficients': np.array(data[1:]).astype(float)}

res = _T_funhddt_twinit(data, wlist, par, par['nux'])

with open('twinitTest4Res.csv', 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    for i in res:
        datawriter.writerow(i)
        