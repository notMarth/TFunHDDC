import numpy as np
import pandas as pd
import math
import csv

def _T_funhddt_e_step1(fdobj, Wlist, par, clas=0, known=None, kno=None):

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

    t = np.repeat(0., N*K).reshape(N, K)
    tw = np.repeat(0., N*K).reshape(N, K)
    mah_pen = np.repeat(0., N*K).reshape(N,K)
    K_pen = np.repeat(0., N*K).reshape(N,K)
    num = np.repeat(0., N*K).reshape(N,K)
    ft = np.repeat(0., N*K).reshape(N,K)

    s = np.repeat(0., K)

    for i in range(0,K):
        s[i] = np.sum(np.log(a[i, 0:int(d[i])]))

        Qk = Q1[f"{i}"]
        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:int(d[i])],np.repeat(1/b[i], p-int(d[i])) ))))
        #print(aki)
        muki = mu[i]

        Wki = Wlist["W_m"]
        dety = Wlist["dety"]

        mah_pen[:, i] = _T_imahalanobis(x, muki, Wki, Qk, aki)

        tw[:, i] = (nux[i]+p)/(nux[i] + mah_pen[:,i])

        K_pen[:,i] = np.log(prop[i]) + math.lgamma((nux[i] + p)/2) - (1/2) * \
        (s[i] + (p-d[i])*np.log(b[i]) - np.log(dety)) - ((p/2) * (np.log(np.pi)\
        + np.log(nux[i])) + math.lgamma(nux[i]/2) + ((nux[i] + p)/2) * \
        (np.log(1+mah_pen[:, i] / nux[i])))

    ft = np.exp(K_pen)
    ft_den = np.sum(ft, axis=1)
    kcon = - np.apply_along_axis(np.max, 1, K_pen)
    #print(K_pen)
    #print(kcon)
    K_pen = K_pen + np.atleast_2d(kcon).T
    num = np.exp(K_pen)
    #print(num)
    #print(np.sum(num, axis=1))
    t = num / np.atleast_2d(np.sum(num, axis=1)).T

    L1 = np.sum(np.log(ft_den))
    L = np.sum(np.log(np.sum(np.exp(K_pen), axis=1)) - kcon)

    #Why assign these if they get reassigned immediately?
    #trow = N
    #tcol = K
    trow = np.sum(t, axis=1)
    tcol = np.sum(t, axis=0)

    if(np.any(tcol<p)):
        t = (t + 0.0000001) / np.atleast_2d(trow + (K*0.0000001)).T

    if (clas > 0):
        t = unkno*t

        for i in range(0,N):
            if (kno[i] == 1):
                t[i, known[i]] = 1

    #Nothing is returned here in R
    # Return this for testing purposes
    return {'t': t, 'tw': tw, 'L': L}

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

#so_file = "./t-funHDDC/src/TFunHDDC.so"
#c_lib = ctypes.CDLL(so_file)
#print(type(c_lib))

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
a[0] = np.array([3.e45, 8.9e60, 5.e70])
a[1] = np.array([9.e55, 8.5e39, None])
a[2] = np.array([1.6e55, 5.6e75, None])

par = {'K': 3, 'nux':np.array([2.,2,2]), 'a': a, 'b': np.array([-5.e40, -3.5e55, -7.2e30]), 'd':np.array([3,2,2.]), 'mu':mu, 'prop': np.array([1/3,1/3,1/3]), 'Q':Q, 'Q1':Q1}

data = []
with open('data.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

data = {'coefficients': np.array(data[1:]).astype(float)}
res = _T_funhddt_e_step1(data, wlist, par)
print(res)
names = ['t', 'tw', 'L']

with open('twinitTest1Res.csv', 'w', newline='') as csvfile:
    datawriter = csv.writer(csvfile)
    for i in res.items():
        if i[0] != 'L':
            for j in i[1]:
                datawriter.writerow(j)

        else:
            datawriter.writerow(i)