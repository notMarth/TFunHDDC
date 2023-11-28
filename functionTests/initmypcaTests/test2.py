import skfda
import numpy as np
import csv
import NOxBenchmark as NOx

def _T_initmypca_fd1(fdobj, Wlist, Ti):
        
    #Univariate here
    if type(fdobj) == skfda.FDataBasis:
        temp = fdobj.copy()
        mean_fd = fdobj.copy()
        coef = fdobj.coefficients.copy()
        #by default numpy cov function uses rows as variables and columns as observations, opposite to R
        mat_cov = np.cov(m=coef, aweights=Ti, ddof=0, rowvar=False)
        #may need to try this with other params depending on how weights are passed in
        coefmean = np.average(coef, axis=0, weights=Ti)
        #Verify this
        temp.coefficients = np.apply_along_axis(lambda row: row - coefmean, axis=1, arr=temp.coefficients)
        #Replaces as.matrix(data.frame(mean=coefmean))
        mean_fd.coefficients = coefmean
        cov = (Wlist['W_m']@mat_cov)@(Wlist['W_m'].T)

        valeurs_propres, vecteurs_propres = np.linalg.eig(cov.astype(float))
        fonctionspropres = fdobj.copy()
        bj = np.linalg.solve(Wlist['W_m'].astype(float), np.eye(Wlist['W_m'].shape[0]).astype(float))@vecteurs_propres
        fonctionspropres.coefficients = bj

        scores = skfda.misc.inner_product_matrix(temp.basis, fonctionspropres.basis)
        varprop = valeurs_propres / np.sum(valeurs_propres)
        ipcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd, 'mux': coefmean}

    #Multivariate
    else:
        mean_fd = {}
        temp = fdobj.copy()
        for i in range(len(fdobj)):

            #TODO Start at 0? or should we start at 1?
            mean_fd[f'{i}'] = temp[f'{i}'].copy()

        coef = temp['0'].coefficients
        for i in range(1, len(fdobj)):
            coef = np.c_[coef, temp[f'{i}'].coefficients.copy()]

        #print(coef)
        mat_cov = np.cov(m=coef, aweights=Ti, ddof=0, rowvar=False)
        coefmean = np.average(coef, axis=0, weights=Ti)

        n_lead = 0
        #R Doesn't transpose this here, might need shape[1] instead
        n_var = temp['0'].coefficients.shape[1]
        #Sweep
        tempi = temp['0'].copy()
        tempi.coefficients = np.apply_along_axis(lambda row: row - coefmean[(n_lead):(n_var + n_lead)], axis=1, arr=tempi.coefficients)

        mean_fd['0'].coefficients = coefmean[(n_lead):(n_var + n_lead)]

        for i in range(1, len(fdobj)):
            tempi = temp[f'{i}'].copy()
            n_lead = n_lead + n_var
            #R doesn't transpose this here, might need shape[1] instead
            #print(temp[f'{i}'].coefficients)
            n_var = temp[f'{i}'].coefficients.shape[1]
            tempi.coefficients = np.apply_along_axis(lambda row: row - coefmean[(n_lead):(n_var + n_lead)], axis=1, arr=tempi.coefficients)
            mean_fd[f'{i}'].coefficients = coefmean[(n_lead):(n_var + n_lead)]

        cov = (Wlist['W_m']@mat_cov)@(Wlist['W_m'].T)
        valeurs_propres, vecteurs_propres = np.linalg.eig(cov.astype(float))
        bj = np.linalg.solve(Wlist['W_m'].astype(float), np.eye(Wlist['W_m'].shape[0]).astype(float))@vecteurs_propres
        fonctionspropres = temp['0'].copy()
        fonctionspropres.coefficients = bj
        scores = (coef@Wlist['W'])@bj

        varprop = valeurs_propres / np.sum(valeurs_propres)

        ipcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres,
                  'scores': scores, 'covariance': cov, 'U': bj, 'varprop': varprop,
                  'meanfd': mean_fd, 'mux': coefmean}

    return ipcafd

data = []
mats = {'t': [], 'tw': []}
with open('functionTests/initmypcaTests/t1.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

#print(data)
#print(data[1][0:3])
for i in data[1:]:
    mats['t'].append(i[0:3])
t = np.array(mats['t']).astype(int)

data = []
mats = {'W': [], 'W_m': []}
with open('functionTests/initmypcaTests/Wmulti.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

for i in data[1:]:
    mats['W'].append(i[0:30])
    mats['W_m'].append(i[30:60])
    mats['dety'] = i[60]
W = np.array(mats['W']).astype(np.longdouble)
W_m = np.array(mats['W_m']).astype(np.longdouble)
dety = np.longdouble(mats["dety"])

Wlist = {'W':W, 'W_m':W_m, "dety":dety}

data = NOx.fitNOxBenchmark().data
data = {'0': data, '1': data}
#print(data)
for i in range(len(t[0])):
    
    print(_T_initmypca_fd1(data, Wlist, t[:,i])['mux'])
    #print(data)