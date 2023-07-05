import skfda
import numpy as np
from scipy import special as scip
import csv

def _T_initmypca_fd1(fdobj, Wlist, Ti):
    
    #TODO add multivariate
    mean_fd = fdobj
    #TODO does pass by reference happen here like with lists?
    coef = fdobj.coefficients.T
    #by default numpy cov function uses rows as variables and columns as observations, opposite to R
    mat_cov = np.cor(m=coef, aweights=Ti, ddof=0, rowvar=False)
    #may need to try this with other params depending on how weights are passed in
    coefmean = np.average(coef, axis=0, weights=Ti)
    #Verify this
    fdobj.coefficients = np.apply_along_axis(lambda col: col - coefmean, axis=0, arr=fdobj.coefficients)

    #Replaces as.matrix(data.frame(mean=coefmean))
    mean_fd.coefficients = {'mean':coefmean}
    cov = (Wlist['W_m']@mat_cov)@(Wlist['W_m'].T)

    valeurs = np.linalg.eig(cov)
    valeurs_propres = valeurs.eigenvalues
    vecteurs_propres = valeurs.eigenvectors
    fonctionspropres = fdobj
    bj = np.linalg.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@vecteurs_propres
    fonctionspropres['coefficients'] = bj

    scores = skfda.misc.inner_product_matrix(fdobj, fonctionspropres)
    varprop = valeurs_propres / np.sum(valeurs_propres)
    ipcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd, 'mux': coefmean}

    return ipcafd

data = []

with open('coefs.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

data = {'coefficients': np.array(data[1:]).astype(float)}

temp=[]
t=[]
with open('t.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        temp.append(row[0].split(',')[1:])

#print(data)
#print(data[1][0:3])
for i in temp[1:]:
    t.append(i[0:3])
    

t = np.array(t).astype(float)

print(data)
print(t)
print(_T_initmypca_fd1())