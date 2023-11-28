import skfda
import numpy as np
import NOxBenchmark as NOx
import csv

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

def _T_hdclassif_dim_choice(ev, n, method, threshold, graph, noise_ctrl, d_set):
    
    N = np.sum(n)
    prop = n/N
    #Is ev a matrix? then K = # of rows in ev
    K = len(ev) if ev.ndim > 1 else 1

    if (ev.ndim > 1 and K > 1):
        p = len(ev[0])

        if (method == "cattell"):
            #Trivial tests show that diff does the same thing in Python that it does in R
            dev = np.abs(np.apply_along_axis(np.diff, 1, ev))
            #print(dev)
            max_dev = np.apply_along_axis(np.nanmax, 1, dev)
            #print(max_dev)
            dev = (dev / np.repeat(max_dev, p-1).reshape(dev.shape)).T
            #print(dev)
            #Apply's axis should cover this situation. Try axis=1 in *args if doesn't work
            d = np.apply_along_axis(np.argmax, 1, (dev > threshold).T*(np.arange(0, p-1))*((ev[:,1:] > noise_ctrl)))

        elif (method == "bic"):

            d = np.repeat(0, K)
            
            for i in range(K):
                Nmax = np.max(np.where(ev[i] > noise_ctrl)) - 1
                B = np.empty((1, Nmax))
                p2 = np.sum(not np.isnan(ev[i]))
                Bmax = -np.inf

                for kdim in range(Nmax):
                    if d[i] != 0 and kdim > d[i] + 10: break
                    a = np.sum(ev[i, 0:kdim])/kdim
                    b = np.sum(ev[i, (kdim + 1):p2])/(p2-kdim)

                    if b < 0 or a < 0:
                        B[kdim] = -np.inf

                    else:
                        L2 = -1/2*(kdim*np.log(a) + (p2 - kdim)*np.log(b) - 2*np.log(prop[i]) +p2*(1+1/2*np.log(2*np.pi))) * n[i]
                        B[kdim] = 2*L2 - (p2+kdim*(p2-(kdim+1)/2)+1) * np.log(n[i])

                    if B[kdim] > Bmax:
                        Bmax = B[kdim]
                        d[i] = kdim

            if graph:
                None

        elif method == "grid":
            d = d_set

        else:
            ev = ev.flatten()
            p = len(ev)

            if method == "cattell":
                dvp = np.abs(np.diff(ev))
                Nmax = np.max(np.which(ev>noise_ctrl)) - 1
                if p ==2:
                    d = 1
                else:
                    d = np.max(np.which(dvp[0:Nmax] >= threshold*np.max(dvp[0:Nmax])))
                diff_max = np.max(dvp[0:Nmax])

            elif method == "bic":
                d = 0
                Nmax = np.max(np.where(ev > noise_ctrl)) - 1
                B = np.empty((1, Nmax))
                Bmax = -np.inf

                for kdim in range(Nmax):
                    if d != 0 and kdim > d+10:
                        break
                    a = np.sum(ev[0:kdim])/kdim
                    b = np.sum(ev[(kdim+1):p])/(p-kdim)
                    if(b <= 0 or a <= 0):
                        B[kdim] = -np.inf
                    else:
                        L2 = -1/2*(kdim*np.log(a) + (p-kdim)*np.log(b)+p*(1+1/2*np.log(2*np.pi)))*N
                        B[kdim] = 2*L2 - (p+kdim * (p-(kdim + 1)/2)+1)*np.log(N)

                    if B[kdim] > Bmax:
                        Bmax = B[kdim]
                        d = kdim

    return d

noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3


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
t = np.array(mats['t']).astype(np.longdouble)

data = []
mats = {'W': [], 'W_m': []}
with open('functionTests/initmypcaTests/W.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

for i in data[1:]:
    mats['W'].append(i[0:15])
    mats['W_m'].append(i[15:30])
    mats['dety'] = i[30]
W = np.array(mats['W']).astype(np.longdouble)
W_m = np.array(mats['W_m']).astype(np.longdouble)
dety = np.longdouble(mats["dety"])

Wlist = {'W':W, 'W_m':W_m, "dety":dety}

#print(W)

n = np.sum(t, axis=1)
data = NOx.fitNOxBenchmark().data
p = data.coefficients.shape[1]
K = len(t[0])
ev = np.repeat(0., K*p).reshape((K,p))

for i in range(K):
    
    donnees = _T_initmypca_fd1(data, Wlist, t[:,i])

    ev[i] = donnees["valeurs_propres"]

print(ev[:, 1:])
d = _T_hdclassif_dim_choice(ev, n, 'cattell', threshold, False, noise, d_set)
print(d)