#Required Libraries------------------------------------------------------------#
import skfda
import sklearn
from sklearn.utils import Bunch
import numpy as np
import warnings
import pandas as pd
import math
import ctypes
import scipy.special as scip
import scipy.optimize as scio
#------------------------------------------------------------------------------#


class _Table:
    """
    Store data with row names and column names
    
    Attributes
    ----------
    data : numpy array
    rownames : numpy array
    colnames : numpy array

    
    """
    def __init__(self, data, rownames, colnames):
        self.data = data
        self.rownames = rownames
        self.colnames = colnames

        #check rows and columns match data dimension
        #Removed
        """
        if rownames.shape != data.shape[0]:
            raise ValueError("Dimension of row names doesn't match dimension of rows in data")
        if colnames.shape != data.shape[1]:
            raise ValueError("Dimension of column names doesn't match dimension of columns in data")
        """

    def switchRow(self, ind1, ind2):
        if ind1 < ind2:
            self.data[[ind1, ind2]] = self.data[[ind2, ind1]]
            self.rownames[[ind1, ind2]] = self.rownames[[ind1, ind2]]
        else:
            self.data[[ind2, ind1]] = self.data[[ind1, ind2]]
            self.rownames[[ind2, ind1]] = self.rownames[[ind1, ind2]]

    def switchCol(self, ind1, ind2):
        if ind1 < ind2:
            self.data.T[[ind1, ind2]] = self.data.T[[ind2, ind1]]
            self.colnames[[ind1, ind2]] = self.colnames[[ind2, ind1]]
        else:
            self.data.T[[ind2, ind1]] = self.data.T[[ind1, ind2]]
            self.colnames[[ind2, ind1]] = self.colnames[[ind1, ind2]]
    


#TODO add default values
#*args argument replaces ... in R code
#fdobj should be a dictionary containing an FData object and optionally labels
def _T_funhddc_main1(fdobj, wlist, K, dfstart, dfupdate, dfconstr, model,
                     itermax, threshold, method, eps, init, init_vector,
                     mini_nb, min_individuals, noise_ctrl, com_dim,
                     kmeans_control, d_max, d_set, known, *args):
    modelNames = ["AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", 
                  "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD",
                  "AKBQKD", "ABQKD"]
    
    #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis or type(fdobj == skfda.FDataGrid)):
        data = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    if type(fdobj) == dict:
        #Multivariate
        if len(data.keys() > 1):
            data = fdobj[0].coefficients
            for i in range(0, len(fdobj)):
                data = np.c_[data, fdobj[f'{i}'].coefficients]
        #univariate
        else:
            data = fdobj.coeficients

    n, p = data.shape
    #com_ev is None (better way of phrasing) instead of = None
    com_ev = None

    d_max = min(n,p,d_max)

    #classification

    if(known == None):
        #clas = 0
        kno = None
        test_index = None

    else:

        if len(known) != n:
            print("Known classifications vector not given, or not the same length as the number of samples (see help file)")
            return None
        
        #TODO find out how to handle missing values. For now, using numpy NaN

        #Known should use None when data is not known
        #np.isnan(np.sum) is faster than np.isnan(np.min) for the general case
        else:
            if (not np.isnan(np.sum(known))):
                warnings.warn("No Nones in 'known' vector supplied. All values have known classification (parameter estimation only)")

                test_index = np.linspace(0, n-1, n)
                kno = np.repeat(1, n)
                unkno = (kno-1)*(-1)
                K = len(np.unique(known))
                #clas = len(training)/n
                init_vector = known.astype(int)
                init = "vector"

            else:
                #isnan will detect only nan objects. Will crash if strings are supplied
                #numpy arrays will convert None to nan if converted to float
                #TODO switch NaNs to Nones here for known
                training = np.where((not np.isnan(known)))
                test_index = training
                kno = np.zeros(n)
                kno[test_index] = 1
                unkno = (kno - 1)*(-1)

                #clas = len(training)/n
            #why do the evaluations to clas if its set to 1 here?
            #clas = 1

        if K > 1:
            t = np.zeros((n, K))
            tw = np.zeros((n, K))

            #TODO finish init
            match init:
                case "vector":
                    #clas always > 0 though
                    #if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    #ensure this is the correct size. In R: matchtab[1:cn1, 1:K]
                    temp = pd.crosstab(index = known, columns = init_vector)
                    matchtab[0:cn1, 0:K] = temp.values
                    table = _Table(data=matchtab, rownames=temp.index, colnames = temp.columns)
                    
                    #___OLD___
                    #matchtab[0:cn1, 0:K] = pd.crosstab(index = known, columns = init_vector, rownames = ["known"], colnames = ["init"]) 
                    
                    #numpy.where always returns a tuple of at least length 1 (useless since we're not multidim here),
                    #so we need to access the zeroth element
                    #np.isin behaves identical to %in% in R
                    rownames = np.array([table.rownames, np.where(not np.isin(np.arange(0,K,1), np.unique(known[test_index])))[0]]).flatten()
                    table.rownames = rownames
                    matchit = np.repeat(0, K)

                    while(np.max(table.data)>0):
                        #TODO figure out how to match the R dimensions
                        #Check R code here
                        ij = int(np.where(matchtab == max(matchtab.to_numpy().flatten())))
                        ik = np.where(matchtab == max(matchtab.to_numpy()[:,ij]))
                        matchit[ij] = np.repeat(-1, K)
                        matchtab[:,ij] = np.repeat(-1, K)
                        matchtab[int(ik)] = np.repeat(-1, K)

                    #TODO FIX
                    matchit[np.where(matchit == 0)] = np.where((np.arange(0,K,1) not in np.unique(matchit)))
                    #TODO pass by reference might not let this work correctly like it does in R
                    initnew = init_vector
                    for i in range(0, K):
                        initnew[init_vector == i] = matchit[i]

                    init_vector = initnew

                    for i in range(0, K):
                        t[np.array(np.where(init_vector == i), i)] = 1

                case "kmeans":
                    kmc = kmeans_control
                    km = sklearn.cluster.KMeans(n_clusters = K, max_iter = itermax)
                    cluster = km.fit_predict(data)

                    #if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.rep(-1, K*K).reshape((K,K))
                    matchtab[0:cn1, 0:K] = pd.crosstab(known, cluster, rownames=['known'], colnames=['init'])
                    #Rownames here
                    matchit = np.repeat(0, K)

                    while(max(matchtab.to_numpy().flatten())>0):
                        #TODO figure out how to match the R dimensions
                        ij = int(np.where(matchtab == max(matchtab.to_numpy().flatten())))
                        ik = np.where(matchtab == max(matchtab.to_numpy()[:,ij]))
                        matchit[ij] = np.repeat(-1, K)
                        matchtab[:,ij] = np.repeat(-1, K)
                        matchtab[int(ik)] = np.repeat(-1, K)

                    matchit[np.where(matchit == 0)] = np.where(np.arange(0, K, 1) != np.unique(matchit))
                    #TODO pass by reference may not let this work correctly
                    knew = cluster
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.array(np.where(cluster == i), i)] = 1

                #skip trimmed kmeans
                case "tkmeans":
                    kmc = kmeans_control

                case "mini-em":
                    prms_best = 1
                    for i in range(0, mini_nb[0]):
                        prms = _T_funhddc_main1(fdobj=fdobj, wlist=wlist, known=known, dfstart=dfstart,
                                                dfupdate=dfupdate, dfconstr=dfconstr, model=model,
                                                threshold=threshold, method=method,
                                                itermax=mini_nb[1], init_vector=0, init="random",
                                                mini_nb=mini_nb, min_individuals=min_individuals,
                                                noise_ctrl=noise_ctrl, kmeans_control=kmeans_control,
                                                com_dim=com_dim, d_max=d_max, d_set=d_set)
                        if len(prms) != 1:
                            if len(prms_best == 1):
                                prms_best = prms
                            #TODO check what is being returned (array? dataframe? fdatabasis? fdataFrame?)
                            #Assuming dictionary here
                            elif prms_best['loglik'][-1] < prms['loglik'][-1]:
                                prms_best = prms

                    #verify that this is what R line 627 is doing
                    if len(prms_best) == 1:
                        return 1
                    
                    t = prms_best['posterior']

                    #if clas > 0:
                    #TODO verify this is identical to R
                    cluster = np.argmax(t, axis=0)

                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.rep(-1, K*K).reshape((K,K))
                    matchtab[0:cn1, 0:K] = pd.crosstab(known, cluster, rownames=['known'], colnames=['init'])
                    #Rownames here
                    matchit = np.repeat(0, K)

                    while(max(matchtab.to_numpy().flatten())>0):
                        #TODO figure out how to match the R dimensions
                        ij = int(np.where(matchtab == max(matchtab.to_numpy().flatten())))
                        ik = np.where(matchtab == max(matchtab.to_numpy()[:,ij]))
                        matchit[ij] = np.repeat(-1, K)
                        matchtab[:,ij] = np.repeat(-1, K)
                        matchtab[int(ik)] = np.repeat(-1, K)

                    matchit[np.where(matchit == 0)] = np.where(np.arange(0, K, 1) != np.unique(matchit))
                    #TODO pass by reference may not let this work correctly
                    knew = cluster
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.array(np.where(cluster == i), i)] = 1

                case "random":
                    rangen = np.random.default_rng()
                    t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size = n)
                    compteur = 1

                    #sum columns
                    while(min(np.sum(t, axis=0)) < 1 and compteur + 1 < 5):
                        compteur += 1
                        t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size=n)

                    if(min(np.sum(t, axis=0)) < 1):
                        print("Random initialization failed (n too small)")
                        return None
                    
                    #if clas > 0:
                    cluster = np.argmax(t, axis=0)

                    while(max(matchtab.to_numpy().flatten())>0):
                        #TODO figure out how to match the R dimensions
                        ij = int(np.where(matchtab == max(matchtab.to_numpy().flatten())))
                        ik = np.where(matchtab == max(matchtab.to_numpy()[:,ij]))
                        matchit[ij] = np.repeat(-1, K)
                        matchtab[:,ij] = np.repeat(-1, K)
                        matchtab[int(ik)] = np.repeat(-1, K)

                    matchit[np.where(matchit == 0)] = np.where(np.arange(0, K, 1) != np.unique(matchit))
                    #TODO pass by reference may not let this work correctly
                    knew = cluster
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.array(np.where(cluster == i), i)] = 1

        else:
            t = np.ones(shape = (n, 1))
            tw = np.ones(shape = (n, 1))

        #if clas > 0:
            t = unkno*t
            
            for i in range(0, n):
                if kno[i] == 1:
                    t[i, known[i]] = 1

        #R uses rep.int here: does it matter?
        nux = np.rep(dfstart, K)
        #call to init function here
        #initx = ._T_funhddt_init
        
        #call to twinit function here
        tw = _T_funhddt_twinit

        #I indexes lists later on, should it be -1? it is 0 in R
        I = 0
        #Check if 
        likely = []
        test = np.Inf

        while(I <= itermax and test >= eps):
            if K > 1:
                #does t have a NaN/None?
                if(np.isnan(np.sum(np.asarray(t, float)))):
                   print("Unknown error: NA in t_ik")
                   return None
                
                #try numpy any
                #does t have column sums less than min_individuals?
                #if (any(npsum(np.where(t>1/K,t,0), axis=0) < min_individuals))
                if(len(np.where(np.sum(np.where(t>1/K, t, 0), axis=0) < min_individuals)[0]) > 0):
                    return "pop<min_individuals"
            
            #m_step1 called here
            m = _T_funhddt_m_step1(fdobj, wlist, K, t, tw, nux, dfupdate, dfconstr, model, threshold, method, noise_ctrl, com_dim, d_max, d_set)

            nux = m['nux']
            
            #e_step1 called here
            to = _T_funhddt_e_step1(fdobj, wlist, K, t, tw, nux, dfupdate, dfconstr, model, threshold, method, noise_ctrl, com_dim, d_max, d_set)

            
            L = to['L']
            t = to['t']
            tw = to['tw']
            

            #likely[I] = L in R. Is there a reason why we would need NAs?
            likely.append(L)

            #TODO I-1 and I-2 adjusted for Python indicies
            if(I == 2):
                test = abs(likely[I] - likely[I-1])
            elif I > 2:
                lal = (likely[I] - likely[I-1])/(likely[I-1] - likely[I-2])
                lbl = likely[I-1] + (likely[I] - likely[I-1])/(1.0/lal)
                test = abs(lbl - likely[I-1])
            
            I += 1

        #a
        if np.isin(model, np.array(["AKBKQKDK", "AKBQKDK", "AKBKQKD", "AKBQKD"])):
            a = _Table(data = m['a'][:,0], rownames=["Ak:"], colnames=np.arange(0,m['K']))

        #find Python paste equivalent
        #Solution: for loop
        elif model == "AJBQD":
            colnamesA1 = []
            for i in range(0, m['d'][0]):
                colnamesA1.append('a' + str(i))
            a = _Table(data = m['a'][0], rownames= ['Aj:'], colnames = colnamesA1)
        
        elif np.isin(model, np.array(["ABKQKDK", "ABQKDK", "ABKQKD", "ABQKD", "ABQD"])):
            a = _Table(data = m['a'][0], rownames=['A:'], colnames=[''])

        else:
            colnamesA2= []
            for i in range(0, np.max(m['d'])):
                colnamesA2.append('a' + str(i))
            a = _Table(data = m['a'], rownames=np.arange(0, m['K']), colnames=colnamesA2)


        #b
        if np.isin(model, np.array(["AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBQKD", "ABQKD", "AJBQD", "ABQD"])):
            b = _Table(data = m['b'][0], rownames=["B:"], colnames=[''])
        else:
            b = _Table(data = m['b'], rownames=["Bk:"], colnames=np.arange(0, m['K']))

        d = _Table(m['d'], rownames=["dim:"], colnames=np.arange(0, m['K']))

        colnamesmu = []
        for i in range(0, p):
            colnamesmu.append("V" + str(i))

        mu = _Table(m['mu'], rownames=np.arange(0, m['K']), colnames=colnamesmu)

        prop = _Table(m['prop'], rownames=[''], colnames=np.arange(0, m['K']))
        nux = _Table(m['nux'], rownames=[''], colnames=np.arange(0, m['K']))

        complexity = _T_hdc_getComplexityt(m, p, dfconstr)
        #TODO class here
        cls = np.argmax(t, axis=0)


        converged = test < eps

        params = {'params': params, 'wlist': wlist, 'model':model, 'K':K, 'd':d,
                  'a':a, 'b':b, 'mu':mu, 'prop':prop, 'nux':nux, 'ev': m['ev'],
                  'Q': m['Q'], 'Q1':m['Q1'], 'fpca': m['fpcaobj'], 
                  'loglik':likely[-1], 'loglik_all': likely, 'posterior': t,
                  'class': cls, 'com_ev': com_ev, 'n':n, 'complexity':complexity,
                  'threshold': threshold, 'd_select': method, 
                  'converged': converged, "index": test_index}
        
        bic_icl = _T_hdclassift_bic(params, p, dfconstr)
        params['BIC'] = bic_icl["bic"]
        params["ICL"] = bic_icl['icl']

        #TODO class here

        return params
        
def _T_initmypca_fd1(fdobj, Wlist, Ti):
    
    #TODO add multivariate
    mean_fd = fdobj
    #TODO does pass by reference happen here like with lists?
    coef = fdobj.coefficients
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

# Why not just pass in x instead of fdobj?
def _T_funhddt_twinit(fdobj, wlist, par, nux):

    #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis or type(fdobj) == skfda.FDataGrid):
        x = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    if type(fdobj) == dict or type(fdobj) == pd.DataFrame:
        #Multivariate
        if len(x.keys() > 1):
            x = fdobj[0].coefficients
            for i in range(0, len(fdobj)):
                x = np.c_[x, fdobj[f'{i}'].coefficients]
        #univariate
        else:
            x = fdobj.coeficients

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
        Qk = Q1['i']

        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:d[i]],np.repeat(1/b[i], p-d[i]) ))))
        muki = mu[i]

        wki = wlist['W_m']
        mah_pen[:,i] = _T_imahalanobis(x, muki, wki, Qk, aki)
        W[:, i] = (nux[i] + p) / (nux[i] + mah_pen[:, i])

    return W


# In R, this function doesn't return anything?

def _T_funhddt_e_step1(fdobj, Wlist, par, clas=0, known=None, kno=None):
    
    #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis or type(fdobj == skfda.FDataGrid)):
        x = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    if type(fdobj) == dict or type(fdobj) == pd.DataFrame:
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
    K_pen = K_pen + np.atleast_2d(kcon).T
    num = np.exp(K_pen)
    t = num / np.atleast_2d(np.sum(num, axis=1)).T

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

    return {t: t, tw: tw, L: L}



def _T_funhddt_m_step1(fdobj, Wlist, K, t, tw, nux, dfupdate, dfconstr, model, 
                       threshold, method, noise_ctrl, com_dim, d_max, d_set):
    
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

    N = x.shape[0]
    p = x.shape[1]
    n = np.sum(t, axis=0)
    prop = n/N
    #matrix with K columns and p rows
    mu = np.repeat(None, K*p).reshape((K, p))
    mu1 = np.repeat(None, K*p).reshape((K, p))

    #This is in R code but just gets overwritten later
    #corX = np.repeat(0, N*K).reshape((N, K))

    #TODO Verify if this is matrix multiplication (better to be safe using matmul)
    corX = t*tw

    for i in range(0, K):
        #Verify calculation in apply
        mu[i] = np.apply_along_axis(sum,1,(np.matmul(corX[:,i], np.repeat(1, p)).T)*(x.T))/np.sum(corX[:,i])
        mu1[i] = np.sum(corX[:,i]*x, axis=0)/np.sum(corX[:,i])

    ind = np.apply_along_axis(np.where, 0, t>0)
    
    n_bis = np.arange(0,K)
    for i in range(0,K):
        #verify this is the same in R code. Should be, since [[i]] acceses the list item i
        n_bis[i] = len(ind[f'{i}'])

    match dfupdate:
        case "approx":
            #TODO add try/catch to this like in R
            jk861 = _T_tyxf8(dfconstr, nux, n, t, tw, K, p, N)
            testing = jk861
            if np.all(np.isfinite(testing)):
                nux = jk861
        
        case "numeric":
            #TODO add try/catch to this like in R
            jk681 = _T_tyxf7(dfconstr, nux, n, t, tw, K, p, N)
            testing = jk681
            if np.all(np.isfinite(testing)):
                nux = jk861


    traceVect = np.zeros(K)

    ev = np.repeat(0, K*p).reshape((K,p))
    #try dictionary here
    Q = {}
    fpcaobj = {}

    for i in range(0, K):
        donnees = _T_mypcat_fd1(fdobj, Wlist, t[:,i], corX[:,i])
        #What is the context for the diag call in R? (ie. what data type is diag being called on)
        traceVect[i] = sum(np.diag(donnees['valerus_propres']))
        ev[i] = donnees['valerus_propres']
        Q[f'{i}'] = donnees['U']
        fpcaobj[f'{i}'] = donnees


    if model in ["AJBQD", "ABQD"]:
        d = np.repeat(com_dim, K)

    elif model in ["AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD"]:
        #rep using each=K gives the same result as np.repeat normally when repeating K times
        dmax = np.min(np.apply_along_axis(np.argmax, 1, (ev>noise_ctrl)*np.repeat(np.arange(0, ev.shape[1]), K)))
        if com_dim > dmax:
            com_dim = max(dmax, 1)
        d = np.repeat(com_dim, K)
    else:
        d = _T_hdclassif_dim_choice(ev, n, method, threshold, False, noise_ctrl, d_set)

    Q1 = Q
    for i in range(0, K):
        # verify that in R, matrix(Q[[i]]... ) just constructs a matrix with same dimenstions as Q[[i]]...
        Q[f'{i}'] = Q[f'{i}'][:,0:d[i]]

    ai = np.repeat(None, K*np.max(d))
    if model in ['AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD']:
        for i in range(0, K):
            ai[i, 0:d[i]] = ev[i, 0:d[i]]

    elif model in ['AKBKQKDK', 'AKBQKDK', 'AKBKQKD', 'AKBQKD']:
        for i in range(0, K):
            ai[i] = np.repeat(np.sum(ev[i, 0:d[i]]/d[i]), np.max(d))

    elif model == 'AJBQD':#LINE 1256
        print("notdone")


#degrees of freedom functions modified yxf7 and yxf8 functions
# /*
# * Authors: Andrews, J. Wickins, J. Boers, N. McNicholas, P.
# * Date Taken: 2023-01-01
# * Original Source: teigen (modified)
# * Address: https://github.com/cran/teigen
# *
# */
def _T_tyxf7(dfconstr, nux, n, t, tw, K, p, N):
    newnux = np.zeros(3)
    if dfconstr == "no":
        dfoldg = nux.copy()

        #scipy digamma is slow? https://gist.github.com/timvieira/656d9c74ac5f82f596921aa20ecb6cc8
        for i in range(0, K):
            constn = 1 + (1/n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log((dfoldg[i] + p)/2)
            temp = scip.digamma((dfoldg[i] + p)/2)
            
            f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
            newnux[i] = scio.brentq(f, 0.0001, 1000, xtol=0.00001)

            if newnux[i] > 200:
                newnux[i] = 200.

            if newnux[i] < 2:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]
        constn = 1 + (1/N) * np.sum(t *(np.log(tw) - tw)) + scip.digamma( (dfoldg + p) / 2) - np.log( (dfoldg + p) / 2)

        print(constn)
        f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
        dfsamenewg = scio.brentq(f, 0.0001, 1000, xtol=0.01)

        if dfsamenewg > 200:
            dfsamenewg = 200.
        
        if dfsamenewg < 2:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

def _T_tyxf8(dfconstr, nux, n, t, tw, K, p, N):
    newnux = np.zeros(3)

    if(dfconstr == "no"):
        dfoldg = nux.copy()
        
        for i in range(0, K):
            constn = 1 + (1 / n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log( (dfoldg[i] + p)/2)
            constn = -constn
            newnux[i] = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg[i] / 2)) - ( (dfoldg[i]/2) - (1/2)))) / (1 - np.exp(constn))

            if newnux[i] > 200:
                newnux[i] = 200.

            if newnux[i] < 2:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]
        constn = 1 + (1 / N) * np.sum(t * (np.log(tw) - tw)) + scip.digamma((dfoldg + p)/2) - np.log( (dfoldg + p)/2)
        constn = -constn

        dfsamenewg = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg / 2)) - ( (dfoldg/2) - (1/2)))) / (1 - np.exp(constn))

        if dfsamenewg > 200:
            dfsamenewg = 200.

        if dfsamenewg < 2:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

def _T_mypcat_fd1(fdobj, Wlist, Ti, corI):

    mean_fd = fdobj

    coefmean = np.apply_along_axis(np.sum, axis=0, arr=(corI.T)@np.repeat(1,fdobj.coefficients.shape[1]))/ np.sum(corI)

    fdobj.coefficients = np.apply_along_axis(lambda col: col - coefmean, axis=0, arr=fdobj.coefficients)
    mean_fd.coefficients = {'mean': coefmean}
    coef = fdobj.coefficients.T
    temp = _T_repmat(np.sqrt(corI), n=coef.T.shape[0], p=1) * coef.T
    mat_cov = np.inner(temp, temp) / np.sum(Ti)
    cov = (Wlist['W_m']@ mat_cov)@(Wlist['W_m'].T)

    valeurs = np.linalg.eig(cov)
    valeurs_propres = valeurs.eigenvalues
    vecteurs_propres = valeurs.eigenvectors
    fonctionspropres = fdobj
    bj = np.linalg.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@vecteurs_propres
    fonctionspropres['coefficients'] = bj

    scores = skfda.misc.inner_product_matrix(fdobj, fonctionspropres)
    varprop = valeurs_propres / np.sum(valeurs_propres)
    pcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd}

    return pcafd

def _T_hddc_ari(x, y):
    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    tab = pd.crosstab(x, y)
    if np.all(tab.shape == (1,1)): return 1
    a = np.sum(scip.binom(tab, 2))
    b = np.sum(scip.binom(np.sum(tab, axis=1), 2)) - a
    c = np.sum(scip.binom(np.sum(tab, axis=0), 2)) - a
    d = scip.binom(np.sum(tab), 2) - a - b - c
    ari = (a - (a + b) * (a + c)/(a+b+c+d))/(a+b+a+c)/2 - (a+b) * (a + c)/(a+b+c+d)
    return ari

def _T_hdclassift_bic(par, p, dfconstr):
    #mux and mu not used, should we get rid of them?
    model = par['model']
    K = par['K']
    d = par['d']
    b = par['b']
    a = par['a']
    mu = par['mu']
    N = par['N']
    prop = par['prop']
    mux = par['mux']

    if len(b) == 1:
        eps = np.sum(prop*d)
        #get ncol from ev
        n_max = par['ev'].shape[1]
        b = b*(n_max-eps) / (p-eps)
        b = np.tile(b, K)

    if len(a) == 1:
        #repeat single element
        a = np.repeat(a, K*max(d)).reshape((K, max(d)))

    elif len(a) == K:
        #Repeat vector column-wise
        a = np.tile(a, K*max(d)).reshape((K, max(d))).T

    elif model == "AJBQD":
        #repeat vector row-wise
        a = np.tile(a, K*d[0]).reshape((K, d[1]))

    if np.nanmin(a) <= 0 or np.any(b < 0):
        return - np.Inf
    
    if np.isnan(par['loglik']):
        som_a = np.zeros(K)

        for i in range(K):
            som_a[i] = np.sum(np.log(a[i, 0:d[i]]))
        L = -(1/2)*np.sum(prop * (som_a + (p-d) * np.log(b) - 2 * np.log(prop) + p * (1 + np.log(2*np.pi))))*N

    else:
        L = par['loglik'][len(par['loglik'])]

    if dfconstr == 'no':
        ro = K*(p+1)+K-1
    else:
        ro = K*p+K
    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0]
    to = d*(p-(d+1)/2)

    if model == 'AKJBKQKDK':
        m = ro + tot + D + K
    elif model == "AKBKQKDK":
        m = ro + tot + 2*K
    elif model == "ABKQKDK":
        m = ro + tot + K + 1
    elif model == "AKJBQKDK":
        m = ro + tot + D + 1
    elif model == "AKBQKDK":
        m = ro + tot + K + 1
    elif model == "ABQKDK":
        m = ro + tot + 2

    bic = - (-2*L + m * np.log(N))

    t = par['posterior']

    Z = ((t - np.apply_along_axis(np.max, t, axis=0)) == 0) + 0
    icl = bic - 2*np.sum(Z*np.log(t + 1.e-15))

    return {'bic': bic, 'icl': icl}

def _T_hdc_getComplexityt(par, p, dfconstr):
    model = par['model']
    K = par['K']
    d = par['d']
    b = par['b']
    a = par['a']
    mu = par['mu']
    prop = par['prop']

    if dfconstr == 'no':
        ro = K*(p+1)+K - 1

    else:
        ro = K*p + K

    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0]
    to = d*(p-(d+1)/2)

    if model == 'AKJBKQKDK':
        m = ro + tot + D + K

    elif model == "AKBKQKDK":
        m = ro + tot + 2*K

    elif model == "ABKQKDK":
        m = ro + tot + K + 1

    elif model == "AKJBQKDK":
        m = ro + tot + D + 1

    elif model == "AKBQKDK":
        m = ro + tot + K + 1

    elif model == "ABQKDK":
        m = ro + tot + 2

    return m

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
