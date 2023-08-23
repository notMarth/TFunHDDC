#Required Libraries------------------------------------------------------------#
import skfda
from sklearn import cluster as clust
from sklearn.model_selection import ParameterGrid
from sklearn.utils import Bunch
import numpy as np
import warnings
import pandas as pd
import multiprocessing as multi
import time
from scipy import linalg as scil
import scipy.special as scip
import scipy.optimize as scio
import math
from shutil import get_terminal_size
#------------------------------------------------------------------------------#

#GLOBALS
LIST_TYPES = (list, np.ndarray)
UNKNOWNS = (np.NaN, np.inf, -np.inf, None)
INT_TYPES = (int, np.integer)
FLOAT_TYPES = (float, np.floating)
NUMERIC_TYPES = (INT_TYPES, FLOAT_TYPES)

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


class _Table:
    """
    Store data with row names and column names
    
    Attributes
    ----------
    data : numpy array
    rownames : numpy array
    colnames : numpy array

    
    """
    def __init__(self, data=None, rownames=None, colnames=None):
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

    def __str__(self):
        string = f'\t{self.colnames}\n'
        for i in range(len(self.data)):
            try:
                string += f'{self.rownames[i]}\t{self.data[i]}\n'
            except:
                string += f'\t{self.data[i]}\n'

        return string

    def __len__(self):
        return len(self.data)
    
class TFunHDDC:

    def __init__(self, Wlist, model, K, d, a, b, mu, prop, nux, ev, Q, Q1,
                 fpca, loglik, loglik_all, posterior, cl, com_ev, N,
                 complexity, threshold, d_select, converged, index, bic, icl):
        self.Wlist = Wlist
        self.model = model
        self.K = K
        self.d = d
        self.a = a
        self.b = b
        self.mu = mu
        self.prop = prop
        self.nux = nux
        self.ev = ev
        self.Q=Q
        self.Q1 = Q1
        self.fpca = fpca
        self.loglik = loglik
        self.loglik_all = loglik_all
        self.posterior = posterior
        self.cl = cl
        self.com_ev = com_ev
        self.N = N
        self.complexity = complexity
        self.threshold=threshold
        self.d_select=d_select
        self.converged=converged
        self.index = index
        self.bic = bic
        self.icl = icl
        self.criterion = None
        self.complexity_all = None
        self.allCriteria = None
        self.allRes = None

    def predict(self, data):
        #TODO check if data Null
        #TODO figure out what .T_myCallAlerts is doing

        #univariate
        x = data.coefficients.copy()

        #TODO fix this, should be number of basis functions
        Np = 15
        p = x.shape[1]
        N = x.shape[0]
        if Np != p:
            raise ValueError("New observations should be represented using the same base as for the training set")
        a = self.a.data.copy()
        b = self.b.data.copy()
        d = self.d.data.copy()
        nux = self.nux.data.copy()
        mu = self.mu.data.copy()
        prop = self.prop.data.copy()
        b[b<1.e-6] = 1.e-6
        wki = self.Wlist['W_m']

        t = np.zeros((N, self.K))
        mah_pen = np.zeros((N, self.K))
        K_pen = np.zeros((N, self.K))
        num = np.zeros((N, self.K))
        s = np.repeat(0., self.K)

        match self.model:
            case 'AKJBKQKDK':
                for i in range(self.K):
                    s[i] = np.sum(np.log(a[i, 0:d[i]]))
                    Qk = self.Q1[f'{i}'].copy()
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = 1/a[i, 0:d[i]]
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #scipy logamma vs math lgamma?
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[i]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))

            case 'AKJBQKDK':
                for i in range(self.K):
                    s[i] = np.sum(np.log(a[i, 0:d[i]]))
                    Qk =self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = 1/a[i, 0:d[i]]
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]
                    
                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #Copied from previous case with b[i] changed to b[1]
                    #scipy logamma vs math lgamma?
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[0]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))

            case 'AKBKQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[i])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = np.repeat(1/a[i], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #copied from AKJBKQKDK
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[i]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))


            case 'ABKQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[0])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[i], p-d[i])
                    diag1 = np.repeat(1/a[0], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #copied from AKJBKQKDK
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[i]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))

            case 'AKBQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[i])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = np.repeat(1/a[i], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #copied from AKJBKQKDK
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[0]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))

            case 'ABQKDK':
                for i in range(self.K):
                    s[i] = d[i]*np.log(a[0])
                    Qk = self.Q1[f'{i}']
                    diag2 = np.repeat(1/b[0], p-d[i])
                    diag1 = np.repeat(1/a[0], d[i])
                    aki = np.sqrt(np.diag(np.concatenate((diag1, diag2))))
                    muki = mu[i]

                    mah_pen[:, i] = _T_imahalanobis(x, muki, wki, Qk, aki)

                    #copied from AKJBKQKDK
                    K_pen[:, i] = np.log(prop[i]) + math.lgamma( (nux[i] + p) / 2) - (1/2) * (s[i] + (p-d[i]) * np.log(b[0]) - np.log(self.Wlist['dety'])) - ( ( p/2)*(np.log(np.pi) + np.log(nux[i])) + math.lgamma(nux[i] /2) + ( (nux[i] + p) / 2) * (np.log(1 + mah_pen[:, i] / nux[i])))

        kcon = -np.apply_along_axis(np.max, 1, K_pen)
        K_pen += np.atleast_2d(kcon).T
        num = np.exp(K_pen)
        t = num / np.atleast_2d(np.sum(num, axis=1)).T

        cl = np.argmax(t, axis=1)
        return {'t': t, 'class': cl}
    '''
    #Try returning models + diverged?
    def __str__(self):
        return None 
    
    #Try printing params?
    def __repr__(self):
        return None
    '''

def callbackFunc(res):
    print("Complete!")

def tfunHDDC(data, K=np.arange(1,11), model='AKJBKQKDK', known=None, dfstart=50., 
             dfupdate='approx', dfconstr='no', threshold=0.1, itermax=200, 
             eps=1.e-6, init='random', criterion='bic', d_select='cattell', 
             init_vector=None, show=True, mini_nb=[5,10], min_individuals=2,
             mc_cores=1, nb_rep=2, keepAllRes=True, kmeans_control={'n_init':1, 'max_iter':10, 'algorithm':'lloyd'}, d_max=100,
             d_range=2, verbose=True):
    
    com_dim = None
    noise_ctrl = 1.e-8

    _T_hddc_control(locals())

    model = _T_hdc_getTheModel(model, all2models=True)
    if init == "random" and nb_rep < 20:
        nb_rep = 20

    if mc_cores > 1:
        verbose = False

    BIC = []
    ICL = []

    fdobj = data.copy()

    if isinstance(fdobj, skfda.FDataBasis):
        x = fdobj.coefficients
        p = x.shape[1]

        W = skfda.misc.inner_product_matrix(fdobj.basis, fdobj.basis)
        W[W<1.e-15] = 0
        W_m = scil.cholesky(W)
        dety = scil.det(W)
        Wlist = {'W': W, 'W_m': W_m, 'dety':dety}

    else:
        x = fdobj['0'].coefficients

        for i in range(1, len(fdobj)):
            x = np.c_[x, fdobj[f'{i}'].coefficients]

        p = x.shape[1]

        W_fdobj = []
        for i in range(len(data)):
            W_fdobj.append(skfda.misc.inner_product_matrix(data[f'{i}'].basis, data[f'{i}'].basis))

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

    if not(type(threshold) == list or type(threshold) == np.ndarray):
        threshold = np.array([threshold])

    if not(type(K) == list or type(K) == np.ndarray):
        K = np.array([K])

    if len(np.unique(K)) != len(K):
        warnings.warn("The number of clusters, K, should be unique (repeated values will be removed)")
        K = np.sort(np.unique(K))

    mkt_list = {}
    if d_select == 'grid':
        #Why can't we use multiple values for K here?
        if len(K) > 1:
            raise ValueError("When using d_select='grid, K must be only one value (ie. not a list)")
        
        #take first element of K since it is a list/array
        for i in range(K[0]):
            mkt_list[f'd{i}'] = [str(d_range)]
        mkt_list.update({'model':model, 'K':[str(a) for a in K], 'threshold':[str(a) for a in threshold]})

        

    else:
        for i in range(np.max(K)):
            mkt_list[f'd{i}'] = ['2']
        mkt_list.update({'model':model, 'K':[str(a) for a in K], 'threshold':[str(a) for a in threshold]})

        
    
    mkt_expand = ParameterGrid(mkt_list)
    mkt_expand = list(mkt_expand)
    repeat = mkt_expand.copy()
    for i in range(nb_rep-1):
        mkt_expand = np.concatenate((mkt_expand, repeat))
    
    model = [a['model'] for a in mkt_expand]
    K = [int(a['K']) for a in mkt_expand]
    d = {}

    for i in range(max(K)):
        d[f'{i}'] = [a[f'd{i}'] for a in mkt_expand]
    

    #TODO can we make this more efficient with the mkts?
    #mkt_univariate = ['_'.join(list(a.values())) for a in mkt_expand]

    #Pass in dict from mkt_expand
    def hddcWrapper(mkt, verbose, start_time = 0, totmod=1):
        modelNo = mkt[1]
        mkt = mkt[0]
        model = mkt['model']
        K = int(mkt['K'])
        threshold = float(mkt['threshold'])

        d_set = np.repeat(2, K)
        for i in range(K):
            d_set[i] = int(mkt[f'd{i}'])

        #TODO may need to modify this from R
        try:
            res = _T_funhddc_main1(fdobj=fdobj, wlist=Wlist, K=K, dfstart=dfstart, dfupdate=dfupdate, dfconstr=dfconstr,
                                    itermax=itermax, model=model, threshold=threshold,
                                    method=d_select, eps=eps, init=init, init_vector=init_vector,
                                    mini_nb=mini_nb, min_individuals=min_individuals, noise_ctrl=noise_ctrl,
                                    com_dim=com_dim, kmeans_control=kmeans_control, d_max=d_max, d_set=d_set, known=None)
            
            if verbose:
                _T_estimateTime(stage=modelNo, start_time=start_time, totmod=totmod)

        except Exception as e:
            raise e
        

        return res
    

    nRuns = len(mkt_expand)
    if nRuns < mc_cores:
        mc_cores = nRuns

    max_cores = multi.cpu_count()
    if mc_cores > max_cores:
        warnings.warn(f"mc_cores was set to a value greater than the maximum number of cores on this system.\nmc_cores will be set to {max_cores}")
        mc_cores = max_cores

    start_time = time.process_time()

    if mc_cores == 1:
        if verbose:
            _T_estimateTime("init")
            #Add model numbers if we are tracking time
            mkt_expand = np.c_[mkt_expand, np.arange(0, len(mkt_expand))]

        res = [hddcWrapper(a, verbose, start_time, len(mkt_expand)) for a in mkt_expand]

    else:
        try:
            p = multi.Pool(mc_cores)
            '''
            params = [(x, verbose, start_time, len(mkt_expand)) for x in mkt_expand]
            with p:
                res = p.starmap_async(dec, params).get()
            '''
            models = [mkt['model'] for mkt in mkt_expand]
            Ks = [int(mkt['K']) for mkt in mkt_expand]
            thresholds = [float(mkt['threshold']) for mkt in mkt_expand]
            d_sets = []
            for i in range(len(Ks)):
                d_temp = []
                for j in range(Ks[i]):
                    d_temp.append(int(mkt_expand[i][f'd{j}']))
                d_sets.append(d_temp)
            
            with p:
                params = [(fdobj, Wlist, Ks[i], dfstart, dfupdate, dfconstr, models[i], itermax, thresholds[i], d_select, eps, init, init_vector, mini_nb, min_individuals, noise_ctrl, com_dim, kmeans_control, d_max, d_sets[i], None) for i in range(len(models))]
                res = p.starmap_async(_T_funhddc_main1, params, callback=callbackFunc).get()

        #TODO add descriptive text here explaining that this is an unknown error
        except Exception as e:
            raise e
    if verbose:
        mkt_expand = mkt_expand[:,0]
    res = np.array(res)
    loglik_all = np.array([x.loglik if isinstance(x, TFunHDDC) else - np.Inf for x in res])
    comment_all = np.array(["" if isinstance(x, TFunHDDC) else x for x in res])

    threshold = np.array([float(x['threshold']) for x in mkt_expand])

    if np.all(np.invert(np.isfinite(loglik_all))):
        warnings.warn("All models diverged")

        #TODO do we need to add allcriteria here?

        return {'model': model, 'K': K, 'threshold':threshold, 'LL':loglik_all, 'BIC': None, 'comment': comment_all}

    n = len(mkt_expand)
    uniqueModels = mkt_expand[:int(n/nb_rep)]
    #For each param combo pick best run based on log liklihood
    #TODO can we simplify the rest of this with something like a dictionary comprehension?
    #mkt_expand has the first len(mkt_expand)/nb_rep entries unique
    #What if multiple of the same params gives the same loglik?
    #modelKeep = [(np.argmax(loglik_all[np.nonzero(uniqueModels[x] == mkt_expand)[0]])*len(uniqueModels)) + x for x in range(len(uniqueModels))]
    
    modelKeep = np.arange(0, len(res))

    # modelCheck = [isinstance(result, TFunHDDC) for result in modelKeep]
    # if len(modelCheck) == 0:
    #     return "All models Diverged"
    
    # modelKeep = np.array(modelKeep)[modelCheck]

    loglik_all = loglik_all[modelKeep]
    comment_all = comment_all[modelKeep]
    chosenRes = res[modelKeep]    
    
    bic = [res.bic if isinstance(res, TFunHDDC) else -np.Inf for res in chosenRes]
    icl = [res.icl if isinstance(res, TFunHDDC) else -np.Inf for res in chosenRes]
    allComplex = [res.complexity if isinstance(res, TFunHDDC) else -np.Inf for res in chosenRes]
    model = np.array(model)[modelKeep]
    threshold = np.array(threshold)[modelKeep]
    K = np.array(K)[modelKeep]
    d_keep = {}
    for i in range(np.max([int(x) for x in K])):
        d_keep[f'{i}'] = np.array([int(x[f'd{i}']) for x in np.array(mkt_expand)[modelKeep]])

    CRIT = bic if criterion == 'bic' else icl
    resOrdering = np.argsort(CRIT)[::-1]

    qui = np.nanargmax(CRIT)
    bestCritRes = chosenRes[qui]
    bestCritRes.criterion = criterion
    #is complexity all really necessary?
    bestCritRes.complexity_all = [('_'.join(mkt_expand[modelKeep[i]].values()), allComplex[i]) for i in range(len(mkt_expand))]
    if show:
        if n > 1:
            print("tfunHDDC: \n")

        printModel = np.array([x.rjust(max([len(a) for a in model])) for x in model])
        printK = np.array([str(x).rjust(max([len(str(a)) for a in K])) for x in K])
        printTresh = np.array([str(x).rjust(max([len(str(a)) for a in threshold])) for x in threshold])
        resout = np.c_[printModel[resOrdering], printK[resOrdering], printTresh[resOrdering], _T_addCommas((np.array(bestCritRes.complexity_all)[resOrdering])[:,1].astype(float)), _T_addCommas(np.array(CRIT)[resOrdering])]
        resout = np.c_[np.arange(1,len(mkt_expand)+1), resout]

        resout = np.where(resout != "-inf", resout, 'NA')
        if np.any(np.nonzero(comment_all != '')[0]): 
            resout = np.c_[resout, comment_all[resOrdering]]
            resPrint = pd.DataFrame(data = resout[:,1:], columns=['Model', 'K', 'Threshold', 'Complexity', criterion.upper(), 'Comment'], index=resout[:, 0])
        else:
            resPrint = pd.DataFrame(data = resout[:,1:], columns=['Model', 'K', 'Threshold', 'Complexity', criterion.upper()], index=resout[:, 0])

        print(resPrint)
        print(f'\nSelected model {bestCritRes.model} with {bestCritRes.K} clusters')
        print(f'\nSelection Criterion: {criterion}\n')

    allCriteria = resPrint
    bestCritRes.allCriteria=allCriteria
    #allcriteria goes here. Do we need it?

    if keepAllRes:
        allRes = chosenRes
        bestCritRes.allRes=allRes
    #allresults goes here. Do we need it?

    #R assigns threshold here. Does it not already get assigned during main1?

    return bestCritRes

#TODO add default values
#*args argument replaces ... in R code
#fdobj should either be a FDataBasis object or a dictionary of FDataBasis objects
def _T_funhddc_main1(fdobj, wlist, K, dfstart, dfupdate, dfconstr, model,
                     itermax, threshold, method, eps, init, init_vector,
                     mini_nb, min_individuals, noise_ctrl, com_dim,
                     kmeans_control, d_max, d_set, known, *args):
    modelNames = ["AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", 
                  "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD",
                  "AKBQKD", "ABQKD"]
    #Univariate
    if(type(fdobj) == skfda.FDataBasis):
        data = fdobj.coefficients

    
    if type(fdobj) == dict:
        #Multivariate
        if len(fdobj.keys() > 1):
            data = fdobj['0'].coefficients
            for i in range(1, len(fdobj)):
                data = np.c_[data, fdobj[f'{i}'].coefficients]
        #univariate in dict
        else:
            data = fdobj['0'].coeficients

    n, p = data.shape
    #com_ev is None (better way of phrasing) instead of = None
    com_ev = None

    d_max = min(n,p,d_max)

    #classification

    if(known is None):
        clas = 0
        kno = None
        test_index = None

    else:

        if len(known) != n:
            raise ValueError("Known classifications vector not the same length as the number of samples (see help file)")
        
        #TODO find out how to handle missing values. For now, using numpy NaN

        #Known should use None when data is not known
        #np.isnan(np.sum) is faster than np.isnan(np.min) for the general case
        else:
            if (not np.isnan(np.sum(known))):
                warnings.warn("No Nans in 'known' vector supplied. All values have known classification (parameter estimation only)")

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
                training = np.where((np.invert(np.isnan(known))))
                test_index = training
                kno = np.zeros(n).astype(int)
                kno[test_index] = 1
                unkno = np.atleast_2d((kno - 1)*(-1))

                #clas = len(training)/n
            #why do the evaluations to clas if its set to 1 here?
            clas = 1

    if K > 1:
        t = np.zeros((n, K))
        tw = np.zeros((n, K))

        match init:
            case "vector":
                if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    #ensure this is the correct size. In R: matchtab[1:cn1, 1:K]
                    temp = pd.crosstab(index = known, columns = init_vector).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab[0:cn1, 0:K] = temp.to_numpy()
                    table = _Table(data=matchtab, rownames=temp.index, colnames = temp.columns)
                    
                    #___OLD___
                    #matchtab[0:cn1, 0:K] = pd.crosstab(index = known, columns = init_vector, rownames = ["known"], colnames = ["init"]) 
                    
                    #numpy.where always returns a tuple of at least length 1 (useless since we're not multidim here),
                    #so we need to access the zeroth element
                    #np.isin behaves identical to %in% in R
                    rownames = np.concatenate(([table.rownames, np.nonzero(np.invert(np.isin(np.arange(0,K,1), np.unique(known[test_index]))))[0]]))
                    table.rownames = rownames
                    matchit = np.repeat(-1, K)

                    while(np.max(table.data)>0):
                        ij = int(np.nonzero(table.data == np.max(table.data))[1][0])

                        ik = np.argmax(table.data[:,ij])
                        matchit[ij] = table.rownames[ik]
                        table.data[:,ij] = np.repeat(-1, K)
                        table.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    initnew = init_vector.copy()
                    for i in range(0, K):
                        initnew[init_vector == i] = matchit[i]

                    init_vector = initnew

                for i in range(0, K):
                    t[np.nonzero(init_vector == i)[0], i] = 1


            case "kmeans":
                kmc = kmeans_control
                km = clust.KMeans(n_clusters = K, max_iter = kmeans_control['max_iter'], n_init = kmeans_control['n_init'], algorithm=kmeans_control['algorithm'])
                cluster = km.fit_predict(data)

                if clas > 0:
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data = matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                for i in range(0, K):
                    t[np.nonzero(cluster == i)[0], i] = 1.

            #skip trimmed kmeans
            case "tkmeans":
                raise ValueError("tkmeans not supported")

            case "mini-em":
                prms_best = 1
                for i in range(0, mini_nb[0]):
                    prms = _T_funhddc_main1(fdobj=fdobj, wlist=wlist, K=K, known=known, dfstart=dfstart,
                                            dfupdate=dfupdate, dfconstr=dfconstr, model=model,
                                            threshold=threshold, method=method, eps=eps,
                                            itermax=mini_nb[1], init_vector=0, init="random",
                                            mini_nb=mini_nb, min_individuals=min_individuals,
                                            noise_ctrl=noise_ctrl, kmeans_control=kmeans_control,
                                            com_dim=com_dim, d_max=d_max, d_set=d_set)
                    if isinstance(prms, TFunHDDC):
                        if not isinstance(prms_best, TFunHDDC):
                            prms_best = prms
                        
                        elif prms_best.loglik < prms.loglik:
                            prms_best = prms

                #TODO figure out better way to signal that mini-em didn't converge
                if not isinstance(prms, TFunHDDC):
                    return "mini-em did not converge"
                
                t = prms_best.posterior

                if clas > 0:
                    cluster = np.argmax(t, axis=1)

                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data = matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.nonzero(cluster == i)[0], i] = 1.

            case "random":
                rangen = np.random.default_rng()
                t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size = n)
                compteur = 1

                #sum columns
                while(np.min(np.sum(t, axis=0)) < 1 and compteur + 1 < 5):
                    compteur += 1
                    t = rangen.multinomial(n=1, pvals=np.repeat(1/K, K), size=n)

                if(np.min(np.sum(t, axis=0)) < 1):
                    raise ValueError("Random initialization failed (n too small)")
                if clas > 0:
                    cluster = np.argmax(t, axis=1)
                    cn1 = len(np.unique(known[test_index]))
                    matchtab = np.repeat(-1, K*K).reshape((K,K))
                    matchtab = _Table(data=matchtab)
                    temptab = pd.crosstab(known, cluster, rownames=['known'], colnames=['init']).reindex(columns=np.arange(K), index=np.arange(cn1))
                    matchtab.data[0:cn1, 0:K] = temptab.to_numpy()
                    matchtab.rownames = np.concatenate((temptab.index, np.nonzero(np.invert(np.isin(np.arange(K), np.unique(known[test_index]))))[0]))
                    matchtab.colnames = temptab.columns
                    matchit = np.repeat(-1, K)

                    while(np.max(matchtab.data)>0):
                        ij = int(np.nonzero(matchtab.data == np.max(matchtab.data))[1][0])

                        ik = np.argmax(matchtab.data[:,ij])
                     
                        matchit[ij] = matchtab.rownames[ik]
                        matchtab.data[:,ij] = np.repeat(-1, K)
                        matchtab.data[ik] = np.repeat(-1, K)

                    matchit[np.nonzero(matchit == -1)[0]] = np.nonzero(np.invert(np.isin(np.arange(K), np.unique(matchit))))[0]
                    #TODO pass by reference may not let this work correctly
                    knew = cluster.copy()
                    for i in range(0, K):
                        knew[cluster == i] = matchit[i]
                    
                    cluster = knew

                    for i in range(0, K):
                        t[np.nonzero(cluster == i)[0], i] = 1.


    else:
        t = np.ones(shape = (n, 1))
        tw = np.ones(shape = (n, 1))

    if clas > 0:
        t = unkno.T*t
        
        for i in range(0, n):
            if kno[i] == 1:
                t[i, int(known[i])] = 1.


    #R uses rep.int here: does it matter? (shouldn't)
    nux = np.repeat(dfstart, K)

    #call to init function here
    initx = _T_funhddt_init(fdobj, wlist, K, t, nux, model, threshold, method, noise_ctrl, com_dim, d_max, d_set)
    
    #call to twinit function here
    tw = _T_funhddt_twinit(fdobj, wlist, initx, nux)

    #Start I at 0, likely compared on second iteration when I == 1
    I = 0
    #Check if 
    likely = []
    test = np.Inf
    # I <= itermax means I + 1 iterations when starting at 0
    while(I < itermax and test >= eps):
        if K > 1:
            #does t have a NaN/None?
            if(np.isnan(np.sum(t))):
                return "t matrix contatins NaNs/Nones"
            
            #try numpy any
            #does t have column sums less than min_individuals?
            #if (any(npsum(np.where(t>1/K,t,0), axis=0) < min_individuals))
            if(np.any(np.sum(t>(1/K), axis=0) < min_individuals)):
                return "pop<min_individuals"
        #m_step1 called here
        m = _T_funhddt_m_step1(fdobj, wlist, K, t, tw, nux, dfupdate, dfconstr, model, threshold, method, noise_ctrl, com_dim, d_max, d_set)
        nux = m['nux'].copy()

        #e_step1 called here
        to = _T_funhddt_e_step1(fdobj, wlist, m, clas, known, kno)

        
        L = to['L']
        t = to['t']
        tw = to['tw']

        
        

        #likely[I] = L in R. Is there a reason why we would need NAs?
        likely.append(L)

        if(I == 1):
            test = abs(likely[I] - likely[I-1])
        elif I > 1:
            lal = (likely[I] - likely[I-1])/(likely[I-1] - likely[I-2])
            lbl = likely[I-1] + (likely[I] - likely[I-1])/(1.0/lal)
            test = abs(lbl - likely[I-1])
        
        I += 1

    #a
    if np.isin(model, np.array(["AKBKQKDK", "AKBQKDK"])):
        a = _Table(data = m['a'][:,0], rownames=["Ak:"], colnames=np.arange(0,m['K']))

    
    elif np.isin(model, np.array(["ABKQKDK", "ABQKDK"])):
        a = _Table(data = m['a'][0], rownames=['A:'], colnames=[''])

    else:
        colnamesA2= []
        for i in range(0, np.max(m['d'])):
            colnamesA2.append(f'a{i}')
        a = _Table(data = m['a'], rownames=np.arange(0, m['K']), colnames=colnamesA2)


    #b
    if np.isin(model, np.array(["AKJBQKDK", "AKBQKDK", "ABQKDK"])):
        b = _Table(data = np.array([m['b'][0]]), rownames=["B:"], colnames=[''])
    else:
        b = _Table(data = m['b'], rownames=["Bk:"], colnames=np.arange(0, m['K']))
    
    #d
    d = _Table(m['d'], rownames=["dim:"], colnames=np.arange(0, m['K']))


    #mu
    colnamesmu = []
    for i in range(0, p):
        colnamesmu.append(f"V{i}")

    mu = _Table(m['mu'], rownames=np.arange(0, m['K']), colnames=colnamesmu)

    prop = _Table(m['prop'], rownames=[''], colnames=np.arange(0, m['K']))
    nux = _Table(m['nux'], rownames=[''], colnames=np.arange(0, m['K']))

    complexity = _T_hdc_getComplexityt(m, p, dfconstr)

    cl = np.argmax(t, axis=1)

    converged = test < eps

    params = {'wlist': wlist, 'model':model, 'K':K, 'd':d,
                'a':a, 'b':b, 'mu':mu, 'prop':prop, 'nux':nux, 'ev': m['ev'],
                'Q': m['Q'], 'Q1':m['Q1'], 'fpca': m['fpcaobj'], 
                'loglik':likely[-1], 'loglik_all': likely, 'posterior': t,
                'class': cl, 'com_ev': com_ev, 'N':n, 'complexity':complexity,
                'threshold': threshold, 'd_select': method, 
                'converged': converged, "index": test_index}

    bic_icl = _T_hdclassift_bic(params, p, dfconstr)
    params['BIC'] = bic_icl["bic"]
    params["ICL"] = bic_icl['icl']

    tfunobj = TFunHDDC(Wlist=params['wlist'], model=params['model'], K=params['K'], d=params['d'], 
                        a=params['a'], b=params['b'], mu=params['mu'], prop=params['prop'], nux=params['nux'],
                        ev=params['ev'], Q=params['Q'], Q1=params['Q1'], fpca=params['fpca'],
                        loglik=params['loglik'], loglik_all=params['loglik_all'], posterior=params['posterior'],
                        cl=params['class'], com_ev=params['com_ev'], N=params['N'], complexity=params['complexity'],
                        threshold=params['threshold'], d_select=params['d_select'], converged=params['converged'], 
                        index=params['index'], bic=params['BIC'], icl=params['ICL'])
    return tfunobj

def _T_funhddt_init(fdobj, Wlist, K, t, nux, model, threshold, method, noise_ctrl, com_dim, d_max, d_set):

    #univariate case
    if(type(fdobj) == skfda.FDataBasis):
        x = fdobj.coefficients

    
    if type(fdobj) == dict or type(fdobj) == pd.DataFrame:
        #Multivariate
        if len(fdobj.keys()) > 1:
            x = fdobj['0'].coefficients.copy()
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[f'{i}'].coefficients.copy()]

    N = x.shape[0]
    p = x.shape[1]
    n = np.sum(t, axis=0)

    prop = n/N

    mu = np.repeat(0., K*p).reshape((K, p))
    #ind = np.apply_along_axis(np.where, 1, t>0)
    #n_bis = np.repeat(0., K)

    #for i in range(K):
        #n_bis[i] = len(ind[i][0])

    traceVect = np.repeat(0., K)
    ev = np.repeat(0., K*p).reshape((K, p))
    Q = {}
    fpcaobj = {}

    for i in range(K):
        donnees = _T_initmypca_fd1(fdobj, Wlist, t[:,i])

        mu[i] = donnees['mux']
        traceVect[i] = np.sum(np.diag(donnees['valeurs_propres']))
        ev[i] = donnees["valeurs_propres"]
        Q[f'{i}'] = donnees["U"]
        fpcaobj[f"{i}"] = donnees

    
    #Intrinsic dimension selection
    d = _T_hdclassif_dim_choice(ev, n, method, threshold, False, noise_ctrl, d_set)
    #adjust for Python indices
    d+=1
    #Set up Qi matrices
    Q1 = Q.copy()
    for i in range(K):
        Q[f'{i}'] = Q[f'{i}'][:, 0:d[i]]

    #a parameter
    ai = np.repeat(np.NaN, K*(np.max(d))).reshape((K, np.max(d)))
    if np.isin(model, np.array(['AKJBKQKDK', 'AKJBQKDK'])):
        for i in range(K):
            ai[i, 0:d[i]] = ev[i, 0:d[i]]

    elif np.isin(model, np.array(['AKBKQKDK', 'AKBQKDK'])):
        for i in range(K):
            ai[i] = np.repeat(np.sum(ev[i, 0:d[i]])/d[i], np.max(d))

    else:
        a = 0
        eps = np.sum(prop*(d))

        for i in range(K):
            a += (np.sum(ev[i, 0:d[i]])*prop[i])
            ai = np.full((K, max(d)), a/eps)


    
    #b parameter

    bi = np.zeros(K)
  
    if np.isin(model, np.array(['AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK'])):
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])

            bi[i] = remainEV/(p-d[i])

    else:
        b = 0
        eps = np.sum(prop*(d))

        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])

            b += (remainEV*prop[i])

        bi[0:K] = b/(min(N, p) - eps)

    return {'model': model, "K": K, 'd':d, 'a':ai, 'b':bi, 'mu':mu, 'prop':prop,
            'nux':nux, 'ev':ev, 'Q':Q, 'fpcaobj': fpcaobj, 'Q1':Q1}

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
        if not check_symmetric(cov, 1.e-12):
            ind = np.nonzero(cov - cov.T > 1.e-12)
            cov[ind] = cov.T[ind]
            

        
        valeurs_propres, vecteurs_propres = scil.eig(cov)
        #TODO This may make the program slower: see if not needed
        #indices = valeurs_propres.argsort()
        #valeurs_propres = valeurs_propres[indices[::-1]]
        #vecteurs_propres = vecteurs_propres[indices[::-1]]
        fonctionspropres = fdobj.copy()
        bj = scil.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@np.real(vecteurs_propres)
        fonctionspropres.coefficients = bj

        scores = skfda.misc.inner_product_matrix(temp.basis, fonctionspropres.basis)
        varprop = valeurs_propres / np.sum(valeurs_propres)
        ipcafd = {'valeurs_propres': np.real(valeurs_propres), 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd, 'mux': coefmean}

    #Multivariate
    else:
        mean_fd = {}
        temp = fdobj.copy()
        for i in range(len(fdobj)):

            mean_fd[f'{i}'] = temp[f'{i}'].copy()

        coef = temp['0'].coefficients
        for i in range(1, len(fdobj)):
            coef = np.c_[coef, temp[f'{i}'].coefficients.copy()]

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
            
            n_var = temp[f'{i}'].coefficients.shape[1]
            tempi.coefficients = np.apply_along_axis(lambda row: row - coefmean[(n_lead):(n_var + n_lead)], axis=1, arr=tempi.coefficients)
            mean_fd[f'{i}'].coefficients = coefmean[(n_lead):(n_var + n_lead)]

        cov = (Wlist['W_m']@mat_cov)@(Wlist['W_m'].T)
        if not check_symmetric(cov, 1.e-12):
            ind = np.nonzero(cov - cov.T > 1.e-12)
            cov[ind] = cov.T[ind]

        valeurs_propres, vecteurs_propres = scil.eig(cov)
        bj = scil.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@np.real(vecteurs_propres)
        fonctionspropres = temp['0'].copy()
        fonctionspropres.coefficients = bj
        scores = (coef@Wlist['W'])@bj

        varprop = valeurs_propres / np.sum(valeurs_propres)

        ipcafd = {'valeurs_propres': np.real(valeurs_propres), 'harmonic': fonctionspropres,
                  'scores': scores, 'covariance': cov, 'U': bj, 'varprop': varprop,
                  'meanfd': mean_fd, 'mux': coefmean}

    return ipcafd


# Why not just pass in x instead of fdobj?
def _T_funhddt_twinit(fdobj, wlist, par, nux):

    #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis):
       x = fdobj.coefficients

    #For R testing if fdobj gets passed as a dict
    #Should also work for dataframe (converts to pandas dataframe)
    #Will be changed outside of R testing so that the expected element in the
    #dict or dataframs is an FDataBasis
    if type(fdobj) == dict:
        #Multivariate
        #Here in R, the first element will be named '1'
        if len(fdobj.keys()) > 1:
            x = fdobj['0'].coefficients
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[f'{i}'].coefficients]
        #univariate
        else:
            x = fdobj['0'].coefficients

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


# In R, this function doesn't return anything?

def _T_funhddt_e_step1(fdobj, Wlist, par, clas=0, known=None, kno=None):

   #try this if fdobj is an fdata (Univariate only right now)
    if(type(fdobj) == skfda.FDataBasis):
        x = fdobj.coefficients

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
    nux = par["nux"].copy()
    a = par["a"].copy()
    b = par["b"].copy()
    mu = par["mu"].copy()
    d = par["d"].copy()
    prop = par["prop"].copy()
    Q = par["Q"].copy()
    Q1 = par["Q1"].copy()
    b[b<1e-6] = 1e-6

    if clas > 0:
        unkno = np.atleast_2d((kno-1)*(-1)).T

    t = np.repeat(0., N*K).reshape(N, K)
    tw = np.repeat(0., N*K).reshape(N, K)
    mah_pen = np.repeat(0., N*K).reshape(N,K)
    K_pen = np.repeat(0., N*K).reshape(N,K)
    ft = np.repeat(0., N*K).reshape(N,K)

    s = np.repeat(0., K)

    for i in range(0,K):
        s[i] = np.sum(np.log(a[i, 0:int(d[i])]))

        Qk = Q1[f"{i}"]
        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:int(d[i])],np.repeat(1/b[i], p-int(d[i])) ))))
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
   
    K_pen = K_pen + np.atleast_2d(kcon).T
    num = np.exp(K_pen)
    t = num / np.atleast_2d(np.sum(num, axis=1)).T
    #TODO might speed up a bit if we use num variable here
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
                t[i, int(known[i])] = 1

    #Nothing is returned here in R
    # Return this for testing purposes
    return {'t': t, 'tw': tw, 'L': L}



def _T_funhddt_m_step1(fdobj, Wlist, K, t, tw, nux, dfupdate, dfconstr, model, 
                       threshold, method, noise_ctrl, com_dim, d_max, d_set):

    #Univariate Case    
    if(type(fdobj) == skfda.FDataBasis):
        x = fdobj.coefficients

    if type(fdobj) == dict:
        #Multivariate case
        if len(fdobj.keys()) > 1:
            x = fdobj['0'].coefficients
            for i in range(1, len(fdobj)):
                x = np.c_[x, fdobj[f'{i}'].coefficients]
        #univariate but as dict
        else:
            x = fdobj['0'].coeficients

    N = x.shape[0]
    p = x.shape[1]
    n = np.sum(t, axis=0)
    prop = n/N
    #matrix with K columns and p rows
    mu = np.repeat(0., K*p).reshape((K, p))
    mu1 = np.repeat(0., K*p).reshape((K, p))

    corX = t*tw

    for i in range(0, K):
        mu[i] = np.apply_along_axis(np.sum,1,np.atleast_2d(np.atleast_2d(corX[:, i]).T@np.atleast_2d(np.repeat(1,p))).T * x.T)  / np.sum(corX[:,i])
        mu1[i] = np.sum(np.atleast_2d(corX[:,i])*x.T, axis=1)/np.sum(corX[:,i])

    #ind = np.apply_along_axis(np.where, 1, t>0)
    
    #n_bis = np.arange(0,K)
    #for i in range(0,K):
        #verify this is the same in R code. Should be, since [[i]] acceses the list item i
        #n_bis[i] = len(ind[i])


    match dfupdate:

        case "approx":
            jk861 = _T_tyxf8(dfconstr, nux, n, t, tw, K, p, N)
            testing = jk861
            if np.all(np.isfinite(testing)):
                nux = jk861
        
        case "numeric":
            jk681 = _T_tyxf7(dfconstr, nux, n, t, tw, K, p, N)
            testing = jk681
            if np.all(np.isfinite(testing)):
                nux = jk681


    traceVect = np.zeros(K)

    ev = np.repeat(0., K*p).reshape((K,p))

    Q = {}
    fpcaobj = {}

    for i in range(0, K):
        donnees = _T_mypcat_fd1(fdobj, Wlist, t[:,i], corX[:,i])
        traceVect[i] = np.sum(np.diag(donnees['valeurs_propres']))
        ev[i] = donnees['valeurs_propres']
        Q[f'{i}'] = donnees['U']
        fpcaobj[f'{i}'] = donnees

    #Intrinsic dimensions selection
    
    d = _T_hdclassif_dim_choice(ev, n, method, threshold, False, noise_ctrl, d_set)
    #correct for Python indices
    d+=1

    #Setup Qi matrices

    Q1 = Q.copy()
    for i in range(0, K):
        # verify that in R, matrix(Q[[i]]... ) just constructs a matrix with same dimenstions as Q[[i]]...
        Q[f'{i}'] = Q[f'{i}'][:,0:d[i]]

    #Parameter a

    ai = np.repeat(np.NaN, K*np.max(d)).reshape((K, np.max(d)))
    if model in ['AKJBKQKDK', 'AKJBQKDK']:
        for i in range(0, K):
            ai[i, 0:d[i]] = ev[i, 0:d[i]]

    elif model in ['AKBKQKDK', 'AKBQKDK']:
        for i in range(0, K):
            ai[i] = np.repeat(np.sum(ev[i, 0:d[i]]/d[i]), np.max(d))

    else:
        a = 0
        eps = np.sum(prop*d)
        for i in range(K):
            a = a + np.sum(ev[i, 0:d[i]]) * prop[i]
        ai = np.repeat(a/eps, K*np.max(d)).reshape((K, np.max(d)))

    #Parameter b

    bi = np.repeat(np.NaN, K)
    if model in ['AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK']:
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])
            bi[i] = remainEV/(p-d[i])

    else:
        b = 0
        eps = np.sum(prop*d)
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]])
            b = b+remainEV*prop[i]
        bi[0:K] = b/(min(N,p)-eps)

    result = {'model':model, "K": K, "d":d, "a":ai, "b": bi, "mu":mu, "prop": prop, "nux":nux, "ev":ev, "Q":Q, "fpcaobj":fpcaobj, "Q1":Q1}
    return result        


#degrees of freedom functions modified yxf7 and yxf8 functions
# /*
# * Authors: Andrews, J. Wickins, J. Boers, N. McNicholas, P.
# * Date Taken: 2023-01-01
# * Original Source: teigen (modified)
# * Address: https://github.com/cran/teigen
# *
# */
def _T_tyxf7(dfconstr, nux, n, t, tw, K, p, N):
    newnux = nux.copy()
    if dfconstr == "no":
        dfoldg = nux.copy()

        #scipy digamma is slow? https://gist.github.com/timvieira/656d9c74ac5f82f596921aa20ecb6cc8
        for i in range(0, K):
            constn = 1 + (1/n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log((dfoldg[i] + p)/2)
            
            f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
          
            #Verify this outputs the same as R: may need to set rtol to 0
            newnux[i] = scio.brentq(f, 0.0001, 1000, xtol=0.00001)

            if newnux[i] > 200.:
                newnux[i] = 200.

            if newnux[i] < 2.:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]

        constn = 1 + (1/N) * np.sum(t *(np.log(tw) - tw)) + scip.digamma( (dfoldg + p) / 2) - np.log( (dfoldg + p) / 2)

        f = lambda v : np.log(v/2) - scip.digamma(v/2) + constn
            #Verify this outputs the same as R: may need to set rtol to 0
        
        dfsamenewg = scio.brentq(f, a=0.0001, b=1000, xtol=0.01)

        if dfsamenewg > 200.:
            dfsamenewg = 200.
        
        if dfsamenewg < 2.:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

def _T_tyxf8(dfconstr, nux, n, t, tw, K, p, N):
    newnux = nux.copy()

    if(dfconstr == "no"):
        dfoldg = nux.copy()
        
        for i in range(0, K):
            constn = 1 + (1 / n[i]) * np.sum(t[:, i] * (np.log(tw[:, i]) - tw[:, i])) + scip.digamma((dfoldg[i] + p)/2) - np.log( (dfoldg[i] + p)/2)
            
            constn = -constn
            newnux[i] = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg[i] / 2)) - ( (dfoldg[i]/2) - (1/2)))) / (1 - np.exp(constn))

            if newnux[i] > 200.:
                newnux[i] = 200.

            if newnux[i] < 2.:
                newnux[i] = 2.

    else:
        dfoldg = nux[0]
        constn = 1 + (1 / N) * np.sum(t * (np.log(tw) - tw)) + scip.digamma((dfoldg + p)/2) - np.log( (dfoldg + p)/2)
        constn = -constn

        dfsamenewg = (-np.exp(constn) + 2 * (np.exp(constn)) * (np.exp(scip.digamma(dfoldg / 2)) - ( (dfoldg/2) - (1/2)))) / (1 - np.exp(constn))

        if dfsamenewg > 200.:
            dfsamenewg = 200.

        if dfsamenewg < 2.:
            dfsamenewg = 2.

        newnux = np.repeat(dfsamenewg, K)

    return newnux

def _T_mypcat_fd1(fdobj, Wlist, Ti, corI):
    #Univariate here
    if type(fdobj) == skfda.FDataBasis:
        temp = fdobj.copy()

        mean_fd = fdobj.copy()
        #Check this element-wise multiplication
        coefmean = np.apply_along_axis(np.sum, axis=1, arr=np.atleast_2d(np.atleast_2d(np.atleast_2d(corI).T@np.atleast_2d(np.repeat(1, fdobj.coefficients.shape[1]))).T * temp.coefficients.T)) / np.sum(corI)
        temp.coefficients = np.apply_along_axis(lambda row: row - coefmean, axis=1, arr=temp.coefficients)
        mean_fd.coefficients = coefmean
        coef = temp.coefficients.copy().T
        rep = (_T_repmat(np.sqrt(corI), n=coef.shape[0], p=1) * coef).T
        mat_cov = (rep.T@rep) / np.sum(Ti)
        cov = (Wlist['W_m']@ mat_cov)@(Wlist['W_m'].T)
        if not check_symmetric(cov, 1.e-12):
            ind = np.nonzero(cov - cov.T > 1.e-12)
            cov[ind] = cov.T[ind]


        valeurs_propres, vecteurs_propres = scil.eig(cov)
        #indices = valeurs_propres.argsort()
        #valeurs_propres = valeurs_propres[indices[::-1]]
        #vecteurs_propres = vecteurs_propres[indices[::-1]]

        fonctionspropres = fdobj.copy()
        bj = scil.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@np.real(vecteurs_propres)
        fonctionspropres.coefficients = bj

        scores = skfda.misc.inner_product_matrix(temp.basis, fonctionspropres.basis)
        varprop = valeurs_propres / np.sum(valeurs_propres)
        pcafd = {'valeurs_propres': np.real(valeurs_propres), 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd}

    #Multivariate here
    else:
        mean_fd = {}
        temp = fdobj.copy()
        for i in range(len(fdobj)):
            #TODO should we start indexing multivariate at 0? or at 1?
            mean_fd[f'{i}'] = temp[f'{i}'].copy()


        for i in range(len(fdobj)):
            #Check this element-wise multiplication
            coefmean = np.apply_along_axis(np.sum, axis=1, arr=np.atleast_2d(np.atleast_2d(np.atleast_2d(corI).T@np.atleast_2d(np.repeat(1, fdobj[f'{i}'].coefficients.shape[1]))).T * temp[f'{i}'].coefficients.T)) / np.sum(corI)
            temp[f'{i}'].coefficients = np.apply_along_axis(lambda row: row - coefmean, axis=1, arr=temp[f'{i}'].coefficients)
            mean_fd[f'{i}'].coefficients = coefmean
        
        #R transposes here
        coef = temp['0'].coefficients.copy()

        for i in range(1, len(fdobj)):
            coef = np.c_[coef, temp[f'{i}'].coefficients.copy()]

        rep = (_T_repmat(np.sqrt(corI), n=coef.shape[1], p=1) * coef).T
        mat_cov = (rep.T@rep) / np.sum(Ti)
        cov = (Wlist['W_m']@ mat_cov)@(Wlist['W_m'].T)

        valeurs_propres, vecteurs_propres = scil.eig(cov)
        # indices = valeurs_propres.argsort()
        # valeurs_propres = valeurs_propres[indices[::-1]]
        # vecteurs_propres = vecteurs_propres[indices[::-1]]

        bj = scil.solve(Wlist['W_m'], np.eye(Wlist['W_m'].shape[0]))@np.real(vecteurs_propres)
        
        fonctionspropres = fdobj['0']
        fonctionspropres.coefficients = bj
        scores = (coef@Wlist['W_m'])@bj

        varprop = valeurs_propres/np.sum(valeurs_propres)

        pcafd = {'valeurs_propres': np.real(valeurs_propres), 'harmonic': fonctionspropres, 'scores': scores,
                 'covariance': cov, 'U':bj, 'varprop': varprop, 'meanfd': mean_fd}

    return pcafd

def _T_hddc_ari(x, y):
    if type(x) != np.ndarray:
        x = np.array(x)

    if type(y) != np.ndarray:
        y = np.array(y)

    tab = pd.crosstab(x, y).values
    if np.all(tab.shape == (1,1)): return 1
    a = np.sum(scip.binom(tab, 2))
    b = np.sum(scip.binom(np.sum(tab, axis=1), 2)) - a
    c = np.sum(scip.binom(np.sum(tab, axis=0), 2)) - a
    d = scip.binom(np.sum(tab), 2) - a - b - c
    ari = (a - (a + b) * (a + c)/(a+b+c+d))/((a+b+a+c)/2 - (a+b) * (a + c)/(a+b+c+d))
    return ari

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
            max_dev = np.apply_along_axis(np.nanmax, 1, dev)
            dev = (dev / np.repeat(max_dev, p-1).reshape(dev.shape)).T
            #Apply's axis should cover this situation. Try axis=1 in *args if doesn't work
            d = np.apply_along_axis(np.argmax, 1, (dev > threshold).T*(np.arange(0, p-1))*((ev[:,1:] > noise_ctrl)))
        elif (method == "bic"):

            d = np.repeat(0, K)
            
            for i in range(K):
                Nmax = np.max(np.nonzero(ev[i] > noise_ctrl)[0]) - 1
                B = np.empty((Nmax,1))
                p2 = np.sum(np.invert(np.isnan(ev[i])))
                Bmax = -np.inf

                for kdim in range(0, Nmax):
                    if d[i] != 0 and kdim > d[i] + 10: break
                    #adjusted for python indices
                    a = np.sum(ev[i, 0:(kdim+1)])/(kdim+1)
           
                    b = np.sum(ev[i, (kdim + 1):p2])/(p2-(kdim+1))
                 

                    if b < 0 or a < 0:
                        B[kdim] = -np.inf

                    else:
                        #Adjusted for python indices
                        L2 = -1/2*((kdim+1)*np.log(a) + (p2 - (kdim + 1))*np.log(b) - 2*np.log(prop[i]) +p2*(1+1/2*np.log(2*np.pi))) * n[i]
                        B[kdim] = 2*L2 - (p2+(kdim+1)*(p2-(kdim+2)/2)+1) * np.log(n[i])
                       
                    if B[kdim] > Bmax:
                        Bmax = B[kdim]
                        d[i] = kdim

            if graph:
                None

        elif method == "grid":
            d = d_set.copy()

    else:
        ev = ev.flatten()
        p = len(ev)

        if method == "cattell":
            dvp = np.abs(np.diff(ev))
            Nmax = np.max(np.nonzero(ev>noise_ctrl)[0]) - 1
            if p ==2:
                d = 0
            else:
                d = np.max(np.nonzero(dvp[0:Nmax] >= threshold*np.max(dvp[0:Nmax]))[0])
            diff_max = np.max(dvp[0:Nmax])

        elif method == "bic":
            d = 0
            Nmax = np.max(np.nonzero(ev > noise_ctrl)[0]) - 1
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

    if type(d) != np.ndarray:
        d=np.array([d])
    return d


def _T_hdclassift_bic(par, p, dfconstr):
    #mux and mu not used, should we get rid of them?
    model = par['model']
    K = par['K']
    #d already adjusted for Python indices
    d = np.array(par['d'].data)
    b = np.array(par['b'].data)
    a = np.array(par['a'].data)
    #mu = par['mu']
    N = par['N']
    prop = np.array(par['prop'].data)

    if len(b) == 1:
        eps = np.sum(prop*d)
        #get ncol from ev
        n_max = par['ev'].shape[1]
        b = b*(n_max-eps) / (p-eps)
        b = np.repeat(b, K)

    if len(a.flatten()) == 1:
        #repeat single element
        a = np.repeat(a, K*np.max(d)).reshape((K, np.max(d)))

    elif len(a.flatten()) == K:
        #Repeat vector column-wise
        a = np.tile(a, np.max(d)).reshape((K, np.max(d))).T


    if np.nanmin(a) <= 0 or np.any(b < 0):
        return - np.Inf
    
    if np.isnan(np.sum(par['loglik'])):
        som_a = np.zeros(K)

        for i in range(K):
            som_a[i] = np.sum(np.log(a[i][0:d[i]]))
        L = -(1/2)*np.sum(prop * (som_a + (p-d) * np.log(b) - 2 * np.log(prop) + p * (1 + np.log(2*np.pi))))*N

    else:
        L = par['loglik']

    if dfconstr == 'no':
        ro = K*(p+1)+K-1
    else:
        ro = K*p+K
    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0]
    #to = d*(p-(d+1)/2)

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

    Z = ( (t - np.atleast_2d(np.apply_along_axis(np.max, 1, t)).T) == 0. ) + 0.
    icl = bic - 2*np.sum(Z*np.log(t + 1.e-15))

    return {'bic': bic, 'icl': icl}

def _T_hdc_getComplexityt(par, p, dfconstr):
    model = par['model']
    K = par['K']
    #d should already be adjusted for Python indices
    d = par['d']

    #These don't get used
    #b = par['b']
    #a = par['a']
    #mu = par['mu']
    #prop = par['prop']

    if dfconstr == 'no':
        ro = K*(p+1)+K - 1

    else:
        ro = K*p + K

    tot = np.sum(d*(p-(d+1)/2))
    D = np.sum(d)
    d = d[0] + 1
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

def _T_hdc_getTheModel(model, all2models = False):
    model_in = model
    #is the model a list or array?
    try:
        if type(model) == np.ndarray or type(model) == list:
            new_model = np.array(model,dtype='<U9')
            model = np.array(model)
        else:
            new_model = np.array([model],dtype='<U9')
            model = np.array([model])
    except:
        raise ValueError("Model needs to be an array or list")

    #one-dimensional please
    if(model.ndim > 1):
        raise ValueError("The argument 'model' must be 1-dimensional")
    #check for invalid values
    if type(model[0]) != np.str_:
        if np.any(np.apply_along_axis(np.isnan, 0, model)):
            raise ValueError("The argument 'model' cannot contain any Nan")

    #List of model names accepted
    ModelNames = np.array(["AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK"])
    #numbers between 0 and 5 inclusive are accepted, so check if numbers are
    #sent in as a string before capitalizing
    if type(model[0]) == np.str_:
        if model[0].isnumeric():
            model = model.astype(np.int_)
            
        else:
            new_model = [np.char.upper(m) for m in model]

    #shortcut for all the models
    if len(model) == 1 and new_model[0] == "ALL":
        if all2models:
            model = np.arange(0, 6)
            new_model = np.arange(0,6)
        else:
            return "ALL"
        
    #are the models numbers?
    if type(model[0]) == np.int_:
        qui = np.nonzero(np.isin(model, np.arange(0, 14)))[0]
        if len(qui) > 0:
            new_model[qui] = ModelNames[model[qui]]

    #find model names that are incorrect    
    qui = np.nonzero( np.invert(np.isin(new_model, ModelNames)))[0]
    if len(qui) > 0:
        if len(qui) == 1:
            msg = f'(e.g. {model_in[qui[0]]} is incorrect.)'

        else:
            msg = f'(e.g. {model_in[qui[0]]} or {model_in[qui[1]]} are incorrect.)'

        raise ValueError("Invalid model name " + msg)
    
    #Warn user that the models *should* be unique
    if np.max(np.unique(model, return_counts=True)[1]) > 1:
        warnings.warn("Values in 'model' argument should be unique.", UserWarning)

    mod_num = []
    for i in range(len(new_model)):
        mod_num.append(np.nonzero(new_model[i] == ModelNames)[0])
    mod_num = np.sort(np.unique(mod_num))
    new_model = ModelNames[mod_num]

    return new_model

def _T_addCommas(x):
    vfunc = np.vectorize(_T_addCommas_single)
    return vfunc(x)

def _T_addCommas_single(x):
    #R code
    '''if not np.isfinite(x):
        return str(x)
    
    s = np.sign(x)
    x = np.abs(x)

    decimal = x - np.floor(x)
    if decimal > 0:
        dec_string = str(decimal)[1:4]
    else:
        dec_string = ""

    entier = str(np.floor(x))
    quoi = list(entier[-1::-1])
    n = len(quoi)
    sol = []
    for i in range(n):
        sol.extend(quoi[i])
        if i % 3 == 0 and i != n:
            sol.append(",")

    '''
    return "{:,.2f}".format(x)

def _T_repmat(v, n, p):
    M = np.c_[np.repeat(1, n)]@np.atleast_2d(v)
    M = np.tile(M, p)

    return M

def _T_diago(v):
    if len(v) == 1:
        res = v
    else:
        res = np.diag(v)

    return res

def _T_imahalanobis(x, muk, wk, Qk, aki):
    
    #C code not working for now, try compiling dll on current machine?
    #so_file = "./src/TFunHDDC.so"
    #c_lib = ctypes.CDLL(so_file)

    p = x.shape[1]
    N = x.shape[0]
    
    X = x - muk

    Qi = np.matmul(wk, Qk)

    #xQu = np.matmul(X, Qi)

    proj = np.matmul(np.matmul(X, Qi), aki)

    res = np.sum(proj ** 2, axis=1)

    return res

def _T_estimateTime(stage, start_time=0, totmod=0):
    curwidth = get_terminal_size()[0]
    outputString = ""

    if stage == 'init':
        medstring = "????"
        longstring = "????"
        shortstring = "0"
        modelCount = None
        unitsRun = ""
        unitsRemain=""

    else:

        modelCount = stage + 1
        modsleft = totmod - modelCount
        timerun = time.process_time() - start_time
        timeremain = (timerun/modelCount)*modsleft
        
        if timeremain > 60 and timeremain <=3600:
            unitsRemain = 'mins'
            timeremain = timeremain/60
        elif timeremain > 3600 and timeremain <= 86400:
            unitsRemain = 'hours'
            timeremain = timeremain/3600
        elif timeremain > 86400 and timeremain <= 604800:
            unitsRemain = 'days'
            timeremain = timeremain/86400
        elif timeremain > 604800:
            unitsRemain = 'weeks'
            timeremain = timeremain/604800
        else:
            unitsRemain = 'secs'

        if timerun > 60 and timerun <=3600:
            unitsRun = 'mins'
            timerun = timerun/60
        elif timerun > 3600 and timerun <= 86400:
            unitsRun = 'hours '
            timerun = timerun/3600
        elif timerun > 86400 and timerun <= 604800:
            unitsRun = 'days'
            timerun = timerun/86400
        elif timerun > 604800:
            unitsRun = 'weeks '
            timerun = timerun/604800
        else:
            unitsRun = 'secs'


        shortstring = round((1-modsleft/totmod)*100)
        medstring = round(timeremain, 1)
        longstring = round(timerun,1)


    if curwidth >=15:
        shortstring = str(shortstring).rjust(5)
        outputString = f'{shortstring}% complete'

        if curwidth >=48:
            medstring = str(medstring).rjust(10)
            outputString = f'Approx. remaining:{medstring} {unitsRemain}  |  {outputString}'

            if curwidth >=74:
                longstring = str(longstring).rjust(10)
                outputString = f'Time taken:{longstring} {unitsRun}  |  {outputString}'

    print(outputString,'\r', flush=True, end='')

def _T_hddc_control(params):

    K = ('K',params['K'])
    checkMissing(K)
    checkType(K, (INT_TYPES))
    checkRange(K, lower=1)

    data = ('data', params['data'])
    checkMissing(data)
    checkType(data, (skfda.FDataBasis, dict))

    if isinstance(data[1], skfda.FDataBasis):
        checkType(('data', data[1].coefficients), (LIST_TYPES, (INT_TYPES, FLOAT_TYPES, LIST_TYPES)))
        naCheck=np.sum(data[1].coefficients)
        if naCheck in UNKNOWNS or pd.isna(naCheck):
            raise ValueError(f"'data' parameter contains unsupported values. Please remove NaNs, NAs, infs, etc. if they are present")
        if np.any(np.array(K[1])>2*data[1].coefficients.shape[1]):
            raise ValueError("The number of observations in the data must be at least twice the number of clusters")
        row_length = data[1].coefficients.shape[0]
    else:
        data_length = 0
        row_length = 0
        for i in range(len(data[1])):
            checkType((f'data', data[1][f'{i}'].coefficients), (LIST_TYPES, (INT_TYPES, FLOAT_TYPES, LIST_TYPES)))
            naCheck=np.sum(data[1][f'{i}'].coefficients)
            if naCheck in UNKNOWNS or pd.isna(naCheck):
                raise ValueError(f"'data' parameter contains unsupported values. Please remove NaNs, NAs, infs, etc. if they are present")
            data_length += data[1][f'{i}'].coefficients.shape[1]
            row_length += data[1][f'{i}'].coefficients.shape[0]

        if np.any(np.array(K[1])>2*data_length):
            raise ValueError("The number of observations in the data must be at least twice the number of clusters")

    model = ("model", params['model'])
    checkMissing(model)
    checkType(model, (str, INT_TYPES))

    known = ('known', params['known'])
    if not (known[1] is None):
        checkType(known, (LIST_TYPES, (INT_TYPES, FLOAT_TYPES)))

        if(len(np.nonzero(np.array(known[1]).dtype == FLOAT_TYPES and known[1] != np.Nan)[0]) > 0):
            raise ValueError("'Known' parameter should not contain values of type float except for NaN")
    
        if isinstance(K[1], LIST_TYPES):
            k_temp = K[1][0]
            if len(K[1]) > 1:
                raise ValueError("K should not use multiple values when using 'known' parameter")
        else:
            k_temp = K[1]

        if np.all(np.isnan(known[1])) or np.all(pd.isna(known[1])):
            raise ValueError("'known' should have values from each class (should not all be unknown)")
        
        if len(known[1]) != row_length:
            raise ValueError("length of 'known' parameter must match number of observations from data")
        knownTemp = np.where(pd.isna(known[1]) or np.isnan(known[1]), 0, known[1])
        if len(np.unique(knownTemp)) > k_temp:
            raise ValueError("at most K different classes can be present in the 'known' parameter")
        
        if np.max(knownTemp) > K-1:
            raise ValueError("group numbers in 'known' parameter must come from integers up to K (ie. for K=3, 0,1,2 are acceptable)")

    dfstart = ('dfstart', params['dfstart'])
    checkMissing(dfstart)
    checkType(dfstart, (INT_TYPES, FLOAT_TYPES))
    checkRange(dfstart, lower=2)

    threshold = ('threshold', params['threshold'])
    checkMissing(threshold)
    checkType(threshold, (INT_TYPES, FLOAT_TYPES))
    checkRange(threshold, upper=1, lower=0)
    
    dfupdate = ('dfupdate', params['dfupdate'])
    checkMissing(dfupdate)
    checkType(dfupdate, (str))
    if dfupdate[1] not in ['approx', 'numeric']:
        raise ValueError("'dfupdate' parameter should be either 'approx' or 'numeric'")
    
    dfconstr = ('dfconstr', params['dfconstr'])
    checkMissing(dfconstr)
    checkType(dfconstr, (str))
    if dfconstr[1] not in ['no', 'yes']:
        raise ValueError("'dfconstr' parameter should be either 'no' or 'yes'")
    
    itermax = ('itermax', params['itermax'])
    checkMissing(itermax)
    checkType(itermax, (INT_TYPES))
    checkRange(itermax, lower=2)

    eps = ('eps', params['eps'])
    checkMissing(eps)
    checkType(eps, (INT_TYPES, FLOAT_TYPES))
    checkRange(eps, lower=0)

    init = ('init', params['init'])
    checkMissing(init)
    checkType(init, (str))

    match init[1]:

        case "vector":
            vec = ('init_vector', params['init_vector'])

            checkMissing(vec)
                
            checkType(vec, (LIST_TYPES, (INT_TYPES)))

            if isinstance(K[1], LIST_TYPES):
                k_temp = K[1][0]
                if len(K[1]) > 1:
                    raise ValueError("K should not use multiple values when using init = 'vector'")

            if len(np.unique(vec[1])) < k_temp:
                raise ValueError(f"'init_vector' lacks representation from all K classes (K={K})")

            if len(vec[1]) != row_length:
                raise ValueError("Size 'init_vector' is different from size of data")

        case "mini-em":
            mini = ('mini_nb', params['mini_nb'])

            checkMissing(mini)
            checkType(mini, (LIST_TYPES, (INT_TYPES)))
            checkRange(mini, lower=1)

            if len(mini[1]) != 2:
                raise ValueError(f"Parameter 'mini_nb' should be of length 2, not length {len(mini[1])}")
            
        case "kmeans":
            kmc = ('kmeans_control', params['kmeans_control'])

            if kmc[1] is None:
                pass
            else:
                checkType(kmc, [dict])
                checkKMC(kmc[1])

    criterion = ('criterion', params['criterion'])
    checkMissing(criterion)
    checkType(criterion, (str))

    if criterion[1] not in ['bic', 'icl']:
        raise ValueError("'Criterion' parameter should be either 'bic' or 'icl'")
    
    d_select = ('d_select', params['d_select'])
    checkMissing(d_select)
    checkType(d_select, (str))

    if d_select[1] == 'grid':
        d_range = ('d_range', params['d_range'])
        checkMissing(d_range)
        checkType(d_range, (INT_TYPES))
        checkRange(d_range, lower=1)

        if np.max(d_range) > data_length:
            raise ValueError("Intrinsic dimension 'd' can't be larger than number of input parameters. Please set lower max")

    if d_select[1] not in ['cattell', 'bic']:
        raise ValueError("'d_select' parameter should be 'cattell' 'bic', or 'grid'")
    
    show = ('show', params['show'])
    checkMissing(show)
    checkType(show, [bool])

    min_indiv = ('min_individuals', params['min_individuals'])
    checkMissing(min_indiv)
    checkType(min_indiv, (INT_TYPES))
    checkRange(min_indiv, lower=2)

    cores = ('mc_cores', params['mc_cores'])
    checkMissing(cores)
    checkType(cores, (INT_TYPES))
    checkRange(cores, lower=1)

    rep = ('nb_rep', params['nb_rep'])
    checkMissing(rep)
    checkType(rep, (INT_TYPES))
    checkRange(rep, lower=1)

    keep = ('keepAllRes', params['keepAllRes'])
    checkMissing(keep)
    checkType(keep, [bool])
    
    d_max = ('d_max', params['d_max'])
    checkMissing(d_max)
    checkType(d_max, (INT_TYPES))
    checkRange(d_max, lower=1)

    verbose = ('verbose', params['verbose'])
    checkMissing(verbose)
    checkType(verbose, [bool])


def checkType(param, check):
    if not isinstance(check, type):
        if check[0] is LIST_TYPES:
            result = isinstance(param[1], LIST_TYPES)
            if not np.all(result):
                raise ValueError(f"Parameter {param[0]} is of wrong type (should be of type {LIST_TYPES[0]} for example)")
            
            result = np.array([isinstance(val, check[1]) for val in param[1]])

            if param[0] == "data":
                for i in param[1]:
                    checkType((param[0], i), (INT_TYPES, FLOAT_TYPES))

            if not np.all(result):
                
                result = np.nonzero(result == False)[0]
                raise ValueError(f"Parameter {param[0]} contains data of an incorrect type (cannot contain elements of type {type(param[1][result][0])} for example)")

    else:

        if isinstance(param[1], LIST_TYPES):
            result = np.array([isinstance(val, check) for val in param[1]])

        else:
            result = isinstance(param[1], check)

        if not np.all(result):

            result = np.nonzero(result == False)[0]
            if isinstance(param[1], LIST_TYPES):
                result = param[1][result][0]

            else:
                result = param[1]

            raise ValueError(f"Parameter {param[0]} is of an incorrect type (cannot be of type {type(result)}) for example")
            

def checkMissing(param):

    if param is None:
        raise ValueError(f"Missing required '{param[0]}' parameter")
    
def checkRange(param, upper=None, lower=None):

    result = True
    if isinstance(param[1], LIST_TYPES):
        if not (lower is None):

            result = np.min(param[1]) < lower

        elif not (upper is None):
            
            result = result and (np.max(param[1]) > upper)

    else:

        if not (lower is None):
            result = param[1] < lower
        
        elif not (upper is None):
            result = result and (param[1] > upper)

    if result:
        msg = ""
        if lower != False:
            msg = f' greater than or equal to {lower}'

        elif upper != False:
            if len(msg > 0):
                msg = f'{msg} and'
            msg = f'{msg} less than or equal to {upper}'

        raise ValueError(f"Parameter '{param[0]}' must be {msg}")

def checkKMC(kmc):
    settingNames = ['n_init', 'max_iter', 'algorithm']

    result = [name in kmc for name in settingNames]
    
    if not np.all(result):
        result = np.nonzero(result == False)[0][0]
        raise ValueError(f"Missing setting {result} in parameter 'kmeans_control'")
    
    checkType(('n_init',kmc['n_init']), ((INT_TYPES)))
    checkRange(('n_init',kmc['n_init']), lower=1)
    checkType(('max_iter', kmc['max_iter']), (INT_TYPES))
    checkRange(('max_iter',kmc['max_iter']), lower=1)
    checkType(kmc['algorithm'], (str))

def deafaultKMC(kmc):
    return {}