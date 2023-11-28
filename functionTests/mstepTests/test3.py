import skfda
import numpy as np
import pandas as pd
import NOxBenchmark as NOx
import csv
import scipy.special as scip
import scipy.optimize as scio

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
                Nmax = np.max(np.where(ev[i] > noise_ctrl))
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
                        #print(np.log(n[i]))
                        #print(L2)
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
                Nmax = np.max(np.which(ev>noise_ctrl))
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

    #print(len(B))
    return d

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
        Qk = Q1[f'{i}']

        aki = np.sqrt(np.diag(np.concatenate((1/a[i, 0:d[i]+1],np.repeat(1/b[i], p-d[i]-1) ))))
        muki = mu[i]

        wki = wlist['W_m']
        mah_pen[:,i] = _T_imahalanobis(x, muki, wki, Qk, aki)
        W[:, i] = (nux[i] + p) / (nux[i] + mah_pen[:, i])

    return W



def _T_funhddt_init(fdobj, Wlist, K, t, nux, model, threshold, method, noise_ctrl, com_dim, d_max, d_set):
    if com_dim == None:
        com_dim = False
    if(type(fdobj) == skfda.FDataBasis):
        x = fdobj.coefficients

    if type(fdobj) == dict:
        #Multivariate
        x = fdobj[f'{0}'].coefficients
        for i in range(0, len(fdobj)):
            x = np.c_[x, fdobj[f'{i}'].coefficients]
        

    #coefficients are transposed from R
    N = x.shape[0]
    p = x.shape[1]
    n = np.sum(t, axis=0)
    prop = n/N
    mu = np.repeat(0., K*p).reshape((K, p))
    ind = np.apply_along_axis(np.nonzero, 1, t>0)
    n_bis = np.repeat(0., K)

    for i in range(K):
        n_bis[i] = len(ind[i])

    traceVect = np.repeat(0., K)
    ev = np.repeat(0., K*p).reshape((K, p))
    Q = {}
    fpcaobj = {}

    for i in range(K):
        donnees = _T_initmypca_fd1(fdobj, Wlist, t[:,i])
        print(donnees['mux'])
        mu[i] = donnees['mux']
        traceVect[i] = np.sum(np.diag(donnees['valeurs_propres']))
        ev[i] = donnees["valeurs_propres"]
        Q[f'{i}'] = donnees["U"]
        fpcaobj[f"{i}"] = donnees

    if np.isin(model, np.array(["AJBQD", "ABQD"])):
        d = np.repeat(com_dim, K)

    elif np.isin(model, np.array(["AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD"])):
        dmax = np.min(np.apply_along_axis(np.argmax, 1, (ev>noise_ctrl)*np.repeat(np.arange(0, ev.shape[1]), K).reshape((K, p)))) - 1
        if com_dim != False:
            if com_dim > dmax:
                com_dim = np.max(dmax, 1)
        d = np.repeat(com_dim, K)

    else:
        d = _T_hdclassif_dim_choice(ev, n, method, threshold, False, noise_ctrl, d_set)

    Q1 = Q.copy()
    
    for i in range(K):
        Q[f'{i}'] = Q[f'{i}'][:, 0:d[i]+1]

    ai = np.repeat(np.NaN, K*(np.max(d)+1)).reshape((K, (np.max(d)+1)))
    if np.isin(model, np.array(['AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD'])):
        for i in range(K):
            ai[i, 0:d[i]+1] = ev[i, 0:d[i]+1]

    elif np.isin(model, np.array(['AKBKQKDK', 'AKBQKDK', 'AKBKQKD', 'AKBQKD'])):
        for i in range(K):
            ai[i] = np.repeat(np.sum(ev[i, 0:d[i]+1])/(d[i]+1), np.max(d)+1)

    elif model == "AJBQD":
        for i in range(K):
            ai[i] = ev[0:d[1]+1]
    
    elif model == "ABQD":
        #TODO check if the sum is returning a vector or a number
        ai = np.repeat(np.sum(ev[0:d[1]+1]/(d[1]+1)), K*(np.max(d)+1)).reshape((K, np.max(d)+1))

    else:
        a = 0
        eps = np.sum(prop*(d+1))

        for i in range(K):
            a += (np.sum(ev[i, 0:d[i]+1])*prop[i])
            ai = np.full((K, max(d)+1), a/eps)

    bi = np.zeros(K)
    #Not used
    #denom = np.min(N, p)
    if np.isin(model, np.array(['AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK', 'AKJBKQKD', 'AKBKQKD', 'ABKQKD'])):
        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]+1])

            bi[i] = remainEV/(p-d[i]-1)

    else:
        b = 0
        eps = np.sum(prop*(d+1))

        for i in range(K):
            remainEV = traceVect[i] - np.sum(ev[i, 0:d[i]+1])

            b += (remainEV*prop[i])

        bi[0:K] = b/(min(N, p) - eps)

    return {'model': model, "K": K, 'd':d, 'a':ai, 'b':bi, 'mu':mu, 'prop':prop,
            'nux':nux, 'ev':ev, 'Q':Q, 'fpcaobj': fpcaobj, 'Q1':Q1}

def _T_repmat(v, n, p):
    M = np.c_[np.repeat(1, n)]@np.atleast_2d(v)
    #M = np.tile(M, p)

    return M

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
        #print(coef)
        rep = (_T_repmat(np.sqrt(corI), n=coef.shape[0], p=1) * coef).T
        mat_cov = (rep.T@rep) / np.sum(Ti)
        #print(rep)
        cov = (Wlist['W_m']@ mat_cov)@(Wlist['W_m'].T)

        valeurs_propres, vecteurs_propres = np.linalg.eig(cov.astype(float))
        fonctionspropres = fdobj.copy()
        bj = np.linalg.solve(Wlist['W_m'].astype(float), np.eye(Wlist['W_m'].shape[0]))@vecteurs_propres
        fonctionspropres.coefficients = bj

        scores = skfda.misc.inner_product_matrix(temp.basis, fonctionspropres.basis)
        varprop = valeurs_propres / np.sum(valeurs_propres)
        pcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres, 'scores': scores, 'covariance': cov, 'U':bj, 'meanfd': mean_fd}

    #Multivariate here
    else:
        mean_fd = {}
        temp = fdobj.copy()
        for i in range(len(fdobj)):
            #TODO should we start indexing multivariate at 0? or at 1?
            mean_fd[f'{i}'] = temp[f'{i}'].copy()


        for i in range(len(fdobj)):
            #Check this element-wise multiplication
            coefmean = np.apply_along_axis(np.sum, axis=1, arr=np.atleast_2d(np.atleast_2d(np.atleast_2d(corI).T@np.atleast_2d(np.repeat(1, fdobj[f'{i}'].coefficients.shape[1]))).T * temp['f{i}'].coefficients.T)) / np.sum(corI)
            mean_fd[f'{i}'].coefficients = coefmean
        
        #R transposes here
        coef = temp['0'].coefficients.copy().T

        for i in range(1, len(fdobj)):
            #R transposes here
            coef = np.c_[coef, temp[f'{i}'].coefficients.copy().T]

        rep = (_T_repmat(np.sqrt(corI), n=coef.shape[0], p=1) * coef).T
        mat_cov = (rep.T@rep) / np.sum(Ti)
        cov = (Wlist['W_m']@ mat_cov)@(Wlist['W_m'].T)

        valeurs_propres, vecteurs_propres = np.linalg.eig(cov.astype(float))

        bj = np.linalg.solve(Wlist['W_m'].astype(float), np.eye(Wlist['W_m'].shape[0]))@vecteurs_propres
        fonctionspropres = fdobj['0']
        fonctionspropres.coefficients = bj
        scores = (coef@Wlist['W_m'])@bj

        varprop = valeurs_propres/np.sum(valeurs_propres)

        pcafd = {'valeurs_propres': valeurs_propres, 'harmonic': fonctionspropres, 'scores': scores,
                 'covariance': cov, 'U':bj, 'varprop': varprop, 'meanfd': mean_fd}

    return pcafd

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
    mu = np.repeat(0., K*p).reshape((K, p))
    mu1 = np.repeat(0., K*p).reshape((K, p))

    #This is in R code but just gets overwritten later
    #corX = np.repeat(0, N*K).reshape((N, K))

    #TODO Verify if this is matrix multiplication (better to be safe using matmul)
    corX = t*tw

    for i in range(0, K):
        mu[i] = np.apply_along_axis(np.sum,1,np.atleast_2d(np.atleast_2d(corX[:, i]).T@np.atleast_2d(np.repeat(1,p))).T * x.T)  / np.sum(corX[:,i])
        mu1[i] = np.sum(np.atleast_2d(corX[:,i])*x.T, axis=1)/np.sum(corX[:,i])

    ind = np.apply_along_axis(np.where, 1, t>0)
    
    n_bis = np.arange(0,K)
    for i in range(0,K):
        #verify this is the same in R code. Should be, since [[i]] acceses the list item i
        n_bis[i] = len(ind[i])

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
                nux = jk681


    traceVect = np.zeros(K)

    ev = np.repeat(0., K*p).reshape((K,p))
    #try dictionary here
    Q = {}
    fpcaobj = {}

    for i in range(0, K):
        donnees = _T_mypcat_fd1(fdobj, Wlist, t[:,i], corX[:,i])
        #What is the context for the diag call in R? (ie. what data type is diag being called on)
        traceVect[i] = np.sum(np.diag(donnees['valeurs_propres']))
        ev[i] = donnees['valeurs_propres']
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
        d+=1

    Q1 = Q
    for i in range(0, K):
        # verify that in R, matrix(Q[[i]]... ) just constructs a matrix with same dimenstions as Q[[i]]...
        Q[f'{i}'] = Q[f'{i}'][:,0:d[i]]

    ai = np.repeat(np.NaN, K*np.max(d)).reshape((K, np.max(d)))
    if model in ['AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD']:
        for i in range(0, K):
            ai[i, 0:d[i]] = ev[i, 0:d[i]]

    elif model in ['AKBKQKDK', 'AKBQKDK', 'AKBKQKD', 'AKBQKD']:
        for i in range(0, K):
            ai[i] = np.repeat(np.sum(ev[i, 0:d[i]]/d[i]), np.max(d))

    elif model == 'AJBQD':#LINE 1256
        for i in range(K):
            ai[i] = ev[0:d[0]]

    #Verify if replacement length is an issue
    elif model == "ABQD":
        ai = np.sum(ev[0:d[0]])/d[0]

    else:
        a = 0
        eps = np.sum(prop*d)
        for i in range(K):
            a = a + np.sum(ev[i, 0:d[i]]) * prop[i]
        ai = np.repeat(a/eps, K*np.max(d)).reshape((K, np.max(d)))

    
    bi = np.repeat(None, K)
    denom = min(N, p)
    if model in ['AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK', 'AKJBKQKD', 'AKBKQKD', 'ABKQKD']:
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



noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50

data = []
mats = {'t': [], 'tw': []}
with open('functionTests/initmypcaTests/t1.csv', newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in datareader:
        data.append(row[0].split(',')[1:])

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
n = np.sum(t, axis=0)
data = NOx.fitNOxBenchmark().data
p = data.coefficients.shape[1]
K = len(t[0])

ev = np.repeat(0., K*p).reshape((K,p))
nux = np.repeat(df_start, K)

for i in range(K):
    
    donnees = _T_initmypca_fd1(data, Wlist, t[:,i])

    #ev[i] = donnees["valeurs_propres"]


initx = _T_funhddt_init(data, Wlist, K, t, nux, "AKJBKQKDK", threshold, "cattell", noise, None, d_max, d_set)
tw = _T_funhddt_twinit(data, Wlist, initx, nux)

m = _T_funhddt_m_step1(data, Wlist, K, t, tw, nux, 'approx', 'no', 'AKJBKQKDK', threshold, 'cattell', noise, None, d_max, d_set)
print(m['a'])
print(m['b'])
print(m['d'])
print(m['nux'])