import numpy as np
import scipy.linalg as scil
import skfda
from matplotlib import pyplot as plt
NUMERIC_TYPES = (int, float, np.integer, np.floating)

def _newContaminatedSimulation(ncurves, p, mu, K=3, prop=[], d=[], a=[], b=[], alpha=[], eta=[]):
    
    randgen = np.random.default_rng()


    if len(prop) == 0:
        prop = np.repeat(1/K, K)

    elif len(prop) != K:
        raise ValueError("Proportions don't fit with number of classes")
    
    else:
        prop = prop/np.sum(prop)

    n = np.floor(prop*ncurves).astype(int)
    N = np.sum(n)

    j = np.random.choice(p, K)

    if len(d) == 0:
        d = np.sort(np.ceil(randgen.uniform(0, 12*(p > 20)+5*(p<=20 and p >=6)+(p<6)*(p-1), K)))[::-1]
    elif len(d) != K or (not np.all([isinstance(d, NUMERIC_TYPES) for d in d])):
        raise ValueError("Wrong value of d")
    
    Q = {}
    for i in range(K):
        Q[f'{i}'] = scil.qr(randgen.multivariate_normal(size=p, mean=np.repeat(0,p), cov=np.eye(p)))[0]
    if len(a) == 0:
        a = np.sort(np.ceil(randgen.uniform(30, 350, K)))
    elif len(a) != K or (not np.all([isinstance(a, NUMERIC_TYPES) for a in a])):
        raise ValueError("Wrong value of a")
    if len(b) == 0:
        b = np.sort(np.ceil(randgen.uniform(K, 0, 25)))
    elif len(b) != K or (not np.all([isinstance(b, NUMERIC_TYPES) for b in b])):
        raise ValueError("Wrong value of b")
    
    S1 = {}
    S2 = {}

    for i in range(K):
        temp1 = Q[f'{i}']@np.sqrt(np.diag(np.concatenate((np.repeat(a[i], d[i]), np.repeat(b[i], p-d[i])))))
        temp2 = Q[f'{i}']@np.sqrt(eta[i]*np.diag(np.concatenate((np.repeat(a[i], d[i]), np.repeat(b[i], p-d[i])))))
        S1[f'{i}'] = (temp1.T)@(temp1)
        S2[f'{i}'] = (temp2.T)@(temp2)

    cl = None
    X = None
    clo = np.repeat(0, N)

    for i in range(K):
        for j in range(n[i]):
            if randgen.uniform(0, 1, 1) < alpha[i]:
                X = np.concatenate((X, randgen.multivariate_normal(size=1, mean=mu[i], cov=S1[f'{i}']))) if not X is None else randgen.multivariate_normal(size=1, mean=mu[i], cov=S1[f'{i}'])

            else:
                clo[(i-1)*n[i]+j] = 0
                X = np.concatenate((X, randgen.multivariate_normal(size=1, mean=mu[i], cov=S2[f'{i}']))) if not X is None else randgen.multivariate_normal(size=1, mean=mu[i], cov=S2[f'{i}'])

    for i in range(K):
        cl = np.concatenate((cl, np.repeat(i, n[i]))) if not (cl is None) else np.repeat(i, n[i])

    ind = randgen.choice(np.arange(0, N), n[i])
    prms = {'a':a, 'b':b, 'prop':prop, 'd':d, 'mu':mu, 'ncurves':np.sum(n)}

    return {'X':X, 'cl':cl, 'clo':clo, 'prms':prms}

def genModelFD(ncurves=1000, nsplines=35, alpha=[0.9,0.9,0.9], eta=[10,5,15]):

    mu1 = np.concatenate(([1,0,50,100], np.repeat(0., nsplines-4)))
    mu2 = np.concatenate(([0,0,80,0,40,2], np.repeat(0., nsplines-6)))
    mu3 = np.concatenate((np.repeat(0., nsplines-6), [20,0,80,0,0,100]))
    mu = np.concatenate((np.array([mu1]), np.array([mu2]), np.array([mu3])))
    a=np.array([150., 15, 30])
    b=np.array([5., 8, 10])
    d=np.array([5., 20, 10])
    eta=np.array(eta).astype(float)
    alpha=np.array(alpha).astype(float)
    coef = _newContaminatedSimulation(ncurves, nsplines, mu, d=d, a=a, b=b, eta=eta, alpha=alpha)
    simdata = np.full((coef['prms']['ncurves'], nsplines), 0.)

    basis = skfda.representation.basis.FourierBasis([0,100], n_basis = nsplines)
    #basis = skfda.representation.basis.BSplineBasis([0,100], n_basis = nsplines)
    evalBasis = basis.evaluate(np.arange(0,100))[:,:,0]
    finaldata = coef['X']@(evalBasis)

    fd = skfda.FDataGrid(finaldata, np.linspace(0, 100, 100))
    smoother = skfda.preprocessing.smoothing.BasisSmoother(basis, return_basis=True)
    smooth_fd = smoother.fit_transform(fd)

    return {'data': fd.to_basis(basis), 'labels': coef['cl']}


def plotModelFD(fd):
    fd['data'].plot(group=fd['labels'], group_colors=['red', 'blue', 'green'])
    plt.show()
