import numpy as np

def _T_hdc_getComplexityt(par, p, dfconstr):
    model = par['model']
    K = par['K']
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

par = {'model': "AKBQKDK", 'K': 5., 'd': np.array([3.,2.,2.])}
res1 = _T_hdc_getComplexityt(par, 10, 'no')
res2 = _T_hdc_getComplexityt(par, 10, 'yes')
print(f'dfconstr No: {res1}\ndfconstr Yes: {res2}')