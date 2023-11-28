import numpy as np
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
    if np.floor(x) == x:
        return "{:,}".format(x)
    else:
        return "{:,.2f}".format(x)

a = 123.45
b = 123
c = 123.456
d = 4000.45
e = 4000
f = 4000.456
g = 10000000000
h = 10000000000.123456789
i = -300000.1234
j = -300000

tests = [a,b,c,d,e,f,g,h,i,j]

for i in tests:
    print(_T_addCommas_single(i))