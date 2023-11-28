import numpy as np

def _T_diago(v):
    if len(v) == 1:
        res = v
    else:
        res = np.diag(v)

    return res

a = np.array([1,2,3])
b = np.array([1])
c = np.array([1,2,3,4,5,6,7])
d = np.array([1.2, 3.4, 5.6])
e = np.array([[1,2,3], [4,5,6]])
f = np.array([[1,2,3], [4,5,6], [7,8,9]])

tests = [a,b,c,d,e,f]

for i in tests:
    print(_T_diago(i))