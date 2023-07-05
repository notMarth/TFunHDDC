import numpy as np

def _T_repmat(v, n, p):
    M = np.c_[np.repeat(1, n)]@np.atleast_2d(v)
    M = np.tile(M, p)

    return M

v = np.array([[1,2,3,4],[5,6,7,8]])

res = _T_repmat(v, 2, 3)
print(res)
print(f'rows: {len(res)}; columns: {len(res[0])}')