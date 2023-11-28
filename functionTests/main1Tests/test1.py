#######################################################################
#init=random
#dfupdate = numeric
#dfconstr = cattell
#model = AKJBKQKDK
#######################################################################

import numpy as np
import skfda
import tfunHDDC as tfun
import NOxBenchmark as NOx
import traceback

data = NOx.fitNOxBenchmark()['data']
W = skfda.misc.inner_product_matrix(data.basis, data.basis)
W[W < 1.e-15] = 0
W_m = np.linalg.cholesky(W).T
dety = np.linalg.det(W)
W_list = {'W': W, 'W_m': W_m, 'dety':dety}
noise = 1.e-8
graph = False
threshold = 0.1
d_set = np.array([1,1,1,1])
K=3
d_max = 100
df_start = 50
itermax = 200
try:
    res = tfun._T_funhddc_main1(data, W_list, K, df_start, 'numeric', 'yes', 'AKJBKQKDK', itermax, threshold, 'cattell', 1.e-6, 'random', None, None, 0, noise, None, None, d_max, d_set, None)
    print(res.a)
    print(res.b)
    print(res.converged)
except Exception as e:
    traceback.print_exc()