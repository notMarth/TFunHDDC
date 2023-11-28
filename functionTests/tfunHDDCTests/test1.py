import sys
sys.path.append('../..')
import time
import tfunHDDC as tfun
import NOxBenchmark as NOx

if __name__ == '__main__':
    data = NOx.fitNOxBenchmark()['data']
    start = time.time()
    res = tfun.tfunHDDC(data, K=[2,3], threshold=[0.1, 0.01], mc_cores=16)
    print(time.time() - start)
    # start = time.time()
    # tfun.tfunHDDC(data, K=2)
    # print(time.time() - start)
