import tfunHDDC as tfun
import numpy as np
import NOxBenchmark as NOx

data = NOx.fitNOxBenchmark()['data']

res = tfun.tfunHDDC(data)