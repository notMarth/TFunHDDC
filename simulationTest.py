import tfunHDDC as tfun
import numpy as np
import modelSimulation as modelSim

data = modelSim.genModelFD()['data']

res = tfun.tfunHDDC(data)