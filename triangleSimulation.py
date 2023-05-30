import numpy as np
import skfda
from sklearn.utils import Bunch
from matplotlib import pyplot as plt
################################################################################

def h1(t):
    res = np.zeros(t.shape)
    for i in range(len(t)):
        if 6 - abs(t[i]-7) > 0:
            res[i] = 6-abs(t[i]-7)
    
    return res
    
def h2(t):
    res = np.zeros(t.shape)
    for i in range(len(t)):
        if 6 - abs(t[i]-15) > 0:
            res[i] = 6-abs(t[i]-15)
    
    return res
    
def sin(t):
    return np.sin((t * np.pi)/2)
    
def genTriangleScenario1():
    randGen = np.random.default_rng()
    t = np.linspace(0, 21, num=101)
    h1v = h1(t)
    h2v = h2(t)

    curves = np.zeros((400, 101))
    curves2 = np.zeros((400, 101))

    group1 = np.linspace(0,79, 80)
    group2 = np.linspace(100, 199, 100)
    group3 = np.linspace(200, 279, 80)
    group4 = np.linspace(299, 399, 101)

    for i in group1:
        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves[int(i),] = u + (0.6- u)*h1v + e

        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves2[int(i),] = u + (0.5 - u)*h1v + e

    for i in group2:
        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves[int(i),] = u + (0.6- u)*h2v + e

        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves2[int(i),] = u + (0.5 - u)*h2v + e

    for i in group3:
        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves[int(i),] = u + (0.6- u)*h1v + e

        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves2[int(i),] = u + (0.5 - u)*h2v + e

    for i in group4:
        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves[int(i),] = u + (0.6- u)*h2v + e

        u = randGen.uniform(0, 0.1, 101)
        e = randGen.normal(0, np.sqrt(0.25), 101)
        curves2[int(i),] = u + (0.5 - u)*h1v + e

    contam1 = np.linspace(80, 99, 20)
    contam3 = np.linspace(280, 299, 20)

    for i in contam1 :
        u = randGen.uniform(0, 0.1, 101)
        e1 = randGen.standard_t(4, 101)
        curves[int(i),] = (0.6 - u)*h1v + sin(t) + e1

        u = randGen.uniform(0, 0.1, 101)
        e1 = randGen.standard_t(4, 101)
        curves2[int(i),] = (0.5 - u)*h1v + sin(t) + e1 

    for i in contam3:
        u = randGen.uniform(0, 0.1, 101)
        e2 = randGen.normal(0, 2, 101)
        curves[int(i),] = (0.5 - u)*h1v + sin(t) + e2

        u = randGen.uniform(0, 0.1, 101)
        e2 = randGen.normal(0, 2, 101)
        curves2[int(i),] = (0.6 - u)*h2v + sin(t) + e2


    
    fd1 = skfda.FDataGrid(data_matrix=np.array(curves), grid_points=np.linspace(0, 21, 101))
    fd2 = skfda.FDataGrid(data_matrix=np.array(curves2), grid_points=np.linspace(0, 21, 101))

    basis = skfda.representation.basis.BSplineBasis(n_basis = 25)
    
    fd1 = fd1.to_basis(basis)
    fd2 = fd2.to_basis(basis)
    fdt = [fd1, fd2]
    
    target = np.concatenate((np.repeat(0, 80), np.repeat(4, 20), np.repeat(1, 100), np.repeat(2, 80), np.repeat(5, 20), np.repeat(3, 100)))
    return Bunch(data = fdt, target = target)
    
def plotTriangles(fdt):
    CL = ["black", "red", "green", "blue", "purple", "brown"]

    fig1 = fdt['data'][0].plot(group = fdt['target'], group_colors = CL, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("x1(t)")

    fig2 = fdt['data'][1].plot(group = fdt['target'], group_colors = CL, linestyle = '--')   
    plt.xlabel("t")
    plt.ylabel("x2(t)")

    plt.show()