library(TFunHDDC)

a = fitNOxBenchmark()
res = inprod(a$basis, a$basis)
print(res)