import skfda
import NOxBenchmark as nox
q = nox.fitNOxBenchmark()
a = q.data
#print(a)
res = (skfda.misc.inner_product_matrix(a.basis, a.basis))
print(res)