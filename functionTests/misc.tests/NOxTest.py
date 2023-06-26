import NOxBenchmark

test = NOxBenchmark.fitNOxBenchmark()
print(test['data'].coefficients[0])
NOxBenchmark.plot_NOx(test)