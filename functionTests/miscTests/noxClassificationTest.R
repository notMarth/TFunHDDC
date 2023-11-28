library(TFunHDDC)

models = c("akjbkqkdk", 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk')
data = fitNOxBenchmark()
training = c(1:50)
test = c(51:100)
clm = data$groupd
known = clm[training]

results=list()
for(i in models){
  print(i)
  start = Sys.time()
  res = tfunHDDC(data$fd[training], K=2, model=i, known = known, threshold = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), init = 'kmeans', nb.rep=20, verbose=FALSE) 
  temp1 = table(res$class, clm[training])
  temp2 = TFunHDDC:::.T_hddc_ari(clm[training], res$class)
  temp3 = Sys.time() - start
  pred = TFunHDDC::predict.tfunHDDC(res, data$fd[test])$class
  temp4 = table(pred, clm[test])
  temp5 = TFunHDDC:::.T_hddc_ari(pred, clm[test])
  print(temp1)
  print(temp2)
  print(temp3)
  print(temp4)
  print(temp5)
  print('\n')
}