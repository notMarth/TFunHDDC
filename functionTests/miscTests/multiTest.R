library(TFunHDDC)

data = TFunHDDC::genTriangles()
labels = data$groupd

res = tfunHDDC(data$fd, min.individuals = 6, model= c("akjbkqkdk", 'akjbqkdk', 'akbkqkdk', 'akbqkdk', 'abkqkdk', 'abqkdk'), K=6, threshold = c(0.05, 0.2, 0.4, 0.6), nb.rep=20, init='kmeans')

print(table(res$class, labels))
print(TFunHDDC:::.T_hddc_ari(res$class, labels))