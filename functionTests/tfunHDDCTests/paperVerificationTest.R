library(TFunHDDC)
data = fitNOxBenchmark(15)
labels = fitNOxBenchmark()$groupd

K = 2
inits = c('random', 'vector', 'mini-em', 'kmeans')
vec = rep(2, length(labels))
vec[1:length(labels)/2] = 1
itTest = 1

cl=list()
for(i in 1:K) cl[[as.character(i)]] = rep(0, length(labels))

statsrand = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsvec = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsmini = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsk = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)

j = 0
myDefault = list()
myDefault$iter.max = 10
myDefault$nstart = 1
myDefault$algorithm = c("Hartigan-Wong", "Lloyd", "Forgy","MacQueen")
myDefault$trace = FALSE
myDefault$alpha = 0.2
#kmc = list(iter.max=10, nstart=1, algorithm=c("Hartigan-Wong"))

start = Sys.time()

res = TFunHDDC::tfunHDDC(data$fd, K=2, model=c("AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"), init = 'kmeans', threshold = 0.1, nb.rep = 20, dfconstr = 'no')

print(Sys.time() - start)

# for(i in inits){
#   print(i)
#   j = 1
#   
#   res = TFunHDDC::tfunHDDC(data, K=K, model=c("AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"), init=i, threshold=0.6, nb.rep=20, init.vector=vec)
#   if (!inherits(res, 't-funHDDC')){
#     next
#   }
#   
#   print(paste0("CCR for init=", i))
#   print(sum(diag(table(res$class, labels)))/length(labels))
# }
print(table(labels, res$class))
print(TFunHDDC:::.T_hddc_ari(labels, res$class))
