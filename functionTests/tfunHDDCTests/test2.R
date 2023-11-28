library(TFunHDDC)
data = fitNOxBenchmark()$fd
labels = fitNOxBenchmark()$groupd

K = 2
inits = c('random', 'vector', 'mini-em', 'kmeans')
vec = rep(2, length(labels))
vec[1:length(labels)/2] = 1
kmc = list(iter.max=20, nstart=1, algorithm='Lloyd')
itTest = 1

cl=list()
for(i in 1:K) cl[[as.character(i)]] = rep(0, length(labels))

statsrand = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsvec = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsmini = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)
statsk = list(bic = c(), time=0, cls = cl, conv = 0, ccr = c(), ccrStdev=0, ari=c(), ariStdev=0)

j = 0

for(i in inits){
  print(i)
  j = 1
  
  if(i == 'random') stats = statsrand
  if(i == 'vector') stats = statsvec
  if(i=='mini-em') stats=statsmini
  if(i=='kmeans') stats=statsk
  
  while(j <= itTest){
    print(j)
    start = Sys.time()
    
    res = TFunHDDC::tfunHDDC(data, K=K, model=c("AKJBKQKDK", "AKBKQKDK", "AKBQKDK", "AKJBQKDK", "ABKQKDK", "ABQKDK"), init=i, threshold=0.6, nb.rep=30, init.vector=vec, kmeans.control = kmc)
    
    if (!inherits(res, 't-funHDDC')){
      next
    }
    
    stats$time = stats$time + (Sys.time() - start)
    
    for(k in 1:length(labels)){
      stats$cls[[as.character(res$class[k])]][k] = stats$cls[[as.character(res$class[k])]][k] + 1
    }
    
    stats$ccr[j] = sum(diag(table(res$class, labels)))/length(labels)
    stats$ari[j] = TFunHDDC:::.T_hddc_ari(res$class, labels)
    
    j = j+1
  }
  
  if(i == 'random') statsrand = stats
  if(i == 'vector'){
    statsvec = stats
  }
  if(i=='mini-em') statsmini=stats
  if(i=='kmeans') statsk=stats
  
}

for(i in inits){
  if(i == 'random'){
    stats = statsrand
  }
  if(i == 'vector') {
    stats=statsvec
  }
  if(i=='mini-em'){
    stats=statsmini
  }
  if(i=='kmeans'){
    stats=statsk
  }
  
  stats$time = as.numeric((stats$time)/itTest)
  stats$ariStdev = sd(stats$ari)
  stats$ari = mean(stats$ari)
  stats$ccrStdev = sd(stats$ccr)
  stats$ccr = mean(stats$ccr)

  filename = paste0('R',i,'.csv')
  
  write.csv(list(time=stats$time, cl=stats$cl, ccr=stats$ccr, ccrStdev=stats$ccrStdev, ari=stats$ari, ariStdev=stats$ariStdev), filename)
  
}

dataList = list(Random=statsrand$ari, Vector=statsvec$ari, MiniEM=statsmini$ari, Kmeans=statsk$ari)
boxplot(x=dataList, main='ARI for inits in R')