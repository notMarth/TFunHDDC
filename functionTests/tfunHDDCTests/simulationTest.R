library(TFunHDDC)

inits=c("random", 'vector', 'mini-em', 'kmeans')
threshold=c(0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
model=c("AKJBKQKDK", "AKJBQKDK", "AKBKQKDK", "AKBQKDK", "ABKQKDK", "ABQKDK")
K=3

data = genModelFD()
trainingdata=data$fd
labels = data$groupd

n=length(labels)
vec = rep(3, n)
vec[1:(n/3)] = 1
vec[((n/3)+1):(2*n/3)] = 2

results = list()

for(i in inits){
  for(j in model){
    for(k in threshold){
      print(paste0(i,'_',j,'_',k))
      res = TFunHDDC::tfunHDDC(trainingdata, K=K, threshold = k, model=j, init=i, init.vector = vec, nb.rep=30, min.individuals = 2)
      
      if(!inherits(res, t-funHDDC)){
        res[[paste0(i,'_',j,'_',k)]] = c(NA, NA)
        print(NA)
      }
      
      else{
        res[[paste0(i,'_',j,'_',k)]] = c(sum(diag(table(res$class, labels)))/length(labels),TFunHDDC:::.T_hddc_ari(res.class, labels))
        print(c(sum(diag(table(res$class, labels)))/length(labels),TFunHDDC:::.T_hddc_ari(res.class, labels)))
      }
    }
  }
}

filename = 'simTestResR.csv'
write.csv(results, filename)