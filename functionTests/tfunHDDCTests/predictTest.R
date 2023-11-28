library(stringr)
library(TFunHDDC)
source("R/tfunHDDC.R")
if(getRversion() >= "2.15.1")  utils::globalVariables(c("poblenou"))

NOx = fitNOxBenchmark()

data = NOx$fd
labels = NOx$groupd


data( "poblenou", package = "fda.usc", envir = environment() )
nox_og <- poblenou$nox
training = nox_og$data[1:(nrow(nox_og)/2), ]
testing = nox_og$data[((nrow(nox_og)/2)+1):nrow(nox_og), ]

trainingLabels = labels[1:nrow(training)]
testingLabels = labels[(nrow(training)+1):length(labels)]

if(nrow(training) > nrow(testing)){
  training = training[1:nrow(testing), ]
  trainingLabels = trainingLabels[1:nrow(testing)]
} else{
  testing = testing[1:nrow(training), ]
  testingLabels = testingLabels[1:nrow(training)]
}

# smooth and fit data on functional basis
basis <- create.bspline.basis(rangeval = seq(0, 23, length.out=13),
                              nbasis = 15)
training <- smooth.basis(argvals = seq(0,23, length.out=24),
                       y = t( as.matrix(training) ),
                       fdParobj = basis)$fd

testing <- smooth.basis(argvals = seq(0,23, length.out=24),
                       y = t( as.matrix(testing) ),
                       fdParobj = basis)$fd

# training = NOx$fd
# training$coefs = training$coefs[, 1:(ncol(training$coefs)/2)]
# testing = NOx$fd
# testing$coefs = testing$coefs[, ((ncol(testing$coefs)/2)+1):ncol(testing$coefs)]

training = data
trainingLabels = labels
K=2
inits=c('random', 'vector', 'mini-em', 'kmeans')
models=c("AKJBKQKDK", "AKJBQKDK", 'AKBKQKDK', 'AKBQKDK', 'ABKQKDK', 'ABQKDK')
thresh = c(0.01, 0.1, 0.2, 0.4)
vec = rep(2, length(trainingLabels))
vec[1:(length(trainingLabels)/2)] = 1
kmc = list(iter.max=10, nstart=1, algorithm='Lloyd')
results = list()

for(i in inits){
  for(j in models){
    for(k in thresh){
      res = tfunHDDC(training, model = j, threshold=k, init=i, K=K, init.vector=vec, min.individuals=2, nb.rep = 30, kmeans.control = kmc)
      predict = TFunHDDC::predict.tfunHDDC(res, training)$class
      diff = res$class - predict
      print(diff)
      results[[paste0(i, j, k)]] = res$class
    }
  }
}

write.csv(results, 'RPredictTest.csv')
# print(res$class)
# print(res$nux)
# print(predict)
# print(trainingLabels)
# print(sum(diag(table(predict$class, trainingLabels)))/length(predict))
# print(TFunHDDC:::.T_hddc_ari(predict$class, trainingLabels))
# print(res$class - predict$class)