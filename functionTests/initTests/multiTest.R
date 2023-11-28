library(TFunHDDC)
source('../../W_multigen.R', chdir=T)
noise = 1e-8
graph= FALSE
threshold = 0.1
K = 3
d_set = rep(2, K)
d_max = 100
df_start = 50
methods = c("cattell", "bic", "grid")
models = c('AKJBKQKDK', 'AKBQKDK', 'ABQKDK')

data = TFunHDDC::genTriangles()$fd
vec = rep(3, ncol(data[[1]]$coefs))
vec[1:(length(vec)/3)] = 1
vec[((length(vec)/3) + 1):((2*length(vec)/3))] = 2

Wlist = W_multigen(data)

t = matrix(0, nrow=ncol(data[[1]]$coefs), ncol=K)

for (i in 1:K) t[which(vec == i), i] <- 1

n = colSums(t)
p = nrow(data[[1]]$coefs)
nux = rep(df_start, K)

for(model in models){
  for(method in methods){
    print(paste0('######', model, '#', method, '######'))
    print(TFunHDDC:::.T_funhddt_init(data, Wlist, K, t, nux, model, threshold, method, noise, NULL, d_max, d_set)$a)
  }
}