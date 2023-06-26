library(TFunHDDC)
set.seed(1027)
#simulataed univariate data
data = genModelFD(ncurves=21, nsplines=21, alpha=c(0.9,0.9,0.9),
                  eta=c(10, 7, 17))
data
W=diag(rep(1,21))
W_m=diag(rep(1,21))
dety=1

Wlist<-list(W=W,
            W_m=W_m,
            dety=dety)
Wlist
K=3
Q<-vector(mode='list', length=3)
for (i in 1:K) Q[[i]]<-matrix(i*0.1,ncol=21, nrow=21)
Q
Q1<-vector(mode='list', length=3)
for (i in 1:K) Q1[[i]]<-matrix(i*0.1,ncol=21, nrow=21)
Q1
nsplines=21
mu=rbind(c(1, 0, 50, 100, rep(0, nsplines-4)),
         c(0, 0, 80, 0, 40, 2, rep(0, nsplines-6)),
         c(rep(0, nsplines-6), 20, 0, 80, 0, 0, 100))
mu
a=matrix(nrow=3,ncol=3,0)
a
a[1,]=c(5, 1, 6)
a[2,]=c(7.5, 8.9, NA)
a[3,]=c(7, 1.2, NA)
a
par=list(K=3,nux=c(2,2,2),a=a, b=c(4,5,2.3), d=c(3,2,2),mu=mu,prop=c(1/3,1/3,1/3),
         Q=Q,Q1=Q1)
res = TFunHDDC:::.T_funhddt_e_step1(data$fd,Wlist, par,clas=0,known=NULL, kno=NULL)

# if (!inherits(fdobj, 'list')) {x <- t(fdobj$coefs)}  # THIS IS the univariate CASE
# #else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x = cbind(x,t(fdobj[[i]]$coefs))} #NOT THIS
# p <- ncol(x)
# N <- nrow(x)
# K <- par$K
# nux <- par$nux
# a <- par$a
# b <- par$b
# mu <- par$mu
# d <- par$d
# prop <- par$prop
# Q <- par$Q
# Q1 <- par$Q1
# b[b<1e-6] <- 1e-6
# 
# if(clas>0){
#   unkno <- (kno-1)*(-1)
# }
# ##################################################
# #########################################################
# ###########################################
# t <- matrix(0, N, K)
# tw <- matrix(0, N, K)
# mah_pen <- matrix(0, N, K)
# K_pen <- matrix(0, N, K)
# num <- matrix(0, N, K)
# ft <- matrix(0, N, K)
# 
# s <- rep(0, K)
# 
# for (i in 1:K) { 
#   s[i] <- sum( log(a[i, 1:d[i]]) ) 
#   
#   Qk <- Q1[[i]]
#   aki <- sqrt( diag( c( 1 / a[i, 1:d[i]], rep(1 / b[i], p - d[i]) ) ) )
#   muki <- mu[i, ]
#   
#   Wki <- Wlist$W_m
#   dety<-Wlist$dety
#   
#   mah_pen[, i] <- TFunHDDC:::.T_imahalanobis(x, muki, Wki ,Qk, aki)
#   
#   tw[, i] <- (nux[i] + p) / (nux[i] + mah_pen[, i])
#   K_pen[, i] <- log(prop[i]) +
#     lgamma( (nux[i] + p) / 2) - (1 / 2) * (s[i] + (p - d[i]) * log(b[i]) -log(dety)) -
#     ( ( p / 2) * ( log(pi) + log(nux[i]) ) +
#         lgamma(nux[i] / 2) + ( (nux[i] + p) / 2 ) *
#         (log(1 + mah_pen[, i] / nux[i]) ) )
#   
# }
# ft <- exp(K_pen)
# #ft_den used for likelihood
# ft_den <- rowSums(ft)
# kcon <- -apply(K_pen,1,max)
# K_pen <- K_pen + kcon
# num <- exp(K_pen)
# t <- num / rowSums(num)
# 
# 
# L1 <- sum( log(ft_den) )
# L <- sum(log( rowSums( exp(K_pen) ) ) - kcon)
# 
# trow <- numeric(N)
# tcol <- numeric(K)
# trow <- rowSums(t)
# tcol <- colSums(t)
# if( any(tcol < p) ) { 
#   t <- (t + 0.0000001) / ( trow + (K * 0.0000001) )
#   
# }
# if(clas>0){
#   t <- unkno*t
#   for(i in 1:N){
#     if(kno[i]==1){
#       t[i, known[i]] <- 1
#     }
#   }
# }
# # if(any(is.nan(t))){
# #   break
# # }
# list(t = t,
#      tw = tw,
#      L = L)
# 
