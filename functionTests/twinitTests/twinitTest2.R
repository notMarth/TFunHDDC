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
a[1,]=c(5e67, 1e55, 6e32)
a[2,]=c(7.5e23, 8.9e56, NA)
a[3,]=c(7e44, 1.2e43, NA)
a
par=list(K=3,nux=c(2,2,2),a=a, b=c(4e44,5e67,2.3e30), d=c(3,2,2),mu=mu,prop=c(1/3,1/3,1/3),
         Q=Q,Q1=Q1)
res = TFunHDDC:::.T_funhddt_twinit(data$fd,Wlist, par,par[['nux']])
