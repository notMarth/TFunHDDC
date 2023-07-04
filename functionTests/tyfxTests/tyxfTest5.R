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
a[1,]=c(6, 7, 1)
a[2,]=c(6.8, 3, 9.8)
a[3,]=c(1.2, 4, 2.5)
a
par=list(K=3,nux=c(201,201,201),a=a, b=c(4e-33,4.4e-65,0.1), d=c(3,3,3),mu=mu,prop=c(1/3,1/3,1/3),
         Q=Q,Q1=Q1)
res = TFunHDDC:::.T_funhddt_e_step1(data$fd,Wlist, par,clas=0,known=NULL, kno=NULL)

.T_tyxf7 <- function(dfconstr, nux, n, t, tw, K, p, N){
  if(dfconstr=="no"){
    dfoldg <- nux
    for(i in 1:K){
      constn <- 1 + (1/n[i]) * sum(t[, i] * ( log(tw[, i]) - tw[, i]) ) +
        digamma( (dfoldg[i] + p) / 2 ) - log( (dfoldg[i] + p) / 2 )
      temp = digamma( (dfoldg[i] + p) / 2 )
      print("digamma 7")
      print(temp)
      #print("7 no")
      #print(constn)
      nux[i] <- uniroot( function(v) log(v / 2) - digamma(v / 2) + constn,
                         lower = 0.0001, upper = 1000, tol = 0.00001)$root
      if(nux[i] > 200){
        nux[i] <- 200
      }
      if(nux[i] < 2){
        nux[i] <- 2
      }
    }
  }
  else
  {
    dfoldg <- nux[1]
    constn <- 1 + (1/N) * sum( t * (log(tw) - tw) ) +
      digamma( (dfoldg + p) / 2 ) - log( (dfoldg + p) / 2 )
    
    #print("7 yes")
    #print(constn)
    dfsamenewg <- uniroot( function(v) log(v / 2) - digamma(v / 2) + constn,
                           lower = 0.0001, upper = 1000, tol = 0.01)$root
    if(dfsamenewg > 200){
      dfsamenewg <- 200
    }
    if(dfsamenewg < 2){
      dfsamenewg <- 2
    }
    nux <- c( rep(dfsamenewg, K) )
  }
  return(nux)
} # end of .T_tyxf7

.T_tyxf8 <- function(dfconstr, nux, n, t, tw, K, p, N){
  if(dfconstr == "no"){
    dfoldg <- nux
    for(i in 1:K){
      constn <- 1 + (1 / n[i]) * sum( t[, i] * (log(tw[, i]) - tw[,i]) ) +
        digamma( (dfoldg[i] + p) / 2 ) -
        log( (dfoldg[i]+p) / 2 )
      temp1 = (dfoldg[i] + p) / 2 
      temp = digamma( (dfoldg[i] + p) / 2 )
      print('dfoldg')
      print(dfoldg)
      constn <- -constn
      print("digamma 8")
      print(temp)
      #print("8 no")
      #print(constn)
      nux[i] <- (-exp(constn) + 2 * ( exp(constn)) *
                   ( exp( digamma(dfoldg[i] / 2)) -
                       ( (dfoldg[i]/2) - (1/2) ) ) ) / ( 1 - exp(constn) )
      if(nux[i] > 200){
        nux[i] <- 200
      }
      if(nux[i] < 2){
        nux[i] <- 2
      }
    }
  } else {
    dfoldg <- nux[1]
    constn <- 1 + (1 / N) * sum( t * (log(tw) - tw) ) +
      digamma( (dfoldg + p) / 2 ) - log( (dfoldg + p) / 2 )
    constn <- -constn
    #print("8 yes")
    #print(constn)
    dfsamenewg <- (-exp(constn) + 2 * ( exp(constn)) *
                     (exp( digamma(dfoldg / 2)) -
                        ( (dfoldg /2) - (1 / 2) ) ) ) / ( 1 - exp(constn) )
    if(dfsamenewg > 200){
      dfsamenewg <- 200
    }
    if(dfsamenewg < 2){
      dfsamenewg <- 2
    }
    nux <- c( rep(dfsamenewg, K) )
  }
  return(nux)
} # end of .T_tyxf8

test17 = .T_tyxf7('no', par[['nux']], colSums(res[['t']]), res[['t']], res[['tw']], 3, 21, 21)
test27 = TFunHDDC:::.T_tyxf7('yes', par[['nux']], colSums(res[['t']]), res[['t']], res[['tw']], 3, 21, 21)

test18 = .T_tyxf8('no', par[['nux']], colSums(res[['t']]), res[['t']], res[['tw']], 3, 21, 21)
test28 = TFunHDDC:::.T_tyxf8('yes', par[['nux']], colSums(res[['t']]), res[['t']], res[['tw']], 3, 21, 21)
